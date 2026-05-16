"""Materialize an Allen RoleManifest into a working RoleBundle for downstream tools.

The manifest carries identity + evidence; the bundle carries the actual
numpy arrays / DataFrames. This separation lets manifests be cheap to serve
over MCP while bundle materialization is opt-in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mousehash.core.manifests import RoleManifest
from mousehash.core.role_bundle import RoleBundle
from mousehash.targets.allen.client import get_brain_observatory_cache, resolve_manifest_path
from mousehash.targets.allen.manifest import load_image_catalog
from mousehash.targets.allen.stimuli import fetch_natural_scene_template

logger = logging.getLogger(__name__)


def load_natural_scene_image_stack(scene_set_id: str, manifest_path: Path | str | None = None) -> np.ndarray:
    """Return the full (n_images, H, W) raw template for downstream feature extraction.

    `manifest_path` here is the AllenSDK manifest, not the MouseHash manifest.
    The image stack is fetched fresh from AllenSDK so ViT runs on the canonical
    pixel values (the disk thumbnails are downscaled for display only).
    """
    return fetch_natural_scene_template(manifest_path)


def load_allen_role_bundle(manifest: RoleManifest, manifest_path: Path | str | None = None) -> dict[str, Any]:
    """Materialize the bundle dict for an Allen natural-scenes manifest.

    Returns a plain dict (not a `RoleBundle`) because the bundle's role-class
    fields hold evidence, not arrays. The dict shape is:

        {
            "manifest_id": str,
            "scene_set_id": str,
            "images": np.ndarray (n_images, H, W) uint-ish,
            "image_catalog": list[dict],    # per-image SHA1/path catalog
        }

    Roles themselves are read off `manifest.roles`.
    """
    scene_set_id = manifest.dataset.dataset_id
    catalog = load_image_catalog(scene_set_id)
    images = load_natural_scene_image_stack(scene_set_id, manifest_path=manifest_path)
    if len(images) != catalog["n_images"]:
        raise ValueError(
            f"Image stack / catalog mismatch for {scene_set_id}: "
            f"stack={len(images)} catalog={catalog['n_images']}"
        )
    return {
        "manifest_id": manifest.manifest_id,
        "scene_set_id": scene_set_id,
        "images": images,
        "image_catalog": catalog["images"],
        "roles": manifest.roles,
    }


def get_role_bundle_evidence(manifest: RoleManifest) -> RoleBundle:
    """Return the manifest's role evidence bundle (no array materialization)."""
    return manifest.roles


def load_event_responses(
    *,
    stimulus: str = "natural_scenes",
    n_images: int = 118,
    target_trials_per_image: int = 50,
    threshold: float = 0.0,
    allen_manifest_path: Path | str | None = None,
    session_limit: int | None = None,
) -> dict[str, Any]:
    """For every Allen ophys session showing `stimulus`, compute its
    per-(neuron, image) mean-binary event-probability matrix.

    Per trial: take the L0 event trace over the stim window
    `events[:, start:end]`, threshold its per-cell max, get a binary
    `(n_cells,)` indicator. Average those indicators across the first
    `target_trials_per_image` trials for each of the `n_images` images.

    Strict filter: sessions whose stim_table doesn't yield at least
    `target_trials_per_image` trials for every one of the `n_images`
    image indices are dropped into `skipped`.

    Returns:
        {
          "stimulus": str,
          "n_images": int,
          "target_trials_per_image": int,
          "threshold": float,
          "kept": [
            {
              "session_id": int,
              "experiment_container_id": int,
              "donor_id": int | str | None,
              "session_type": str,
              "targeted_structure": str,
              "cre_line": str,
              "n_cells": int,
              "prob_matrix": np.ndarray (n_cells, n_images) float32,   # in [0, 1]
            }, ...
          ],
          "skipped": [
            {"session_id": int, "reason": str, "trials_per_image": list[int]}
          ],
        }
    """
    from collections import defaultdict

    resolved = resolve_manifest_path(allen_manifest_path)
    boc = get_brain_observatory_cache(str(resolved))

    exps = boc.get_ophys_experiments(stimuli=[stimulus])
    exps = sorted(exps, key=lambda e: int(e["id"]))
    if session_limit is not None:
        exps = exps[: int(session_limit)]
    logger.info(
        "Loading event responses for %d %s sessions (n_images=%d, target_trials=%d)",
        len(exps), stimulus, n_images, target_trials_per_image,
    )

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for exp in exps:
        session_id = int(exp["id"])
        try:
            data = boc.get_ophys_experiment_data(session_id)
            events = boc.get_ophys_experiment_events(session_id)
            stim_table = data.get_stimulus_table(stimulus)
        except Exception as exc:  # AllenSDK exposes a variety of IO errors
            skipped.append({
                "session_id": session_id,
                "reason": f"load_failed:{type(exc).__name__}:{exc}",
                "trials_per_image": [],
            })
            continue

        events = np.asarray(events)
        n_cells = int(events.shape[0])

        frame_col, start_col, end_col = _stim_table_columns(stim_table)

        frame_trials: dict[int, list[np.ndarray]] = defaultdict(list)
        for frame_idx, start_t, end_t in zip(frame_col, start_col, end_col):
            if frame_idx == -1:
                continue
            if frame_idx < 0 or frame_idx >= n_images:
                continue
            if end_t <= start_t:
                trial_vector = np.zeros(n_cells, dtype=np.float32)
            else:
                window = events[:, int(start_t):int(end_t)]
                if window.size == 0:
                    trial_vector = np.zeros(n_cells, dtype=np.float32)
                else:
                    trial_vector = (window.max(axis=1) > threshold).astype(np.float32)
            frame_trials[int(frame_idx)].append(trial_vector)

        trials_per_image = [len(frame_trials.get(i, [])) for i in range(n_images)]
        if any(c < target_trials_per_image for c in trials_per_image):
            skipped.append({
                "session_id": session_id,
                "reason": f"insufficient_trials:min={min(trials_per_image) if trials_per_image else 0}",
                "trials_per_image": trials_per_image,
            })
            continue

        prob_matrix = np.zeros((n_cells, n_images), dtype=np.float32)
        for image_idx in range(n_images):
            trials = frame_trials[image_idx][:target_trials_per_image]
            prob_matrix[:, image_idx] = np.mean(np.stack(trials, axis=0), axis=0)

        kept.append({
            "session_id": session_id,
            "experiment_container_id": int(exp.get("experiment_container_id", -1)),
            "donor_id": exp.get("donor_id") or exp.get("donor_name"),
            "session_type": str(exp.get("session_type", "")),
            "targeted_structure": str(exp.get("targeted_structure", "")),
            "cre_line": str(exp.get("cre_line", "")),
            "n_cells": n_cells,
            "prob_matrix": prob_matrix,
        })

    return {
        "stimulus": stimulus,
        "n_images": n_images,
        "target_trials_per_image": target_trials_per_image,
        "threshold": float(threshold),
        "kept": kept,
        "skipped": skipped,
    }


def _stim_table_columns(stim_table: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull `frame`, `start`, `end` columns from a stim_table object as int64 arrays."""
    def _col(name: str) -> np.ndarray:
        col = stim_table[name]
        if hasattr(col, "to_numpy"):
            col = col.to_numpy()
        return np.asarray(col, dtype=np.int64)
    return _col("frame"), _col("start"), _col("end")


def load_stimulus_schedule(
    stimulus: str = "natural_scenes",
    *,
    allen_manifest_path: Path | str | None = None,
    session_limit: int | None = None,
) -> dict[str, Any]:
    """Pull per-session stim_tables for every Allen ophys session with `stimulus`.

    Returns:
        {
          "stimulus": str,
          "sessions": [
            {
              "session_id": int,
              "experiment_container_id": int,
              "donor_id": int | str | None,
              "session_type": str,
              "targeted_structure": str,
              "cre_line": str,
              "frame_sequence": np.ndarray[int64],   # one row per presentation
              "start_frames": np.ndarray[int64],
              "end_frames": np.ndarray[int64],
            },
            ...
          ],
        }

    `session_limit` is an internal knob for dev / tests — production callers
    should leave it as `None` to get every available session.
    """
    resolved = resolve_manifest_path(allen_manifest_path)
    boc = get_brain_observatory_cache(str(resolved))

    exps = boc.get_ophys_experiments(stimuli=[stimulus])
    exps = sorted(exps, key=lambda e: int(e["id"]))
    if session_limit is not None:
        exps = exps[: int(session_limit)]
    logger.info("Loading stim_table for %d %s sessions", len(exps), stimulus)

    sessions: list[dict[str, Any]] = []
    for exp in exps:
        session_id = int(exp["id"])
        data = boc.get_ophys_experiment_data(session_id)
        tbl = data.get_stimulus_table(stimulus)
        frames = np.asarray(tbl["frame"].to_numpy() if hasattr(tbl["frame"], "to_numpy") else tbl["frame"], dtype=np.int64)
        starts = np.asarray(tbl["start"].to_numpy() if hasattr(tbl["start"], "to_numpy") else tbl["start"], dtype=np.int64)
        ends = np.asarray(tbl["end"].to_numpy() if hasattr(tbl["end"], "to_numpy") else tbl["end"], dtype=np.int64)
        sessions.append(
            {
                "session_id": session_id,
                "experiment_container_id": int(exp.get("experiment_container_id", -1)),
                "donor_id": exp.get("donor_id") or exp.get("donor_name"),
                "session_type": str(exp.get("session_type", "")),
                "targeted_structure": str(exp.get("targeted_structure", "")),
                "cre_line": str(exp.get("cre_line", "")),
                "frame_sequence": frames,
                "start_frames": starts,
                "end_frames": ends,
            }
        )
    return {"stimulus": stimulus, "sessions": sessions}
