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
