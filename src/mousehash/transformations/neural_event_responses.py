"""Aggregate Allen ophys event responses into one (n_neurons, n_images) view.

For every Allen ophys session showing `stimulus`, compute the per-(neuron,
image) probability of an L0 event in the stim window (binary mean across the
fixed `target_trials_per_image` trials), then concatenate across sessions on
the neuron axis. The result is an `OBSERVATION_BY_FEATURE` AnalysisView whose
matrix `event_probabilities.npy` is a drop-in input for `run_pca` / `run_nmf`
via `input_array_name="event_probabilities"`.

Cache layout under `<artifact_root>/neural_responses/<scope>/<spec_hash>/`:
    spec.json
    view.json
    summary.json
    event_probabilities.npy      # float32, shape (n_total_neurons, n_images), in [0, 1]
    neuron_index.json            # one entry per row, aligned with axis 0
    session_metadata.json        # one entry per kept session
    skipped_sessions.json        # one entry per dropped session
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from mousehash.artifacts.cache import ComputationSpec, cached_computation
from mousehash.artifacts.io import save_json, save_npy
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.manifests import RoleManifest

logger = logging.getLogger(__name__)


def _fingerprint_kept_sessions(kept: list[dict[str, Any]]) -> str:
    """sha1 over the sorted (session_id, n_cells) tuples of the kept sessions.

    Changes if Allen exposes new sessions, drops sessions, or reports a
    different cell count for a session — all of which should invalidate the
    cached aggregation.
    """
    h = hashlib.sha1()
    pairs = sorted((int(s["session_id"]), int(s["n_cells"])) for s in kept)
    for sid, n in pairs:
        h.update(f"{sid}:{n};".encode("utf-8"))
    return h.hexdigest()


def extract_event_response_view(
    manifest: RoleManifest,
    stimulus: str = "natural_scenes",
    n_images: int = 118,
    target_trials_per_image: int = 50,
    threshold: float = 0.0,
    session_limit: int | None = None,
    allen_manifest_path: Path | str | None = None,
    label: str | None = None,
) -> tuple[AnalysisView, dict[str, Any]]:
    """Build the per-dataset (n_total_neurons, n_images) event-probability view.

    Idempotent on the set of kept sessions + their cell counts + parameters:
    same inputs -> cache hit.
    """
    from mousehash.targets.allen.loaders import load_event_responses

    bundle = load_event_responses(
        stimulus=stimulus,
        n_images=n_images,
        target_trials_per_image=target_trials_per_image,
        threshold=threshold,
        allen_manifest_path=allen_manifest_path,
        session_limit=session_limit,
    )
    kept = bundle["kept"]
    skipped = bundle["skipped"]
    if not kept:
        raise ValueError(
            f"No Allen sessions passed strict-{target_trials_per_image}-trials filter "
            f"for {stimulus!r} (n_images={n_images}); {len(skipped)} sessions skipped."
        )

    input_fp = _fingerprint_kept_sessions(kept)

    spec = ComputationSpec(
        family="neural_responses",
        scope=manifest.dataset.dataset_id,
        name="event_probability",
        label=label,
        parameters={
            "stimulus": stimulus,
            "n_images": int(n_images),
            "target_trials_per_image": int(target_trials_per_image),
            "threshold": float(threshold),
            "session_limit": int(session_limit) if session_limit is not None else None,
        },
        input_fingerprints=[input_fp],
    )

    def _compute(out_dir: Path):
        sessions_sorted = sorted(kept, key=lambda s: int(s["session_id"]))

        matrices: list[np.ndarray] = []
        neuron_index: list[dict[str, Any]] = []
        session_metadata: list[dict[str, Any]] = []
        donor_ids: set[Any] = set()
        container_ids: set[int] = set()
        row_idx = 0
        for s in sessions_sorted:
            prob = np.asarray(s["prob_matrix"], dtype=np.float32)
            n_cells = int(s["n_cells"])
            if prob.shape != (n_cells, n_images):
                raise ValueError(
                    f"session {s['session_id']} prob_matrix shape {prob.shape} "
                    f"!= expected ({n_cells}, {n_images})"
                )
            matrices.append(prob)
            for cell_idx in range(n_cells):
                neuron_index.append({
                    "row_idx": row_idx,
                    "session_id": int(s["session_id"]),
                    "donor_id": s.get("donor_id"),
                    "experiment_container_id": int(s.get("experiment_container_id", -1)),
                    "cell_index_within_session": cell_idx,
                })
                row_idx += 1
            donor_ids.add(s.get("donor_id"))
            container_ids.add(int(s.get("experiment_container_id", -1)))
            session_metadata.append({
                "session_id": int(s["session_id"]),
                "experiment_container_id": int(s.get("experiment_container_id", -1)),
                "donor_id": s.get("donor_id"),
                "session_type": s.get("session_type", ""),
                "targeted_structure": s.get("targeted_structure", ""),
                "cre_line": s.get("cre_line", ""),
                "n_cells": n_cells,
            })

        event_probabilities = np.concatenate(matrices, axis=0).astype(np.float32)
        n_total_neurons = int(event_probabilities.shape[0])

        save_npy(out_dir / "event_probabilities.npy", event_probabilities)
        save_json(out_dir / "neuron_index.json", neuron_index)
        save_json(out_dir / "session_metadata.json", session_metadata)
        save_json(out_dir / "skipped_sessions.json", skipped)

        summary = {
            "stimulus": stimulus,
            "scope": manifest.dataset.dataset_id,
            "n_sessions_kept": len(sessions_sorted),
            "n_sessions_skipped": len(skipped),
            "n_donors": len({d for d in donor_ids if d is not None}),
            "n_containers": len(container_ids),
            "n_total_neurons": n_total_neurons,
            "n_images": int(n_images),
            "target_trials_per_image": int(target_trials_per_image),
            "threshold": float(threshold),
            "artifacts": {
                "event_probabilities": str(out_dir / "event_probabilities.npy"),
                "neuron_index": str(out_dir / "neuron_index.json"),
                "session_metadata": str(out_dir / "session_metadata.json"),
                "skipped_sessions": str(out_dir / "skipped_sessions.json"),
            },
        }

        view = AnalysisView.new(
            kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
            manifest_id=manifest.manifest_id,
            shape=[n_total_neurons, int(n_images)],
            axes={
                "observations": "neurons_across_sessions",
                "features": "natural_scene_images",
            },
            source_roles=["neural_data", "stimuli", "time_organization", "metadata"],
            transformation_lineage=[
                f"input:{input_fp[:12]}",
                f"stimulus:{stimulus}",
                f"event_probability:max>{threshold}",
                f"trials_per_image:{target_trials_per_image}",
                f"n_images:{n_images}",
            ],
            artifact_path=str(out_dir),
            summary={
                "stimulus": stimulus,
                "n_total_neurons": n_total_neurons,
                "n_images": int(n_images),
                "n_sessions_kept": len(sessions_sorted),
                "n_donors": summary["n_donors"],
            },
        )
        return view, summary

    view, summary, from_cache = cached_computation(spec, _compute)
    if from_cache:
        logger.info("Event-response cache hit (%s) -> %s", spec.hash(), view.artifact_path)
    summary["from_cache"] = from_cache
    return view, summary
