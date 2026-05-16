"""Materialize per-session stimulus presentation tables as a cached AnalysisView.

For each Allen ophys session that contains the requested stimulus, pull
`dataset.get_stimulus_table(stimulus)` and persist the `frame` / `start` /
`end` columns. The resulting view (kind `PRESENTATION_TABLE`) is what the
schedule-comparison tool consumes to answer questions like "are stimuli shown
in the same order each trial, and across animals?".

Cache layout under `<artifact_root>/schedules/<scope>/<spec_hash>/`:
    spec.json
    view.json
    summary.json
    session_ids.npy              # int64 array, sorted
    session_metadata.json        # list aligned with session_ids
    frame_sequences.npz          # keys are str(session_id) -> int64 array
    start_frames.npz             # same keys, int64
    end_frames.npz               # same keys, int64
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


def _fingerprint_session_ids(session_ids: list[int]) -> str:
    """sha1 over the sorted session id list — invalidates the cache if Allen
    exposes a different set of sessions."""
    h = hashlib.sha1()
    for sid in sorted(int(s) for s in session_ids):
        h.update(str(sid).encode("utf-8"))
        h.update(b",")
    return h.hexdigest()


def extract_stimulus_schedule_view(
    manifest: RoleManifest,
    stimulus: str = "natural_scenes",
    session_limit: int | None = None,
    allen_manifest_path: Path | str | None = None,
    label: str | None = None,
) -> tuple[AnalysisView, dict[str, Any]]:
    """Build the per-session presentation table view for `stimulus`.

    Idempotent on the set of available sessions: if Allen exposes the same
    session ids, the second call is a cache hit.
    """
    from mousehash.targets.allen.loaders import load_stimulus_schedule

    bundle = load_stimulus_schedule(
        stimulus=stimulus,
        allen_manifest_path=allen_manifest_path,
        session_limit=session_limit,
    )
    sessions = bundle["sessions"]
    if not sessions:
        raise ValueError(f"No Allen sessions found for stimulus {stimulus!r}")

    session_ids = [int(s["session_id"]) for s in sessions]
    input_fp = _fingerprint_session_ids(session_ids)

    spec = ComputationSpec(
        family="schedules",
        scope=manifest.dataset.dataset_id,
        name="stim_table",
        label=label,
        parameters={
            "stimulus": stimulus,
            "session_limit": int(session_limit) if session_limit is not None else None,
        },
        input_fingerprints=[input_fp],
    )

    def _compute(out_dir: Path):
        ids_sorted = sorted(session_ids)
        ids_array = np.asarray(ids_sorted, dtype=np.int64)
        save_npy(out_dir / "session_ids.npy", ids_array)

        by_id = {int(s["session_id"]): s for s in sessions}
        frame_arrays: dict[str, np.ndarray] = {}
        start_arrays: dict[str, np.ndarray] = {}
        end_arrays: dict[str, np.ndarray] = {}
        session_metadata: list[dict[str, Any]] = []
        donor_ids: set[Any] = set()
        container_ids: set[int] = set()
        total_presentations = 0
        for sid in ids_sorted:
            s = by_id[sid]
            key = str(sid)
            seq = np.asarray(s["frame_sequence"], dtype=np.int64)
            frame_arrays[key] = seq
            start_arrays[key] = np.asarray(s["start_frames"], dtype=np.int64)
            end_arrays[key] = np.asarray(s["end_frames"], dtype=np.int64)
            non_blank = seq[seq >= 0]
            n_unique = int(np.unique(non_blank).size) if non_blank.size else 0
            n_blanks = int((seq == -1).sum())
            total_presentations += int(seq.size)
            donor_ids.add(s.get("donor_id"))
            container_ids.add(int(s.get("experiment_container_id", -1)))
            session_metadata.append(
                {
                    "session_id": sid,
                    "experiment_container_id": int(s.get("experiment_container_id", -1)),
                    "donor_id": s.get("donor_id"),
                    "session_type": s.get("session_type", ""),
                    "targeted_structure": s.get("targeted_structure", ""),
                    "cre_line": s.get("cre_line", ""),
                    "n_presentations": int(seq.size),
                    "n_non_blank_presentations": int(non_blank.size),
                    "n_blanks": n_blanks,
                    "n_unique_frames": n_unique,
                }
            )

        np.savez(out_dir / "frame_sequences.npz", **frame_arrays)
        np.savez(out_dir / "start_frames.npz", **start_arrays)
        np.savez(out_dir / "end_frames.npz", **end_arrays)
        save_json(out_dir / "session_metadata.json", session_metadata)

        summary = {
            "stimulus": stimulus,
            "scope": manifest.dataset.dataset_id,
            "n_sessions": len(ids_sorted),
            "n_donors": len({d for d in donor_ids if d is not None}),
            "n_containers": len(container_ids),
            "n_presentations_total": total_presentations,
            "session_limit": int(session_limit) if session_limit is not None else None,
            "artifacts": {
                "session_ids": str(out_dir / "session_ids.npy"),
                "session_metadata": str(out_dir / "session_metadata.json"),
                "frame_sequences": str(out_dir / "frame_sequences.npz"),
                "start_frames": str(out_dir / "start_frames.npz"),
                "end_frames": str(out_dir / "end_frames.npz"),
            },
        }

        view = AnalysisView.new(
            kind=AnalysisViewKind.PRESENTATION_TABLE,
            manifest_id=manifest.manifest_id,
            shape=[len(ids_sorted)],
            axes={"sessions": "ophys_sessions", "events": "stimulus_presentations"},
            source_roles=["stimuli", "time_organization", "metadata"],
            transformation_lineage=[
                f"input:{input_fp[:12]}",
                f"stimulus:{stimulus}",
                "get_stimulus_table",
            ],
            artifact_path=str(out_dir),
            summary={
                "stimulus": stimulus,
                "n_sessions": len(ids_sorted),
                "n_donors": summary["n_donors"],
                "n_containers": summary["n_containers"],
            },
        )
        return view, summary

    view, summary, from_cache = cached_computation(spec, _compute)
    if from_cache:
        logger.info("Schedule cache hit (%s) -> %s", spec.hash(), view.artifact_path)
    summary["from_cache"] = from_cache
    return view, summary
