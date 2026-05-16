"""Stimulus-schedule analysis: trial periodicity + cross-session agreement.

Consumes a `PRESENTATION_TABLE` AnalysisView (produced by
`extract_stimulus_schedule_view`) and answers two questions:

  1. Within each session, are stimuli shown in the same order each trial?
     A "trial" = one full pass through all unique frames (50 passes of 118
     frames for Allen natural_scenes). We reshape the non-blank sequence into
     `n_blocks x block_size` and check whether every block is the same
     permutation of {0..block_size-1}.

  2. Is the scheduling the same for all animals?
     Pairwise per-position agreement (`fraction_identical`) between sessions,
     clustered into distinct schedules, then broken down by `donor_id`.

The HTML output is a plotly heatmap of the pairwise agreement, with sessions
optionally ordered by donor so groupings are visually obvious.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mousehash.artifacts.cache import ComputationSpec, cached_computation
from mousehash.artifacts.io import load_json, load_npy, save_json, save_npy
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.contracts import ToolContract

logger = logging.getLogger(__name__)


STIMULUS_SCHEDULE_CONTRACT = ToolContract(
    name="analyze_stimulus_schedule",
    family="scheduling",
    required_roles=["stimuli", "time_organization", "metadata"],
    consumes_views={"schedule": AnalysisViewKind.PRESENTATION_TABLE},
    produces=["metric_table", "figure", "summary"],
    allowed_transformations=["extract_stimulus_schedule_view"],
    default_validation=["block_permutation_check", "pairwise_fraction_identical"],
    assumptions=[
        "Presentation table has integer 'frame' column with blanks coded as -1.",
        "Block size equals the number of unique non-blank frames in the stimulus set.",
    ],
    failure_modes=[
        "Sequences with non-divisible lengths cannot be reshaped into blocks; flagged per session.",
        "Sessions whose adapters re-randomize per call would produce spurious cross-session disagreement.",
    ],
)


# ---------- Pure math ----------

def _analyze_one_session(
    session_id: int,
    seq: np.ndarray,
    *,
    drop_blanks: bool,
    block_size: int | None,
) -> dict[str, Any]:
    """Per-session block-permutation diagnostics."""
    seq = np.asarray(seq, dtype=np.int64)
    blanks_mask = seq == -1
    n_blanks = int(blanks_mask.sum())
    non_blank = seq[~blanks_mask] if drop_blanks else seq

    unique_non_blank = np.unique(seq[~blanks_mask]) if seq.size else np.array([], dtype=np.int64)
    n_unique = int(unique_non_blank.size)
    inferred_block = n_unique if block_size is None else int(block_size)

    result: dict[str, Any] = {
        "session_id": int(session_id),
        "n_presentations": int(seq.size),
        "n_blanks": n_blanks,
        "n_non_blank": int(non_blank.size),
        "n_unique_frames": n_unique,
        "block_size": inferred_block,
        "is_block_partitionable": False,
        "n_blocks": 0,
        "n_complete_blocks": 0,
        "block_completeness_fraction": 0.0,
        "within_session_block_order_identical": False,
        "n_distinct_block_orderings": 0,
    }

    if inferred_block <= 0 or non_blank.size == 0 or non_blank.size % inferred_block != 0:
        return result

    n_blocks = non_blank.size // inferred_block
    blocks = non_blank.reshape(n_blocks, inferred_block)
    expected_set = np.arange(inferred_block, dtype=np.int64)

    # A "complete" block is a permutation of {0..block_size-1}.
    complete_mask = np.fromiter(
        (np.array_equal(np.sort(blocks[i]), expected_set) for i in range(n_blocks)),
        dtype=bool,
        count=n_blocks,
    )
    n_complete = int(complete_mask.sum())

    # Distinct block orderings (raw row patterns).
    distinct_rows = {tuple(blocks[i].tolist()) for i in range(n_blocks)}

    result.update(
        {
            "is_block_partitionable": True,
            "n_blocks": int(n_blocks),
            "n_complete_blocks": n_complete,
            "block_completeness_fraction": float(n_complete) / float(n_blocks),
            "within_session_block_order_identical": len(distinct_rows) == 1,
            "n_distinct_block_orderings": len(distinct_rows),
        }
    )
    return result


def _pairwise_identical(seqs: list[np.ndarray]) -> np.ndarray:
    """Fraction of element-wise matches between every pair of sequences.

    Pairs are aligned by min length; the diagonal is set to 1.0.
    """
    n = len(seqs)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            a, b = seqs[i], seqs[j]
            m = min(a.size, b.size)
            if m == 0:
                frac = float("nan")
            else:
                frac = float(np.mean(a[:m] == b[:m]))
            mat[i, j] = frac
            mat[j, i] = frac
    return mat


def _cluster_schedules(pairwise: np.ndarray) -> list[list[int]]:
    """Group indices whose pairwise agreement equals 1.0 (transitive closure)."""
    n = pairwise.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if pairwise[i, j] == 1.0:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)
    # Sort groups largest-first for stable, agent-friendly output.
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def analyze_stimulus_schedules(
    session_ids: np.ndarray,
    frame_sequences: dict[str, np.ndarray],
    session_metadata: list[dict[str, Any]],
    *,
    drop_blanks: bool = True,
    block_size: int | None = None,
) -> dict[str, Any]:
    """Run the full within-session + cross-session schedule analysis.

    Args:
        session_ids: int array of session ids, sorted ascending. Order must
            match the index space of `pairwise_identical` in the output.
        frame_sequences: keyed by str(session_id) -> int64 presentation array.
        session_metadata: list aligned with `session_ids`, each entry has at
            least {donor_id, experiment_container_id}.
        drop_blanks: if True, drop `frame == -1` rows before block reshape.
        block_size: explicit block size; None => infer per session as the
            number of unique non-blank frames.

    Returns the full analysis dict described in the module docstring.
    """
    session_ids = np.asarray(session_ids, dtype=np.int64)
    n = int(session_ids.size)

    # ---- Per-session block diagnostics
    per_session: list[dict[str, Any]] = []
    for sid in session_ids:
        seq = np.asarray(frame_sequences[str(int(sid))], dtype=np.int64)
        per_session.append(
            _analyze_one_session(int(sid), seq, drop_blanks=drop_blanks, block_size=block_size)
        )

    n_block_partitionable = sum(1 for s in per_session if s["is_block_partitionable"])
    n_strictly_periodic = sum(1 for s in per_session if s["within_session_block_order_identical"])

    # ---- Cross-session pairwise agreement (post drop-blanks, for fair comparison)
    seqs_for_pairwise: list[np.ndarray] = []
    for sid in session_ids:
        seq = np.asarray(frame_sequences[str(int(sid))], dtype=np.int64)
        if drop_blanks:
            seq = seq[seq != -1]
        seqs_for_pairwise.append(seq)
    pairwise = _pairwise_identical(seqs_for_pairwise) if n > 0 else np.zeros((0, 0))

    if n >= 2:
        triu = pairwise[np.triu_indices(n, k=1)]
        # NaN-safe percentiles (NaN happens only when one of the sequences is empty).
        finite = triu[np.isfinite(triu)]
        pair_min = float(np.min(finite)) if finite.size else float("nan")
        pair_med = float(np.median(finite)) if finite.size else float("nan")
        pair_max = float(np.max(finite)) if finite.size else float("nan")
    else:
        pair_min = pair_med = pair_max = float("nan")

    clusters_idx = _cluster_schedules(pairwise) if n > 0 else []
    schedule_groups = [
        {
            "group_id": gid,
            "size": len(group),
            "session_ids": [int(session_ids[i]) for i in group],
        }
        for gid, group in enumerate(clusters_idx)
    ]

    # ---- Donor breakdown
    donor_by_idx: list[Any] = [m.get("donor_id") for m in session_metadata]
    donor_groups: dict[Any, list[int]] = {}
    for i, donor in enumerate(donor_by_idx):
        donor_groups.setdefault(donor, []).append(i)

    donor_breakdown: list[dict[str, Any]] = []
    within_donor_means: list[float] = []
    for donor, idxs in donor_groups.items():
        if len(idxs) < 2:
            within_mean = float("nan")
            within_all_identical = True
        else:
            sub = pairwise[np.ix_(idxs, idxs)]
            tri = sub[np.triu_indices(len(idxs), k=1)]
            finite = tri[np.isfinite(tri)]
            within_mean = float(np.mean(finite)) if finite.size else float("nan")
            within_all_identical = bool(np.all(finite == 1.0)) if finite.size else True
            within_donor_means.append(within_mean)
        donor_breakdown.append(
            {
                "donor_id": donor,
                "n_sessions": len(idxs),
                "session_ids": [int(session_ids[i]) for i in idxs],
                "within_donor_mean_agreement": within_mean,
                "within_donor_all_identical": within_all_identical,
            }
        )

    # Cross-donor: mean over pairs (i, j) with donor[i] != donor[j].
    cross_pairs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if donor_by_idx[i] != donor_by_idx[j]:
                v = pairwise[i, j]
                if np.isfinite(v):
                    cross_pairs.append(float(v))
    cross_donor_mean = float(np.mean(cross_pairs)) if cross_pairs else float("nan")
    within_donor_mean_overall = (
        float(np.mean(within_donor_means)) if within_donor_means else float("nan")
    )

    n_donors = len({d for d in donor_by_idx if d is not None})
    n_unique_schedules = len(schedule_groups)

    result: dict[str, Any] = {
        "n_sessions": n,
        "n_donors": n_donors,
        "drop_blanks": bool(drop_blanks),
        "block_size_override": int(block_size) if block_size is not None else None,
        "per_session": per_session,
        "within_session": {
            "n_block_partitionable": n_block_partitionable,
            "n_strictly_periodic": n_strictly_periodic,
            "all_strictly_periodic": (n > 0 and n_strictly_periodic == n),
        },
        "pairwise_agreement": {
            "min": pair_min,
            "median": pair_med,
            "max": pair_max,
        },
        "schedule_groups": schedule_groups,
        "n_unique_schedules": n_unique_schedules,
        "donor_breakdown": donor_breakdown,
        "within_donor_mean_agreement": within_donor_mean_overall,
        "cross_donor_mean_agreement": cross_donor_mean,
    }
    result["summary"] = interpret_schedule(result)
    return result


def interpret_schedule(result: dict[str, Any]) -> str:
    """Plain-English answer to the two user questions."""
    n = result["n_sessions"]
    d = result["n_donors"]
    if n == 0:
        return "No sessions available for the requested stimulus."

    within = result["within_session"]
    n_strict = within["n_strictly_periodic"]
    n_part = within["n_block_partitionable"]
    n_groups = result["n_unique_schedules"]
    within_mean = result["within_donor_mean_agreement"]
    cross_mean = result["cross_donor_mean_agreement"]

    parts: list[str] = []
    parts.append(f"Across {n} sessions from {d} donors:")
    if n_strict == n and n > 0:
        parts.append(
            f"• Same order each trial? YES in every session — all {n} sessions are strictly periodic "
            f"(every repetition-block is the same permutation)."
        )
    elif n_strict == 0:
        parts.append(
            f"• Same order each trial? NO — in 0/{n} sessions are the repetition-blocks identical. "
            f"{n_part}/{n} sessions partition cleanly into blocks of unique frames, but the per-block order varies (block-randomized, not periodic)."
        )
    else:
        parts.append(
            f"• Same order each trial? Mixed — {n_strict}/{n} sessions are strictly periodic; "
            f"the rest are block-randomized."
        )

    if n_groups == 1:
        parts.append(f"• Same schedule for all animals? YES — every session uses the same presentation order.")
    elif n_groups == n:
        parts.append(
            f"• Same schedule for all animals? NO — every session uses a distinct presentation order."
        )
    else:
        parts.append(
            f"• Same schedule for all animals? PARTIALLY — {n} sessions resolve into {n_groups} distinct presentation orders. "
            f"Median within-donor agreement = {within_mean:.2f}; cross-donor agreement = {cross_mean:.2f}."
        )
    return "\n".join(parts)


# ---------- Plot ----------

def make_schedule_heatmap(
    pairwise: np.ndarray,
    session_ids: list[int],
    donor_ids: list[Any],
    output_path: Path,
    title: str,
) -> Path:
    """Pairwise-agreement heatmap, sessions ordered by donor for clarity."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Schedule plots require plotly. Install with: pip install -e '.[viz]'"
        ) from exc

    n = pairwise.shape[0]
    # Order by donor (None goes last), then by session id within donor.
    order = sorted(
        range(n),
        key=lambda i: (donor_ids[i] is None, str(donor_ids[i]), int(session_ids[i])),
    )
    reordered = pairwise[np.ix_(order, order)]
    labels = [f"{session_ids[i]} (d={donor_ids[i]})" for i in order]

    fig = go.Figure(
        data=go.Heatmap(
            z=reordered,
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="fraction identical"),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(tickangle=-60),
        yaxis=dict(autorange="reversed"),
        width=900,
        height=900,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
    return output_path


# ---------- Cached MCP-ready wrapper ----------

def analyze_stimulus_schedule_view(
    schedule_view: AnalysisView,
    drop_blanks: bool = True,
    block_size: int | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Cached schedule analysis keyed on the source view's lineage hash."""
    if schedule_view.artifact_path is None:
        raise ValueError("schedule_view has no artifact_path; can't load frame_sequences.npz.")
    if schedule_view.kind != AnalysisViewKind.PRESENTATION_TABLE:
        from mousehash.core.errors import ViewKindMismatchError
        raise ViewKindMismatchError(
            expected=AnalysisViewKind.PRESENTATION_TABLE.value,
            got=schedule_view.kind.value,
            slot="schedule",
        )

    spec = ComputationSpec(
        family="schedule_comparisons",
        scope=schedule_view.lineage_hash,
        name="schedule_compare",
        parameters={
            "drop_blanks": bool(drop_blanks),
            "block_size": int(block_size) if block_size is not None else None,
            "plot": bool(plot),
        },
        input_fingerprints=[schedule_view.lineage_hash],
    )

    art_dir = Path(schedule_view.artifact_path)
    session_ids_path = art_dir / "session_ids.npy"
    sequences_path = art_dir / "frame_sequences.npz"
    metadata_path = art_dir / "session_metadata.json"

    def _compute(out_dir: Path):
        session_ids = load_npy(session_ids_path).astype(np.int64)
        with np.load(sequences_path) as npz:
            frame_sequences = {k: npz[k].astype(np.int64) for k in npz.files}
        session_metadata = load_json(metadata_path)

        result = analyze_stimulus_schedules(
            session_ids=session_ids,
            frame_sequences=frame_sequences,
            session_metadata=session_metadata,
            drop_blanks=drop_blanks,
            block_size=block_size,
        )

        # Re-derive the pairwise matrix for persistence (the analyze function returns
        # summary stats but not the full matrix — keeping it on disk keeps the view
        # composable for future downstream tools).
        seqs_for_pairwise: list[np.ndarray] = []
        for sid in session_ids:
            seq = np.asarray(frame_sequences[str(int(sid))], dtype=np.int64)
            if drop_blanks:
                seq = seq[seq != -1]
            seqs_for_pairwise.append(seq)
        pairwise = _pairwise_identical(seqs_for_pairwise)
        save_npy(out_dir / "pairwise_identical.npy", pairwise)

        result["input"] = {
            "schedule_view_id": schedule_view.view_id,
            "schedule_artifact_path": str(art_dir),
        }
        artifacts: dict[str, Any] = {
            "pairwise_identical": str(out_dir / "pairwise_identical.npy"),
        }
        if plot and session_ids.size >= 2:
            donor_ids = [m.get("donor_id") for m in session_metadata]
            plot_path = make_schedule_heatmap(
                pairwise=pairwise,
                session_ids=[int(s) for s in session_ids],
                donor_ids=donor_ids,
                output_path=out_dir / "schedule_heatmap.html",
                title=f"Stimulus schedule agreement ({schedule_view.view_id[:14]})",
            )
            artifacts["plot_html"] = str(plot_path)
        result["artifacts"] = artifacts
        save_json(out_dir / "analysis.json", result)

        view = AnalysisView.new(
            kind=AnalysisViewKind.METRIC_TABLE,
            manifest_id=schedule_view.manifest_id,
            shape=[int(session_ids.size), int(session_ids.size)],
            axes={"rows": "sessions", "cols": "sessions"},
            source_roles=schedule_view.source_roles,
            transformation_lineage=[
                f"schedule:{schedule_view.lineage_hash[:12]}",
                f"drop_blanks={int(bool(drop_blanks))}",
                "pairwise_identical",
                "block_permutation_check",
            ],
            artifact_path=str(out_dir),
            summary={
                "n_sessions": result["n_sessions"],
                "n_donors": result["n_donors"],
                "n_unique_schedules": result["n_unique_schedules"],
                "all_strictly_periodic": result["within_session"]["all_strictly_periodic"],
            },
        )
        return view, result

    view, summary, from_cache = cached_computation(spec, _compute)
    if from_cache:
        logger.info("Schedule compare cache hit (%s) -> %s", spec.hash(), view.artifact_path)
    summary["from_cache"] = from_cache
    summary["view_id"] = view.view_id
    summary["artifact_path"] = view.artifact_path
    return summary
