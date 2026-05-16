"""MCP wrappers for stimulus-schedule extraction + analysis.

Two tools that compose the same way `extract_jpeg_sizes` -> `compare_jpeg_*`
does for compressibility: one materializes the per-session presentation table,
the other answers the trial-order / cross-animal scheduling questions on top
of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mousehash.artifacts.paths import manifests_root
from mousehash.core.manifests import RoleManifest
from mousehash.mcp.errors import mcp_safe
from mousehash.mcp.views import find_view_by_id
from mousehash.tools.scheduling.schedule_comparison import analyze_stimulus_schedule_view
from mousehash.transformations.stimulus_schedule import extract_stimulus_schedule_view


def _load_manifest(manifest_id: str) -> RoleManifest:
    path = manifests_root() / f"{manifest_id}.yaml"
    if not path.exists():
        from mousehash.mcp.manifest_tools import ManifestNotFoundError
        raise ManifestNotFoundError(
            f"No manifest with id {manifest_id!r} under {manifests_root()}. "
            "Build one first via allen_build_manifest."
        )
    return RoleManifest.from_yaml(path.read_text(encoding="utf-8"))


@mcp_safe
def extract_stimulus_schedule(
    manifest_id: str,
    allen_manifest_path: str = "",
) -> dict[str, Any]:
    """Materialize per-session stim_tables for the manifest's natural-scenes stimulus.

    Pulls every Allen ophys session that includes `natural_scenes`, persists
    each session's `frame` / `start` / `end` arrays, and returns a
    `PRESENTATION_TABLE` AnalysisView. Idempotent: same set of available
    sessions -> cache hit.

    Args:
        manifest_id: role manifest id returned by `allen_build_manifest`.
        allen_manifest_path: AllenSDK manifest JSON path; empty -> env-resolved.

    Returns:
        {view_id, artifact_path, summary, from_cache}. `summary` includes
        `n_sessions`, `n_donors`, `n_containers`, `n_presentations_total`,
        and the artifact paths.
    """
    manifest = _load_manifest(manifest_id)
    view, summary = extract_stimulus_schedule_view(
        manifest=manifest,
        stimulus="natural_scenes",
        session_limit=None,
        allen_manifest_path=Path(allen_manifest_path).expanduser() if allen_manifest_path else None,
    )
    return {
        "view_id": view.view_id,
        "artifact_path": view.artifact_path,
        "summary": summary,
        "from_cache": summary.get("from_cache", False),
    }


@mcp_safe
def analyze_stimulus_schedule(
    schedule_view_id: str,
    drop_blanks: bool = True,
    block_size: int = 0,
    plot: bool = True,
) -> dict[str, Any]:
    """Answer "same order each trial? same schedule for all animals?" for a schedule view.

    Per-session block-permutation check (are all 50 repetition-blocks the same
    permutation of 118 frames?) + pairwise cross-session agreement + donor
    breakdown + plain-English summary. Optionally renders a plotly heatmap.

    Idempotent via `cached_computation`: same `(schedule_view_id, drop_blanks,
    block_size, plot)` -> cache hit.

    Args:
        schedule_view_id: view returned by `extract_stimulus_schedule`.
        drop_blanks: drop `frame == -1` rows before block reshape + pairwise.
        block_size: explicit block size; 0 => auto-infer per session as the
            number of unique non-blank frames (118 for natural_scenes).
        plot: render an interactive HTML heatmap under the cache dir.

    Returns:
        Full analysis dict — `per_session` block diagnostics, pairwise
        agreement stats, `schedule_groups` clusters, donor breakdown,
        plain-English `summary`, `artifacts`, `view_id`, `artifact_path`,
        `from_cache`.
    """
    schedule_view = find_view_by_id(schedule_view_id)
    return analyze_stimulus_schedule_view(
        schedule_view=schedule_view,
        drop_blanks=bool(drop_blanks),
        block_size=int(block_size) if block_size else None,
        plot=bool(plot),
    )
