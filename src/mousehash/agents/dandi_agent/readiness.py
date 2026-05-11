"""Tool-readiness engine and AnalysisMove builder.

Implements:
- ``compute_tool_readiness`` (mousehash_tools_spec.md §4)
- ``rank_analysis_suggestions`` (mousehash_tools_spec.md §8)
- ``suggest_analysis_moves`` (mousehash_transformations_spec.md §3)
"""

from __future__ import annotations

from typing import Iterable

from mousehash.agents.dandi_agent.catalogs.loaders import (
    load_tools,
    load_transformations,
)
from mousehash.agents.dandi_agent.models import (
    AnalysisMove,
    EvidenceBackedRoleManifest,
    ReadinessStatus,
    RoleRequirement,
    RoleStatus,
    ToolReadinessReport,
    ToolSpec,
    TransformationSpec,
)


def _resolve_status(manifest: EvidenceBackedRoleManifest, role_path: str) -> RoleStatus:
    """Return the best status for a role path or any of its dotted descendants.

    Tools often request a top-level role (``neural_data``); the manifest may
    only have populated a sub-role (``neural_data.spikes``). This helper lets
    a sub-role's status satisfy the parent.
    """
    if role_path in manifest.roles:
        return manifest.roles[role_path].status

    descendant_prefix = role_path + "."
    best: RoleStatus = "unknown"
    rank = {"unknown": 0, "absent": 1, "derived_possible": 2, "ambiguous": 3, "likely_present": 4, "present": 5}
    for path, entry in manifest.roles.items():
        if path.startswith(descendant_prefix) and rank[entry.status] > rank[best]:
            best = entry.status
    return best


def _bucket_role(
    manifest: EvidenceBackedRoleManifest,
    role_path: str,
    buckets: dict[str, list[str]],
) -> None:
    """Drop ``role_path`` into satisfied / uncertain / derivable / missing."""
    status = _resolve_status(manifest, role_path)
    if status == "present":
        buckets["satisfied"].append(role_path)
    elif status == "likely_present" or status == "ambiguous":
        buckets["uncertain"].append(role_path)
    elif status == "derived_possible":
        buckets["derivable"].append(role_path)
    else:
        buckets["missing"].append(role_path)


def _evaluate_any_of_group(
    manifest: EvidenceBackedRoleManifest,
    group: list[str],
) -> str:
    """Pick one role from an any_of group; prefer the strongest evidence."""
    rank = {"present": 5, "likely_present": 4, "ambiguous": 3, "derived_possible": 2, "absent": 1, "unknown": 0}
    return max(group, key=lambda r: rank[_resolve_status(manifest, r)])


def _classify_status(buckets: dict[str, list[str]]) -> ReadinessStatus:
    if not buckets["missing"] and not buckets["uncertain"] and not buckets["derivable"]:
        return "ready"
    if not buckets["missing"] and buckets["derivable"]:
        return "ready_after_transformation"
    if not buckets["missing"] and buckets["uncertain"]:
        return "needs_confirmation"
    return "blocked"


def compute_tool_readiness(
    manifest: EvidenceBackedRoleManifest,
    tool: ToolSpec,
) -> ToolReadinessReport:
    """Per-tool readiness verdict implementing mousehash_tools_spec.md §4."""
    buckets: dict[str, list[str]] = {
        "satisfied": [],
        "uncertain": [],
        "derivable": [],
        "missing": [],
    }

    for role in tool.requires_roles.all_of:
        _bucket_role(manifest, role, buckets)

    # An any_of group is satisfied if at least one member is non-missing.
    for group in tool.requires_roles.any_of:
        chosen = _evaluate_any_of_group(manifest, group)
        _bucket_role(manifest, chosen, buckets)

    status = _classify_status(buckets)

    optional_present = [
        r for r in tool.optional_roles
        if _resolve_status(manifest, r) in ("present", "likely_present", "derived_possible")
    ]

    suggested_transformations: list[str] = []
    if status in ("ready_after_transformation", "needs_confirmation", "ready"):
        suggested_transformations = list(tool.default_transformation_path)

    return ToolReadinessReport(
        tool_id=tool.tool_id,
        tool_name=tool.name,
        status=status,
        score=_score_report(status, buckets, optional_present, tool),
        satisfied_roles=buckets["satisfied"],
        uncertain_roles=buckets["uncertain"],
        derivable_roles=buckets["derivable"],
        missing_roles=buckets["missing"],
        optional_roles_present=optional_present,
        required_view=tool.requires_view,
        suggested_transformations=suggested_transformations,
        rationale=_explain(status, buckets, tool),
    )


def _score_report(
    status: ReadinessStatus,
    buckets: dict[str, list[str]],
    optional_present: list[str],
    tool: ToolSpec,
) -> float:
    """Scoring rules from mousehash_tools_spec.md §8."""
    base = {
        "ready": 100.0,
        "ready_after_transformation": 80.0,
        "needs_confirmation": 50.0,
        "blocked": -100.0,
        "not_recommended": -50.0,
    }[status]

    score = base
    score += 5.0 * len(buckets["satisfied"])
    score += 1.0 * len(optional_present)
    if tool.mvp_priority == "mvp":
        score += 20.0
    if tool.has_visualization:
        score += 10.0
    if tool.leakage_risk == "high" and not tool.validation_checks:
        score -= 30.0
    return round(score, 3)


def _explain(
    status: ReadinessStatus,
    buckets: dict[str, list[str]],
    tool: ToolSpec,
) -> str:
    if status == "ready":
        return f"All required roles present for {tool.name}."
    if status == "ready_after_transformation":
        return (
            f"{tool.name} can run after deriving: " + ", ".join(buckets["derivable"]) + "."
        )
    if status == "needs_confirmation":
        return (
            f"{tool.name} likely runnable but needs human confirmation for: "
            + ", ".join(buckets["uncertain"])
            + "."
        )
    return f"{tool.name} blocked: missing " + ", ".join(buckets["missing"]) + "."


# ---------------------------------------------------------------------------
# Whole-catalog suggestions
# ---------------------------------------------------------------------------

def rank_analysis_suggestions(
    manifest: EvidenceBackedRoleManifest,
    tools: Iterable[ToolSpec] | None = None,
) -> list[ToolReadinessReport]:
    """Compute readiness for every tool in the catalog and rank by score."""
    catalog = list(tools) if tools is not None else list(load_tools().values())
    reports = [compute_tool_readiness(manifest, t) for t in catalog]
    reports.sort(key=lambda r: r.score, reverse=True)
    return reports


def build_analysis_move(
    manifest: EvidenceBackedRoleManifest,
    tool: ToolSpec,
    transformation_catalog: dict[str, TransformationSpec] | None = None,
) -> AnalysisMove:
    """Compose the full AnalysisMove for a tool (mousehash_transformations_spec.md §37)."""
    if transformation_catalog is None:
        transformation_catalog = load_transformations()

    plan = []
    for name in tool.default_transformation_path:
        if name in transformation_catalog:
            plan.append(name)
        else:
            plan.append(name)  # keep as a hint even if not catalogued

    return AnalysisMove(
        move_id=f"{tool.tool_id}__default",
        tool_id=tool.tool_id,
        tool_name=tool.name,
        family=tool.workflow_family,
        role_signature=tool.requires_roles,
        optional_roles=list(tool.optional_roles),
        transformation_plan=plan,
        required_view=tool.requires_view,
        validation_plan=list(tool.validation_checks),
        artifacts=list(tool.output_artifacts),
        assumptions=list(tool.assumptions),
        failure_modes=list(tool.failure_modes),
        question=tool.question,
    )


def suggest_analysis_moves(
    manifest: EvidenceBackedRoleManifest,
    top_k: int | None = None,
    status_filter: Iterable[ReadinessStatus] | None = None,
) -> list[tuple[ToolReadinessReport, AnalysisMove]]:
    """Return ranked (report, move) pairs.

    ``status_filter`` defaults to non-blocked statuses. ``top_k=None`` returns
    everything.
    """
    if status_filter is None:
        status_filter = ("ready", "ready_after_transformation", "needs_confirmation")
    allowed = set(status_filter)

    tools_by_id = load_tools()
    tx_catalog = load_transformations()
    reports = rank_analysis_suggestions(manifest, tools_by_id.values())

    pairs: list[tuple[ToolReadinessReport, AnalysisMove]] = []
    for r in reports:
        if r.status not in allowed:
            continue
        move = build_analysis_move(manifest, tools_by_id[r.tool_id], tx_catalog)
        pairs.append((r, move))

    if top_k is not None:
        pairs = pairs[:top_k]
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Rank MouseHash tools for an EvidenceBackedRoleManifest JSON file.",
    )
    ap.add_argument("manifest", type=Path, help="Path to a manifest JSON.")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--include-blocked", action="store_true")
    args = ap.parse_args()

    manifest = EvidenceBackedRoleManifest.model_validate_json(args.manifest.read_text())
    status_filter = None
    if args.include_blocked:
        status_filter = ("ready", "ready_after_transformation", "needs_confirmation", "blocked")

    pairs = suggest_analysis_moves(manifest, top_k=args.top, status_filter=status_filter)
    for report, move in pairs:
        print(
            f"  {report.score:>7.1f}  {report.status:<28} {report.tool_id:<36} {report.tool_name}"
        )
        if report.derivable_roles:
            print(f"            derivable: {report.derivable_roles}")
        if report.uncertain_roles:
            print(f"            uncertain: {report.uncertain_roles}")
        if report.missing_roles:
            print(f"            missing:   {report.missing_roles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
