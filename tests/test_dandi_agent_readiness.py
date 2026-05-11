"""Readiness engine tests using synthetic manifests."""

from __future__ import annotations

from datetime import datetime, timezone

from mousehash.agents.dandi_agent.catalogs.loaders import load_tools
from mousehash.agents.dandi_agent.models import (
    EvidenceBackedRoleManifest,
    EvidenceItem,
    RoleEntry,
)
from mousehash.agents.dandi_agent.readiness import (
    build_analysis_move,
    compute_tool_readiness,
    rank_analysis_suggestions,
    suggest_analysis_moves,
)


def _entry(status="present", conf=0.95):
    return RoleEntry(
        status=status,
        confidence=conf,
        evidence=[
            EvidenceItem(
                role_path="x",
                status=status,
                confidence=conf,
                source="nwb",
                reason="fixture",
            )
        ],
    )


def _make_manifest(role_statuses: dict[str, str]) -> EvidenceBackedRoleManifest:
    return EvidenceBackedRoleManifest(
        parser_version="0.1.0",
        created_at=datetime.now(timezone.utc),
        roles={path: _entry(status=status) for path, status in role_statuses.items()},
    )


def test_psth_ready_with_spikes_and_trials() -> None:
    manifest = _make_manifest(
        {
            "neural_data.spikes": "present",
            "neural_data": "present",
            "time_organization.trials": "present",
            "time_organization": "present",
        }
    )
    psth = load_tools()["generate_psth_plot"]
    r = compute_tool_readiness(manifest, psth)
    assert r.status == "ready"
    assert "neural_data.spikes" in r.satisfied_roles
    assert r.score >= 100.0


def test_logistic_decoder_needs_either_conditions_or_behavior() -> None:
    # Has neural + time but no conditions/behavior — blocked.
    blocked_manifest = _make_manifest(
        {
            "neural_data": "present",
            "time_organization": "present",
        }
    )
    tool = load_tools()["fit_logistic_decoder"]
    r = compute_tool_readiness(blocked_manifest, tool)
    assert r.status == "blocked"
    assert r.missing_roles  # at least one missing

    # Add behavior — now ready.
    ok_manifest = _make_manifest(
        {
            "neural_data": "present",
            "time_organization": "present",
            "behavior.choices": "present",
            "behavior": "present",
        }
    )
    r2 = compute_tool_readiness(ok_manifest, tool)
    assert r2.status == "ready"


def test_derived_role_marks_tool_ready_after_transformation() -> None:
    manifest = _make_manifest(
        {
            "neural_data": "present",
            "time_organization": "present",
            "behavior.reaction_times": "derived_possible",
            "behavior": "derived_possible",
        }
    )
    tool = load_tools()["fit_logistic_decoder"]
    r = compute_tool_readiness(manifest, tool)
    assert r.status in ("ready_after_transformation", "ready")
    if r.status == "ready_after_transformation":
        assert r.derivable_roles


def test_likely_present_marks_tool_as_needs_confirmation() -> None:
    manifest = _make_manifest(
        {
            "neural_data.spikes": "likely_present",
            "neural_data": "likely_present",
            "time_organization": "present",
        }
    )
    psth = load_tools()["generate_psth_plot"]
    r = compute_tool_readiness(manifest, psth)
    assert r.status == "needs_confirmation"
    assert r.uncertain_roles


def test_rank_orders_ready_above_blocked() -> None:
    manifest = _make_manifest(
        {
            "neural_data.spikes": "present",
            "neural_data": "present",
            "time_organization.trials": "present",
            "time_organization": "present",
        }
    )
    reports = rank_analysis_suggestions(manifest)
    # Ready scores are non-negative; blocked scores are negative.
    ready = [r for r in reports if r.status == "ready"]
    blocked = [r for r in reports if r.status == "blocked"]
    if ready and blocked:
        assert min(r.score for r in ready) > max(r.score for r in blocked)


def test_suggest_returns_full_analysis_moves() -> None:
    manifest = _make_manifest(
        {
            "neural_data.spikes": "present",
            "neural_data": "present",
            "conditions": "present",
            "time_organization.trials": "present",
            "time_organization": "present",
        }
    )
    pairs = suggest_analysis_moves(manifest, top_k=3)
    assert pairs
    for report, move in pairs:
        assert move.tool_id == report.tool_id
        assert move.required_view == report.required_view
        assert move.transformation_plan  # non-empty
        # Trial-Averaged PCA / PSTH should be in the top moves for this manifest.
    tool_names = {move.tool_name for _, move in pairs}
    assert tool_names & {
        "Generate PSTH Plot",
        "Run Trial-Averaged PCA",
        "Generate Raster Plot",
    }


def test_build_analysis_move_includes_validation_plan() -> None:
    manifest = _make_manifest({"neural_data": "present", "time_organization": "present"})
    raster = load_tools()["generate_raster_plot"]
    move = build_analysis_move(manifest, raster)
    assert move.tool_id == "generate_raster_plot"
    assert move.validation_plan  # raster has at least one validation_check
    assert move.assumptions
