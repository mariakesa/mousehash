"""Round-trip and shape tests for the typed core in dandi_agent.models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mousehash.agents.dandi_agent.models import (
    CONFIDENCE_CAPS,
    ROLE_TAXONOMY,
    TOP_LEVEL_ROLES,
    AnalysisMove,
    AnalysisView,
    DerivationRecipe,
    EvidenceBackedRoleManifest,
    EvidenceItem,
    RoleEntry,
    RoleRequirement,
    ToolReadinessReport,
    ToolSpec,
    TransformationSpec,
)


def _evidence(role_path: str = "neural_data.spikes") -> EvidenceItem:
    return EvidenceItem(
        role_path=role_path,
        status="present",
        confidence=0.98,
        source="nwb",
        field="/units/spike_times",
        reason="Units table contains spike_times",
    )


def test_role_taxonomy_top_level_keys() -> None:
    assert TOP_LEVEL_ROLES == (
        "neural_data",
        "stimuli",
        "behavior",
        "conditions",
        "time_organization",
        "metadata",
    )
    # Every top-level role is in ROLE_TAXONOMY.
    assert set(TOP_LEVEL_ROLES) == set(ROLE_TAXONOMY)


def test_confidence_caps_sorted_correctly() -> None:
    assert CONFIDENCE_CAPS["nwb_direct_path"] > CONFIDENCE_CAPS["dandi_assets_summary"]
    assert CONFIDENCE_CAPS["dandi_assets_summary"] > CONFIDENCE_CAPS["paper_abstract_llm"]
    assert CONFIDENCE_CAPS["paper_abstract_llm"] > CONFIDENCE_CAPS["filename_hint"]


def test_evidence_item_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(
            role_path="x",
            status="present",
            confidence=1.5,  # out of bounds
            source="nwb",
            reason="bad",
        )


def test_manifest_roundtrip_via_json() -> None:
    m = EvidenceBackedRoleManifest(
        dandiset_id="000011",
        parser_version="0.1.0",
        created_at=datetime.now(timezone.utc),
        roles={
            "neural_data.spikes": RoleEntry(
                status="present",
                confidence=0.98,
                evidence=[_evidence()],
            )
        },
    )
    raw = m.model_dump_json()
    again = EvidenceBackedRoleManifest.model_validate_json(raw)
    assert again.status("neural_data.spikes") == "present"
    assert again.roles["neural_data.spikes"].evidence[0].field == "/units/spike_times"


def test_manifest_get_returns_blank_entry_for_unknown_role() -> None:
    m = EvidenceBackedRoleManifest(
        parser_version="0.1.0",
        created_at=datetime.now(timezone.utc),
    )
    entry = m.get("neural_data.calcium")
    assert entry.status == "unknown"
    assert entry.confidence == 0.0


def test_role_requirement_supports_any_of_groups() -> None:
    req = RoleRequirement(
        all_of=["neural_data", "time_organization"],
        any_of=[["conditions", "behavior"]],
    )
    raw = req.model_dump_json()
    again = RoleRequirement.model_validate_json(raw)
    assert again.all_of == ["neural_data", "time_organization"]
    assert again.any_of == [["conditions", "behavior"]]


def test_toolspec_roundtrip() -> None:
    t = ToolSpec(
        tool_id="psth",
        name="Generate PSTH Plot",
        workflow_family="visualization",
        requires_roles=RoleRequirement(all_of=["neural_data", "time_organization"]),
        requires_view="condition/trial x time bins -> firing rate",
        mvp_priority="mvp",
        has_visualization=True,
    )
    again = ToolSpec.model_validate_json(t.model_dump_json())
    assert again.mvp_priority == "mvp"
    assert again.has_visualization is True


def test_analysismove_question_is_optional() -> None:
    move = AnalysisMove(
        move_id="m",
        tool_id="t",
        tool_name="T",
        family="x",
        role_signature=RoleRequirement(),
        required_view="v",
    )
    assert move.question is None
    # Round-trip still works.
    AnalysisMove.model_validate_json(move.model_dump_json())


def test_tool_readiness_report_score_round_trips() -> None:
    r = ToolReadinessReport(
        tool_id="psth",
        tool_name="Generate PSTH Plot",
        status="ready",
        score=148.0,
        required_view="...",
    )
    r2 = ToolReadinessReport.model_validate_json(r.model_dump_json())
    assert r2.status == "ready"
    assert r2.score == 148.0


def test_transformation_and_view_models_accept_minimum_payload() -> None:
    tx = TransformationSpec(name="bin_spikes", family="binning", purpose="...")
    AnalysisView(view_id="v", view_type="observations_x_neurons")
    DerivationRecipe(name="rt")
    assert tx.leakage_risk == "low"
