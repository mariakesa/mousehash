"""Parser orchestrator tests: real NWB -> EvidenceBackedRoleManifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousehash.agents.dandi_agent.models import (
    EvidenceBackedRoleManifest,
    EvidenceItem,
)
from mousehash.agents.dandi_agent.parser import (
    PARSER_VERSION,
    _normalize_role_path,
    infer_derived_roles_from_combinations,
    merge_evidence_into_manifest,
    parse_mousehash_roles,
    propagate_top_level_roles,
)

# Canonical fixture: 000011 ALM behavior+ecephys+ogen NWB.
GOLDEN_NWB = Path(
    "/home/maria/mousehash/src/DANDI-agent/dandi_nwb_cache/light_data/000011/"
    "0.220126.1907/sub-291064/sub-291064_ses-20150907_behavior+ecephys+ogen.nwb"
)


def test_normalize_role_path_maps_legacy_labels() -> None:
    assert (
        _normalize_role_path("stimuli", "optogenetic_intervention")
        == "stimuli.interventions.optogenetic"
    )
    assert (
        _normalize_role_path("behavior", "reaction_times_or_movement_timing")
        == "behavior.reaction_times"
    )
    # nwb_structure is internal bookkeeping; drop it.
    assert _normalize_role_path("nwb_structure", "units") is None


def test_merge_picks_strongest_status() -> None:
    items = [
        EvidenceItem(
            role_path="neural_data.spikes",
            status="likely_present",
            confidence=0.85,
            source="dandiset_metadata",
            reason="metadata mentions Units",
        ),
        EvidenceItem(
            role_path="neural_data.spikes",
            status="present",
            confidence=0.99,
            source="nwb",
            field="/units/spike_times",
            reason="Units table contains spike_times",
        ),
    ]
    m = merge_evidence_into_manifest(items, parser_version=PARSER_VERSION)
    assert m.status("neural_data.spikes") == "present"
    assert m.roles["neural_data.spikes"].confidence == pytest.approx(0.99)


def test_merge_records_source_coverage() -> None:
    items = [
        EvidenceItem(
            role_path="behavior.choices",
            status="present",
            confidence=0.95,
            source="nwb",
            reason="r1",
        ),
        EvidenceItem(
            role_path="behavior.choices",
            status="likely_present",
            confidence=0.7,
            source="dandiset_metadata",
            reason="r2",
        ),
    ]
    m = merge_evidence_into_manifest(items, parser_version=PARSER_VERSION)
    coverage = m.roles["behavior.choices"].source_coverage
    assert coverage["nwb"] == 1
    assert coverage["dandiset_metadata"] == 1


def test_derived_reaction_times_from_trial_columns() -> None:
    derived = infer_derived_roles_from_combinations(
        role_paths_present={"time_organization.trials"},
        trial_columns=["stimulus_onset_time", "response_time"],
    )
    assert any(d.role_path == "behavior.reaction_times" for d in derived)
    rt = next(d for d in derived if d.role_path == "behavior.reaction_times")
    assert rt.status == "derived_possible"
    assert rt.source == "derived"


def test_propagate_top_level_lifts_neural_data() -> None:
    m = merge_evidence_into_manifest(
        [
            EvidenceItem(
                role_path="neural_data.spikes",
                status="present",
                confidence=0.99,
                source="nwb",
                reason="r",
            ),
        ],
        parser_version=PARSER_VERSION,
    )
    # Top-level role inherits sub-role status.
    assert m.status("neural_data") == "present"


# ---------------------------------------------------------------------------
# Golden fixture: real NWB
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GOLDEN_NWB.exists(), reason="DANDI cache fixture missing")
def test_parse_real_alm_nwb_yields_expected_core_roles() -> None:
    m = parse_mousehash_roles(nwb_path=str(GOLDEN_NWB), dandiset_id="000011")
    assert isinstance(m, EvidenceBackedRoleManifest)
    assert m.parser_version == PARSER_VERSION

    # Core MouseHash claim: spikes + trials + opto perturbation must be detected
    # in this ALM behavior+ecephys+ogen dandiset.
    assert m.status("neural_data.spikes") == "present"
    assert m.status("time_organization.trials") == "present"
    assert m.status("stimuli.interventions.optogenetic") == "present"
    assert m.status("conditions.perturbation_labels") in (
        "present",
        "likely_present",
        "derived_possible",
    )

    # Reaction-time derivation should fire because the trials table contains
    # both response_time and stimulus-onset-like columns.
    rt_status = m.status("behavior.reaction_times")
    assert rt_status in ("present", "derived_possible", "likely_present")

    # Manifest serialization stable.
    again = EvidenceBackedRoleManifest.model_validate_json(m.model_dump_json())
    assert again.status("neural_data.spikes") == "present"
