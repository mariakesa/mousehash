"""Tests for core/contracts.py — readiness logic."""

from __future__ import annotations

import pytest

from mousehash.core.analysis_view import AnalysisViewKind
from mousehash.core.contracts import ToolContract, check_manifest_satisfies
from mousehash.core.ids import DatasetId, TargetName
from mousehash.core.manifests import DatasetRef, RoleManifest
from mousehash.core.role_bundle import (
    BehaviorRole,
    ConditionsRole,
    NeuralDataRole,
    RoleBundle,
    RoleConfidence,
    RoleStatus,
    StimuliRole,
    TimeOrganizationRole,
)


def _present(role_cls):
    return role_cls(status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH)


def _manifest_with(**role_kwargs) -> RoleManifest:
    return RoleManifest.new(
        dataset=DatasetRef(target=TargetName("dandi"), dataset_id=DatasetId("d1")),
        roles=RoleBundle(**role_kwargs),
    )


# Three reference contracts that exercise required / any_of / both
RSA_CONTRACT = ToolContract(
    name="run_rsa",
    family="geometry",
    required_roles=["neural_data"],
    any_of_roles=["stimuli", "conditions", "behavior"],
    consumes_views={"neural_rdm": AnalysisViewKind.RDM, "target_rdm": AnalysisViewKind.RDM},
    produces=["table", "figure"],
)

RIDGE_CONTRACT = ToolContract(
    name="fit_ridge_encoding_model",
    family="encoding",
    required_roles=["stimuli", "neural_data", "time_organization"],
    consumes_views={"X": AnalysisViewKind.OBSERVATION_BY_FEATURE, "Y": AnalysisViewKind.OBSERVATION_BY_NEURON},
    produces=["model"],
)

DECODER_CONTRACT = ToolContract(
    name="fit_logistic_decoder",
    family="decoding",
    required_roles=["neural_data", "behavior", "time_organization"],
    consumes_views={"X": AnalysisViewKind.OBSERVATION_BY_NEURON},
    produces=["model"],
)


class TestCheckManifestSatisfies:
    def test_runnable_when_required_present(self):
        mf = _manifest_with(
            stimuli=_present(StimuliRole),
            neural_data=_present(NeuralDataRole),
            time_organization=_present(TimeOrganizationRole),
        )
        readiness = check_manifest_satisfies(RIDGE_CONTRACT, mf)
        assert readiness.runnable is True
        assert readiness.missing_required_roles == []

    def test_missing_required_makes_unrunnable(self):
        mf = _manifest_with(neural_data=_present(NeuralDataRole), time_organization=_present(TimeOrganizationRole))
        readiness = check_manifest_satisfies(RIDGE_CONTRACT, mf)
        assert readiness.runnable is False
        assert "stimuli" in readiness.missing_required_roles

    def test_lists_all_missing_required(self):
        mf = _manifest_with(neural_data=_present(NeuralDataRole))
        readiness = check_manifest_satisfies(RIDGE_CONTRACT, mf)
        assert set(readiness.missing_required_roles) == {"stimuli", "time_organization"}

    def test_any_of_satisfied_by_one(self):
        # RSA needs neural_data + (stimuli OR conditions OR behavior). Just stimuli is enough.
        mf = _manifest_with(neural_data=_present(NeuralDataRole), stimuli=_present(StimuliRole))
        readiness = check_manifest_satisfies(RSA_CONTRACT, mf)
        assert readiness.runnable is True
        assert readiness.unsatisfied_any_of == []

    def test_any_of_satisfied_by_alternative(self):
        # behavior alone (instead of stimuli) is also enough
        mf = _manifest_with(neural_data=_present(NeuralDataRole), behavior=_present(BehaviorRole))
        readiness = check_manifest_satisfies(RSA_CONTRACT, mf)
        assert readiness.runnable is True

    def test_any_of_unsatisfied(self):
        # neural_data present but no stimuli/conditions/behavior
        mf = _manifest_with(neural_data=_present(NeuralDataRole))
        readiness = check_manifest_satisfies(RSA_CONTRACT, mf)
        assert readiness.runnable is False
        assert set(readiness.unsatisfied_any_of) == {"stimuli", "conditions", "behavior"}

    def test_both_missing_required_and_any_of(self):
        # Tool requiring neural_data AND any_of fails on both axes when neither present
        mf = _manifest_with()
        readiness = check_manifest_satisfies(RSA_CONTRACT, mf)
        assert readiness.runnable is False
        assert "neural_data" in readiness.missing_required_roles
        assert set(readiness.unsatisfied_any_of) == {"stimuli", "conditions", "behavior"}

    def test_satisfied_roles_listed_sorted(self):
        mf = _manifest_with(
            stimuli=_present(StimuliRole),
            neural_data=_present(NeuralDataRole),
            time_organization=_present(TimeOrganizationRole),
        )
        readiness = check_manifest_satisfies(RIDGE_CONTRACT, mf)
        assert readiness.satisfied_roles == sorted(readiness.satisfied_roles)
        assert set(readiness.satisfied_roles) >= {"stimuli", "neural_data", "time_organization"}

    def test_decoder_blocked_when_behavior_absent(self):
        mf = _manifest_with(
            neural_data=_present(NeuralDataRole),
            time_organization=_present(TimeOrganizationRole),
        )
        readiness = check_manifest_satisfies(DECODER_CONTRACT, mf)
        assert readiness.runnable is False
        assert "behavior" in readiness.missing_required_roles

    def test_non_manifest_input_raises(self):
        with pytest.raises(TypeError):
            check_manifest_satisfies(RSA_CONTRACT, {"not": "a manifest"})  # type: ignore[arg-type]

    def test_readiness_tool_name_propagated(self):
        mf = _manifest_with()
        readiness = check_manifest_satisfies(RSA_CONTRACT, mf)
        assert readiness.tool_name == "run_rsa"
