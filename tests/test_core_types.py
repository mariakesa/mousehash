"""Tests for core/ids.py, errors.py, role_bundle.py, analysis_view.py, artifact.py."""

from __future__ import annotations

import pytest

from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.artifact import Artifact, ArtifactKind
from mousehash.core.errors import (
    ContractViolationError,
    ManifestParseError,
    MouseHashError,
    RoleMissingError,
    ViewKindMismatchError,
)
from mousehash.core.ids import ArtifactId, ManifestId, ToolRunId, stable_hash
from mousehash.core.role_bundle import (
    BehaviorRole,
    ConditionsRole,
    MetadataRole,
    NeuralDataRole,
    RoleBundle,
    RoleConfidence,
    RoleEvidence,
    RoleStatus,
    StimuliRole,
    TimeOrganizationRole,
)


# ---------- ids / stable_hash ----------

class TestStableHash:
    def test_deterministic(self):
        assert stable_hash({"a": 1}) == stable_hash({"a": 1})

    def test_order_invariant(self):
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})

    def test_different_inputs_different_hashes(self):
        assert stable_hash({"a": 1}) != stable_hash({"a": 2})

    def test_length_parameter(self):
        assert len(stable_hash({"x": 1}, length=8)) == 8
        assert len(stable_hash({"x": 1}, length=16)) == 16

    def test_handles_non_serializable_via_str(self):
        from pathlib import Path
        # Path is not JSON-serializable by default; default=str makes it work.
        h = stable_hash({"path": Path("/tmp/foo")})
        assert isinstance(h, str)


# ---------- errors ----------

class TestErrors:
    def test_all_subclass_mousehash_error(self):
        assert issubclass(RoleMissingError, MouseHashError)
        assert issubclass(ViewKindMismatchError, MouseHashError)
        assert issubclass(ContractViolationError, MouseHashError)
        assert issubclass(ManifestParseError, MouseHashError)

    def test_role_missing_message_with_tool(self):
        err = RoleMissingError("stimuli", tool_name="run_rsa")
        assert "stimuli" in str(err)
        assert "run_rsa" in str(err)

    def test_role_missing_message_without_tool(self):
        err = RoleMissingError("stimuli")
        assert "stimuli" in str(err)
        assert "for tool" not in str(err)

    def test_view_kind_mismatch_message(self):
        err = ViewKindMismatchError(expected="rdm", got="observation_by_neuron", slot="neural_rdm")
        msg = str(err)
        assert "rdm" in msg and "observation_by_neuron" in msg and "neural_rdm" in msg
        assert err.expected == "rdm"
        assert err.got == "observation_by_neuron"


# ---------- role bundle ----------

class TestRoleEvidence:
    def test_minimal_evidence(self):
        ev = RoleEvidence(status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH, source="nwb")
        assert ev.path is None and ev.notes is None

    def test_full_evidence(self):
        ev = RoleEvidence(
            status=RoleStatus.LIKELY, confidence=RoleConfidence.MEDIUM,
            source="nwb", path="/units/spike_times", notes="present but sparse",
        )
        assert ev.path == "/units/spike_times"


class TestRoleBundle:
    def test_empty_bundle_no_satisfied(self):
        bundle = RoleBundle()
        assert bundle.satisfied_roles() == []

    def test_present_roles_satisfied(self):
        bundle = RoleBundle(
            stimuli=StimuliRole(status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH),
            neural_data=NeuralDataRole(status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH),
        )
        assert set(bundle.satisfied_roles()) == {"stimuli", "neural_data"}

    @pytest.mark.parametrize("status,expected", [
        (RoleStatus.PRESENT, True),
        (RoleStatus.LIKELY, True),
        (RoleStatus.DERIVABLE, True),
        (RoleStatus.ABSENT, False),
        (RoleStatus.UNKNOWN, False),
    ])
    def test_is_satisfied_per_status(self, status, expected):
        role = StimuliRole(status=status, confidence=RoleConfidence.HIGH)
        assert role.is_satisfied() is expected

    def test_get_role_by_name(self):
        bundle = RoleBundle(behavior=BehaviorRole(status=RoleStatus.PRESENT, confidence=RoleConfidence.LOW))
        assert bundle.get("behavior").status == RoleStatus.PRESENT
        assert bundle.get("conditions").status == RoleStatus.UNKNOWN

    def test_all_six_roles_default_unknown(self):
        bundle = RoleBundle()
        for name in ["conditions", "stimuli", "behavior", "neural_data", "time_organization", "metadata"]:
            assert bundle.get(name).status == RoleStatus.UNKNOWN


# ---------- analysis view ----------

class TestAnalysisView:
    def _make(self, lineage):
        return AnalysisView.new(
            kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
            manifest_id=ManifestId("mf_test"),
            shape=[10, 1000],
            axes={"observations": "stimuli", "features": "imagenet"},
            source_roles=["stimuli"],
            transformation_lineage=lineage,
        )

    def test_deterministic_view_id(self):
        v1 = self._make(["a", "b"])
        v2 = self._make(["a", "b"])
        assert v1.view_id == v2.view_id
        assert v1.lineage_hash == v2.lineage_hash

    def test_different_lineage_different_hash(self):
        v1 = self._make(["a", "b"])
        v2 = self._make(["a", "c"])
        assert v1.view_id != v2.view_id

    def test_view_id_format(self):
        v = self._make(["x"])
        assert v.view_id.startswith("view_")
        assert v.lineage_hash in v.view_id

    def test_kinds_are_strings(self):
        # Pydantic v2 serializes Enum values as strings
        assert AnalysisViewKind.OBSERVATION_BY_NEURON.value == "observation_by_neuron"
        assert AnalysisViewKind.RDM.value == "rdm"


# ---------- artifact ----------

class TestArtifact:
    def test_minimal_artifact(self):
        art = Artifact(artifact_id=ArtifactId("art_1"), kind=ArtifactKind.MODEL, path="/tmp/x.pkl")
        assert art.tool_run_id is None
        assert art.content_hash is None
        assert art.summary == {}

    def test_summary_dict(self):
        art = Artifact(
            artifact_id=ArtifactId("art_2"), kind=ArtifactKind.TABLE,
            tool_run_id=ToolRunId("tr_42"), path="/tmp/y.parquet",
            summary={"rows": 100, "cols": 5},
        )
        assert art.summary == {"rows": 100, "cols": 5}

    def test_json_round_trip(self):
        art = Artifact(artifact_id=ArtifactId("art_3"), kind=ArtifactKind.HTML, path="/tmp/z.html")
        data = art.model_dump_json()
        rebuilt = Artifact.model_validate_json(data)
        assert rebuilt.artifact_id == art.artifact_id
        assert rebuilt.kind == art.kind
