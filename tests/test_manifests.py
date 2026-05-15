"""Tests for core/manifests.py — RoleManifest construction + YAML round-trip."""

from __future__ import annotations

import pytest

from mousehash.core.ids import DatasetId, TargetName
from mousehash.core.manifests import DatasetRef, RoleManifest
from mousehash.core.role_bundle import (
    NeuralDataRole,
    RoleBundle,
    RoleConfidence,
    RoleEvidence,
    RoleStatus,
    StimuliRole,
)


def _stimuli_ds(scene_set_id: str = "abc") -> DatasetRef:
    return DatasetRef(
        target=TargetName("dandi"),
        dataset_id=DatasetId(scene_set_id),
        dataset_version="v1",
        label=f"test {scene_set_id}",
    )


class TestDatasetRef:
    def test_fingerprint_deterministic(self):
        a = _stimuli_ds()
        b = _stimuli_ds()
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_changes_with_id(self):
        assert _stimuli_ds("abc").fingerprint() != _stimuli_ds("xyz").fingerprint()

    def test_fingerprint_changes_with_version(self):
        a = DatasetRef(target=TargetName("dandi"), dataset_id=DatasetId("x"), dataset_version="v1")
        b = DatasetRef(target=TargetName("dandi"), dataset_id=DatasetId("x"), dataset_version="v2")
        assert a.fingerprint() != b.fingerprint()


class TestRoleManifestNew:
    def test_manifest_id_is_derived_from_dataset(self):
        ds = _stimuli_ds()
        mf = RoleManifest.new(dataset=ds)
        assert mf.manifest_id == f"mf_{ds.fingerprint()}"

    def test_same_dataset_yields_same_manifest_id(self):
        ds = _stimuli_ds()
        mf1 = RoleManifest.new(dataset=ds)
        mf2 = RoleManifest.new(dataset=ds)
        assert mf1.manifest_id == mf2.manifest_id

    def test_default_roles_are_empty_bundle(self):
        mf = RoleManifest.new(dataset=_stimuli_ds())
        assert mf.roles.satisfied_roles() == []

    def test_carries_roles_through(self):
        roles = RoleBundle(neural_data=NeuralDataRole(status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH))
        mf = RoleManifest.new(dataset=_stimuli_ds(), roles=roles)
        assert "neural_data" in mf.roles.satisfied_roles()

    def test_parser_version_default(self):
        mf = RoleManifest.new(dataset=_stimuli_ds())
        assert mf.parser_version == "0.1.0"

    def test_notes_preserved(self):
        mf = RoleManifest.new(dataset=_stimuli_ds(), notes="hand-curated")
        assert mf.notes == "hand-curated"


class TestRoleManifestYamlRoundTrip:
    def _full_manifest(self) -> RoleManifest:
        return RoleManifest.new(
            dataset=_stimuli_ds("rt_set"),
            roles=RoleBundle(
                neural_data=NeuralDataRole(
                    status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH,
                    evidence=[RoleEvidence(
                        status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH,
                        source="nwb", path="/units/spike_times",
                        notes="118 units across 4 probes",
                    )],
                ),
                stimuli=StimuliRole(
                    status=RoleStatus.LIKELY, confidence=RoleConfidence.MEDIUM,
                    evidence=[RoleEvidence(
                        status=RoleStatus.LIKELY, confidence=RoleConfidence.MEDIUM,
                        source="nwb", path="/stimulus/presentation",
                    )],
                ),
            ),
            notes="round-trip test",
        )

    def test_round_trip_preserves_manifest_id(self):
        mf = self._full_manifest()
        rebuilt = RoleManifest.from_yaml(mf.to_yaml())
        assert rebuilt.manifest_id == mf.manifest_id

    def test_round_trip_preserves_evidence(self):
        mf = self._full_manifest()
        rebuilt = RoleManifest.from_yaml(mf.to_yaml())
        ev = rebuilt.roles.neural_data.evidence[0]
        assert ev.source == "nwb"
        assert ev.path == "/units/spike_times"
        assert ev.status == RoleStatus.PRESENT

    def test_round_trip_preserves_status_and_confidence_enums(self):
        mf = self._full_manifest()
        rebuilt = RoleManifest.from_yaml(mf.to_yaml())
        assert rebuilt.roles.stimuli.status == RoleStatus.LIKELY
        assert rebuilt.roles.stimuli.confidence == RoleConfidence.MEDIUM

    def test_round_trip_preserves_dataset_ref(self):
        mf = self._full_manifest()
        rebuilt = RoleManifest.from_yaml(mf.to_yaml())
        assert rebuilt.dataset.dataset_id == mf.dataset.dataset_id
        assert rebuilt.dataset.dataset_version == mf.dataset.dataset_version

    def test_yaml_is_human_readable(self):
        text = self._full_manifest().to_yaml()
        # sanity: not JSON-on-one-line
        assert "\n" in text
        assert "manifest_id:" in text
        assert "roles:" in text
