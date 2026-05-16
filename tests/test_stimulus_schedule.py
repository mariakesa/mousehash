"""Tests for the stimulus_schedule transformation.

AllenSDK is mocked end-to-end: we install a fake `BrainObservatoryCache`
whose `get_ophys_experiments` + `get_ophys_experiment_data` return
deterministic per-session stim_tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mousehash.core.analysis_view import AnalysisViewKind
from mousehash.transformations.stimulus_schedule import extract_stimulus_schedule_view


class _FakeDataset:
    def __init__(self, frames: np.ndarray) -> None:
        self._frames = frames

    def get_stimulus_table(self, stimulus: str) -> dict[str, np.ndarray]:
        n = self._frames.size
        return {
            "frame": self._frames,
            "start": np.arange(n, dtype=np.int64) * 10,
            "end": np.arange(n, dtype=np.int64) * 10 + 7,
        }


class _FakeBOC:
    def __init__(self, sessions: list[dict[str, Any]]) -> None:
        self._sessions = sessions
        self._by_id = {int(s["id"]): _FakeDataset(s["_frames"]) for s in sessions}

    def get_ophys_experiments(self, stimuli=None):  # noqa: ARG002
        # Strip the private _frames key before returning experiment dicts.
        return [{k: v for k, v in s.items() if k != "_frames"} for s in self._sessions]

    def get_ophys_experiment_data(self, session_id: int) -> _FakeDataset:
        return self._by_id[int(session_id)]


def _periodic_session(order: np.ndarray, n_blocks: int) -> np.ndarray:
    return np.tile(order, n_blocks).astype(np.int64)


@pytest.fixture
def fake_allen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install a fake BOC with 4 sessions: 2 donors × 2 sessions each."""
    monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
    from mousehash.artifacts import paths as _paths
    _paths._load_dotenv_once.cache_clear()

    rng = np.random.default_rng(0)
    order_a = rng.permutation(6).astype(np.int64)
    order_b = rng.permutation(6).astype(np.int64)
    sessions = [
        {"id": 101, "experiment_container_id": 1, "donor_id": "A",
         "session_type": "three_session_B", "targeted_structure": "VISp",
         "cre_line": "Cux2-CreERT2", "_frames": _periodic_session(order_a, 3)},
        {"id": 102, "experiment_container_id": 1, "donor_id": "A",
         "session_type": "three_session_B", "targeted_structure": "VISp",
         "cre_line": "Cux2-CreERT2", "_frames": _periodic_session(order_a, 3)},
        {"id": 201, "experiment_container_id": 2, "donor_id": "B",
         "session_type": "three_session_B", "targeted_structure": "VISp",
         "cre_line": "Cux2-CreERT2", "_frames": _periodic_session(order_b, 3)},
        {"id": 202, "experiment_container_id": 2, "donor_id": "B",
         "session_type": "three_session_B", "targeted_structure": "VISl",
         "cre_line": "Cux2-CreERT2", "_frames": _periodic_session(order_b, 3)},
    ]
    fake_boc = _FakeBOC(sessions)
    monkeypatch.setattr(
        "mousehash.targets.allen.loaders.get_brain_observatory_cache",
        lambda _path: fake_boc,
    )
    return fake_boc


def _make_manifest():
    from mousehash.core.ids import DatasetId, TargetName
    from mousehash.core.manifests import DatasetRef, RoleManifest
    from mousehash.core.role_bundle import (
        MetadataRole, RoleBundle, RoleConfidence, RoleEvidence, RoleStatus,
        StimuliRole, TimeOrganizationRole,
    )

    roles = RoleBundle(
        stimuli=StimuliRole(
            status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH,
            evidence=[RoleEvidence(status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH, source="allensdk")],
        ),
        time_organization=TimeOrganizationRole(
            status=RoleStatus.DERIVABLE, confidence=RoleConfidence.HIGH,
            evidence=[RoleEvidence(status=RoleStatus.DERIVABLE, confidence=RoleConfidence.HIGH, source="allensdk")],
        ),
        metadata=MetadataRole(
            status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH,
            evidence=[RoleEvidence(status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH, source="allensdk")],
        ),
    )
    dataset = DatasetRef(
        target=TargetName("allen"),
        dataset_id=DatasetId("test_scene_set"),
        dataset_version="allen_brain_observatory",
        label="test",
    )
    return RoleManifest.new(dataset=dataset, roles=roles, parser_version="0.1.0")


class TestExtractStimulusScheduleView:
    def test_builds_presentation_table_view(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        view, summary = extract_stimulus_schedule_view(manifest=manifest)
        assert view.kind == AnalysisViewKind.PRESENTATION_TABLE
        assert view.shape == [4]
        assert summary["n_sessions"] == 4
        assert summary["n_donors"] == 2
        assert summary["n_containers"] == 2
        assert summary["from_cache"] is False

    def test_artifacts_written_to_disk(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        view, summary = extract_stimulus_schedule_view(manifest=manifest)
        art_dir = Path(view.artifact_path)
        assert (art_dir / "session_ids.npy").exists()
        assert (art_dir / "frame_sequences.npz").exists()
        assert (art_dir / "start_frames.npz").exists()
        assert (art_dir / "end_frames.npz").exists()
        assert (art_dir / "session_metadata.json").exists()

        ids = np.load(art_dir / "session_ids.npy")
        assert list(ids) == [101, 102, 201, 202]

        with np.load(art_dir / "frame_sequences.npz") as npz:
            keys = set(npz.files)
            assert keys == {"101", "102", "201", "202"}
            assert npz["101"].size == 18  # 6 frames × 3 blocks

    def test_session_metadata_aligned_and_full(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        view, _ = extract_stimulus_schedule_view(manifest=manifest)
        import json
        meta = json.loads((Path(view.artifact_path) / "session_metadata.json").read_text())
        assert [m["session_id"] for m in meta] == [101, 102, 201, 202]
        assert [m["donor_id"] for m in meta] == ["A", "A", "B", "B"]
        for m in meta:
            assert m["n_presentations"] == 18
            assert m["n_unique_frames"] == 6

    def test_idempotent_cache_hit(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        v1, s1 = extract_stimulus_schedule_view(manifest=manifest)
        v2, s2 = extract_stimulus_schedule_view(manifest=manifest)
        assert s1["from_cache"] is False
        assert s2["from_cache"] is True
        assert v1.view_id == v2.view_id
        assert v1.artifact_path == v2.artifact_path

    def test_raises_when_no_sessions(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        monkeypatch.setattr(
            "mousehash.targets.allen.loaders.get_brain_observatory_cache",
            lambda _path: _FakeBOC([]),
        )
        manifest = _make_manifest()
        with pytest.raises(ValueError, match="No Allen sessions"):
            extract_stimulus_schedule_view(manifest=manifest)

    def test_session_limit_truncates(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        view, summary = extract_stimulus_schedule_view(manifest=manifest, session_limit=2)
        assert summary["n_sessions"] == 2
        ids = np.load(Path(view.artifact_path) / "session_ids.npy")
        assert list(ids) == [101, 102]
