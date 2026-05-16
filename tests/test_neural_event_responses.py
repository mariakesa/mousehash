"""Tests for the neural_event_responses transformation.

AllenSDK is mocked end-to-end: a fake `BrainObservatoryCache` returns
deterministic per-session events + stim_tables. We use small `n_images=4`
and `target_trials_per_image=3` (12 presentations / session) so the
expected probabilities are easy to read by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mousehash.core.analysis_view import AnalysisViewKind
from mousehash.transformations.neural_event_responses import extract_event_response_view


N_IMAGES = 4
TARGET_TRIALS = 3
WINDOW = 5  # samples per trial; start = i*WINDOW, end = i*WINDOW + WINDOW


class _FakeDataset:
    def __init__(self, frame_sequence: np.ndarray) -> None:
        self._frames = frame_sequence

    def get_stimulus_table(self, stimulus: str) -> dict[str, np.ndarray]:
        n = self._frames.size
        starts = np.arange(n, dtype=np.int64) * WINDOW
        ends = starts + WINDOW
        return {"frame": self._frames, "start": starts, "end": ends}


class _FakeBOC:
    def __init__(self, sessions: list[dict[str, Any]]) -> None:
        self._sessions = sessions
        self._datasets = {int(s["id"]): _FakeDataset(s["_frames"]) for s in sessions}
        self._events = {int(s["id"]): s["_events"] for s in sessions}

    def get_ophys_experiments(self, stimuli=None):  # noqa: ARG002
        return [{k: v for k, v in s.items() if not k.startswith("_")} for s in self._sessions]

    def get_ophys_experiment_data(self, session_id: int) -> _FakeDataset:
        return self._datasets[int(session_id)]

    def get_ophys_experiment_events(self, session_id: int) -> np.ndarray:
        return self._events[int(session_id)]


def _balanced_frame_sequence(n_images: int, trials_per_image: int) -> np.ndarray:
    """Interleaved presentation: [0,1,...,n_images-1] repeated trials_per_image times."""
    return np.tile(np.arange(n_images, dtype=np.int64), trials_per_image)


def _events_one_neuron_per_image(n_cells: int, n_images: int, trials_per_image: int) -> np.ndarray:
    """Build an events array where neuron i fires only when image i is shown.

    Length = n_images * trials_per_image presentations, each WINDOW samples.
    Neuron `i` gets event amplitude 1.0 inside every presentation of image i
    and 0 elsewhere. n_cells must be >= n_images for this to make sense.
    """
    n_present = n_images * trials_per_image
    T = n_present * WINDOW
    events = np.zeros((n_cells, T), dtype=np.float32)
    frame_seq = _balanced_frame_sequence(n_images, trials_per_image)
    for present_idx, frame_idx in enumerate(frame_seq):
        if frame_idx < n_cells:
            events[frame_idx, present_idx * WINDOW : (present_idx + 1) * WINDOW] = 1.0
    return events


@pytest.fixture
def fake_allen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """2 sessions, 2 donors, each with 5 cells and a balanced 12-presentation schedule."""
    monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
    from mousehash.artifacts import paths as _paths
    _paths._load_dotenv_once.cache_clear()

    frames = _balanced_frame_sequence(N_IMAGES, TARGET_TRIALS)
    n_cells = 5
    events = _events_one_neuron_per_image(n_cells, N_IMAGES, TARGET_TRIALS)
    sessions = [
        {
            "id": 101, "experiment_container_id": 1, "donor_id": "A",
            "session_type": "three_session_B", "targeted_structure": "VISp",
            "cre_line": "Cux2-CreERT2",
            "_frames": frames, "_events": events,
        },
        {
            "id": 202, "experiment_container_id": 2, "donor_id": "B",
            "session_type": "three_session_B", "targeted_structure": "VISl",
            "cre_line": "Cux2-CreERT2",
            "_frames": frames, "_events": events,
        },
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
        MetadataRole, NeuralDataRole, RoleBundle, RoleConfidence, RoleEvidence, RoleStatus,
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
        neural_data=NeuralDataRole(
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


class TestExtractEventResponseView:
    def test_builds_observation_by_feature_view(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        view, summary = extract_event_response_view(
            manifest=manifest,
            n_images=N_IMAGES,
            target_trials_per_image=TARGET_TRIALS,
        )
        assert view.kind == AnalysisViewKind.OBSERVATION_BY_FEATURE
        assert view.shape == [10, N_IMAGES]  # 2 sessions × 5 cells
        assert view.axes == {
            "observations": "neurons_across_sessions",
            "features": "natural_scene_images",
        }
        assert "neural_data" in view.source_roles
        assert summary["n_sessions_kept"] == 2
        assert summary["n_sessions_skipped"] == 0
        assert summary["n_total_neurons"] == 10
        assert summary["n_donors"] == 2
        assert summary["from_cache"] is False

    def test_event_probabilities_correct(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        view, _ = extract_event_response_view(
            manifest=manifest, n_images=N_IMAGES, target_trials_per_image=TARGET_TRIALS,
        )
        probs = np.load(Path(view.artifact_path) / "event_probabilities.npy")
        # neuron i (i < N_IMAGES) fires only when image i is shown, on every trial.
        # That gives row i = one-hot at column i, value 1.0.
        for i in range(N_IMAGES):
            assert probs[i, i] == pytest.approx(1.0)
            other = np.delete(probs[i], i)
            assert np.all(other == 0.0)
        # row 4 is the "silent" 5th cell in session 101 -> all zeros.
        assert np.all(probs[4] == 0.0)
        # session 202 starts at row 5 and has the same pattern.
        for i in range(N_IMAGES):
            assert probs[5 + i, i] == pytest.approx(1.0)
        assert np.all(probs[9] == 0.0)

    def test_artifacts_written_to_disk(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        view, _ = extract_event_response_view(
            manifest=manifest, n_images=N_IMAGES, target_trials_per_image=TARGET_TRIALS,
        )
        art = Path(view.artifact_path)
        assert (art / "event_probabilities.npy").exists()
        assert (art / "neuron_index.json").exists()
        assert (art / "session_metadata.json").exists()
        assert (art / "skipped_sessions.json").exists()

    def test_neuron_index_aligned(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        view, _ = extract_event_response_view(
            manifest=manifest, n_images=N_IMAGES, target_trials_per_image=TARGET_TRIALS,
        )
        art = Path(view.artifact_path)
        index = json.loads((art / "neuron_index.json").read_text())
        probs = np.load(art / "event_probabilities.npy")
        assert len(index) == probs.shape[0]
        assert [r["row_idx"] for r in index] == list(range(10))
        # First 5 rows are session 101, next 5 are session 202.
        assert [r["session_id"] for r in index[:5]] == [101] * 5
        assert [r["session_id"] for r in index[5:]] == [202] * 5
        assert [r["cell_index_within_session"] for r in index[:5]] == list(range(5))

    def test_strict_filter_drops_short_sessions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()

        full_frames = _balanced_frame_sequence(N_IMAGES, TARGET_TRIALS)
        full_events = _events_one_neuron_per_image(5, N_IMAGES, TARGET_TRIALS)

        # Short session: only one trial per image (3 presentations instead of 12).
        short_frames = np.arange(N_IMAGES, dtype=np.int64)
        short_events = np.zeros((5, N_IMAGES * WINDOW), dtype=np.float32)

        sessions = [
            {
                "id": 101, "experiment_container_id": 1, "donor_id": "A",
                "session_type": "three_session_B", "targeted_structure": "VISp",
                "cre_line": "Cux2-CreERT2",
                "_frames": short_frames, "_events": short_events,
            },
            {
                "id": 202, "experiment_container_id": 2, "donor_id": "B",
                "session_type": "three_session_B", "targeted_structure": "VISl",
                "cre_line": "Cux2-CreERT2",
                "_frames": full_frames, "_events": full_events,
            },
        ]
        monkeypatch.setattr(
            "mousehash.targets.allen.loaders.get_brain_observatory_cache",
            lambda _path: _FakeBOC(sessions),
        )

        manifest = _make_manifest()
        view, summary = extract_event_response_view(
            manifest=manifest, n_images=N_IMAGES, target_trials_per_image=TARGET_TRIALS,
        )
        assert summary["n_sessions_kept"] == 1
        assert summary["n_sessions_skipped"] == 1
        assert view.shape == [5, N_IMAGES]
        skipped = json.loads((Path(view.artifact_path) / "skipped_sessions.json").read_text())
        assert skipped[0]["session_id"] == 101
        assert "insufficient_trials" in skipped[0]["reason"]

    def test_idempotent_cache_hit(self, fake_allen, data_root_tmp: Path):
        manifest = _make_manifest()
        v1, s1 = extract_event_response_view(
            manifest=manifest, n_images=N_IMAGES, target_trials_per_image=TARGET_TRIALS,
        )
        v2, s2 = extract_event_response_view(
            manifest=manifest, n_images=N_IMAGES, target_trials_per_image=TARGET_TRIALS,
        )
        assert s1["from_cache"] is False
        assert s2["from_cache"] is True
        assert v1.view_id == v2.view_id
        assert v1.artifact_path == v2.artifact_path

    def test_raises_when_no_sessions_kept(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()

        short_frames = np.arange(N_IMAGES, dtype=np.int64)
        short_events = np.zeros((5, N_IMAGES * WINDOW), dtype=np.float32)
        sessions = [
            {
                "id": 101, "experiment_container_id": 1, "donor_id": "A",
                "session_type": "three_session_B", "targeted_structure": "VISp",
                "cre_line": "Cux2-CreERT2",
                "_frames": short_frames, "_events": short_events,
            },
        ]
        monkeypatch.setattr(
            "mousehash.targets.allen.loaders.get_brain_observatory_cache",
            lambda _path: _FakeBOC(sessions),
        )
        manifest = _make_manifest()
        with pytest.raises(ValueError, match="No Allen sessions passed strict"):
            extract_event_response_view(
                manifest=manifest, n_images=N_IMAGES, target_trials_per_image=TARGET_TRIALS,
            )
