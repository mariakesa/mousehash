"""End-to-end MCP test for the stimulus-schedule tools.

Chains: allen_build_manifest -> extract_stimulus_schedule -> analyze_stimulus_schedule.
AllenSDK is mocked: the natural-scene template stub feeds allen_build_manifest,
and a fake BrainObservatoryCache feeds extract_stimulus_schedule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mousehash.mcp.manifest_tools import list_runnable_tools
from mousehash.mcp.schedule_tools import analyze_stimulus_schedule, extract_stimulus_schedule
from mousehash.mcp.target_tools import allen_build_manifest


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
        return [{k: v for k, v in s.items() if k != "_frames"} for s in self._sessions]

    def get_ophys_experiment_data(self, session_id: int) -> _FakeDataset:
        return self._by_id[int(session_id)]


def _periodic(order: np.ndarray, n_blocks: int) -> np.ndarray:
    return np.tile(order, n_blocks).astype(np.int64)


def _bootstrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, all_identical: bool = True) -> str:
    """Mock Allen + BOC, build the manifest via MCP, return manifest_id."""
    monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
    from mousehash.artifacts import paths as _paths
    _paths._load_dotenv_once.cache_clear()

    rng = np.random.default_rng(0)
    stub_stack = rng.integers(0, 256, size=(8, 32, 32), dtype=np.uint16).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.targets.allen.manifest.fetch_natural_scene_template",
        lambda manifest_path=None: stub_stack,
    )

    # Build deterministic 6-frame schedules: either all sessions identical,
    # or per-donor identical with 2 donors of 2 sessions each.
    order_a = rng.permutation(6).astype(np.int64)
    if all_identical:
        sessions = [
            {"id": sid, "experiment_container_id": cid, "donor_id": donor,
             "session_type": "three_session_B", "targeted_structure": "VISp",
             "cre_line": "Cux2-CreERT2", "_frames": _periodic(order_a, 3)}
            for sid, cid, donor in [(101, 1, "A"), (102, 1, "A"), (201, 2, "B"), (202, 2, "B")]
        ]
    else:
        order_b = rng.permutation(6).astype(np.int64)
        sessions = [
            {"id": 101, "experiment_container_id": 1, "donor_id": "A",
             "session_type": "three_session_B", "targeted_structure": "VISp",
             "cre_line": "Cux2-CreERT2", "_frames": _periodic(order_a, 3)},
            {"id": 102, "experiment_container_id": 1, "donor_id": "A",
             "session_type": "three_session_B", "targeted_structure": "VISp",
             "cre_line": "Cux2-CreERT2", "_frames": _periodic(order_a, 3)},
            {"id": 201, "experiment_container_id": 2, "donor_id": "B",
             "session_type": "three_session_B", "targeted_structure": "VISp",
             "cre_line": "Cux2-CreERT2", "_frames": _periodic(order_b, 3)},
            {"id": 202, "experiment_container_id": 2, "donor_id": "B",
             "session_type": "three_session_B", "targeted_structure": "VISl",
             "cre_line": "Cux2-CreERT2", "_frames": _periodic(order_b, 3)},
        ]
    fake_boc = _FakeBOC(sessions)
    monkeypatch.setattr(
        "mousehash.targets.allen.loaders.get_brain_observatory_cache",
        lambda _path: fake_boc,
    )

    return allen_build_manifest(scene_set_id="sched_test")["manifest_id"]


class TestExtractStimulusSchedule:
    def test_returns_view_id_and_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap(monkeypatch, tmp_path)
        result = extract_stimulus_schedule(manifest_id=manifest_id)
        assert result["view_id"].startswith("view_")
        assert Path(result["artifact_path"]).exists()
        assert result["summary"]["n_sessions"] == 4
        assert result["summary"]["n_donors"] == 2
        assert result["from_cache"] is False

    def test_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap(monkeypatch, tmp_path)
        r1 = extract_stimulus_schedule(manifest_id=manifest_id)
        r2 = extract_stimulus_schedule(manifest_id=manifest_id)
        assert r1["from_cache"] is False
        assert r2["from_cache"] is True
        assert r1["view_id"] == r2["view_id"]

    def test_unknown_manifest_returns_structured_error(self, data_root_tmp: Path):
        result = extract_stimulus_schedule(manifest_id="mf_doesnotexist")
        assert result["type"] == "ManifestNotFoundError"


class TestAnalyzeStimulusSchedule:
    def test_all_identical_summary_says_yes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap(monkeypatch, tmp_path, all_identical=True)
        sched = extract_stimulus_schedule(manifest_id=manifest_id)
        result = analyze_stimulus_schedule(schedule_view_id=sched["view_id"])
        assert result["n_sessions"] == 4
        assert result["n_unique_schedules"] == 1
        assert "Same order each trial? YES" in result["summary"]
        assert "Same schedule for all animals? YES" in result["summary"]
        assert result["within_session"]["all_strictly_periodic"] is True

    def test_donor_grouped_summary_says_partial(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap(monkeypatch, tmp_path, all_identical=False)
        sched = extract_stimulus_schedule(manifest_id=manifest_id)
        result = analyze_stimulus_schedule(schedule_view_id=sched["view_id"])
        assert result["n_unique_schedules"] == 2
        # Two donors each on their own schedule -> "PARTIALLY" branch.
        assert "Same schedule for all animals? PARTIALLY" in result["summary"]
        # Within-donor: every donor's pair is identical.
        assert result["within_donor_mean_agreement"] == 1.0

    def test_plot_written_by_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap(monkeypatch, tmp_path)
        sched = extract_stimulus_schedule(manifest_id=manifest_id)
        result = analyze_stimulus_schedule(schedule_view_id=sched["view_id"])
        assert "plot_html" in result["artifacts"]
        plot_path = Path(result["artifacts"]["plot_html"])
        assert plot_path.exists() and plot_path.stat().st_size > 1024

    def test_plot_skipped_when_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap(monkeypatch, tmp_path)
        sched = extract_stimulus_schedule(manifest_id=manifest_id)
        result = analyze_stimulus_schedule(schedule_view_id=sched["view_id"], plot=False)
        assert "plot_html" not in result["artifacts"]

    def test_idempotent_cache_hit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap(monkeypatch, tmp_path)
        sched = extract_stimulus_schedule(manifest_id=manifest_id)
        r1 = analyze_stimulus_schedule(schedule_view_id=sched["view_id"])
        r2 = analyze_stimulus_schedule(schedule_view_id=sched["view_id"])
        assert r1["from_cache"] is False
        assert r2["from_cache"] is True
        assert r1["view_id"] == r2["view_id"]

    def test_unknown_view_id_returns_structured_error(self, data_root_tmp: Path):
        result = analyze_stimulus_schedule(schedule_view_id="view_nonexistent")
        assert "error" in result
        assert result["type"] == "ViewNotFoundError"


class TestReadinessRegistry:
    def test_analyze_stimulus_schedule_listed_for_allen_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap(monkeypatch, tmp_path)
        report = list_runnable_tools(manifest_id=manifest_id)
        tools = {r["tool_name"]: r for r in report["readiness"]}
        assert "analyze_stimulus_schedule" in tools
        # Allen manifest carries stimuli=PRESENT, time_organization=DERIVABLE,
        # metadata=PRESENT — all of which `is_satisfied()` treats as satisfied.
        assert tools["analyze_stimulus_schedule"]["runnable"] is True
