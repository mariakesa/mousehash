"""Tests for the pure schedule-analysis math.

Covers:
- Strictly periodic schedule (same permutation in every block).
- Block-randomized schedule (each block is a complete permutation, distinct).
- Identical schedules across sessions vs all-distinct vs donor-grouped.
- Non-divisible sequences flagged but not crashing.
- Blank handling.
"""

from __future__ import annotations

import numpy as np

from mousehash.tools.scheduling.schedule_comparison import (
    _pairwise_identical,
    analyze_stimulus_schedules,
    interpret_schedule,
)


def _periodic_session(order: np.ndarray, n_blocks: int) -> np.ndarray:
    return np.tile(order, n_blocks)


def _randomized_blocks(rng: np.random.Generator, block_size: int, n_blocks: int) -> np.ndarray:
    blocks = [rng.permutation(block_size) for _ in range(n_blocks)]
    return np.concatenate(blocks).astype(np.int64)


class TestAnalyzeOneSession:
    def test_strictly_periodic_session(self):
        rng = np.random.default_rng(0)
        order = rng.permutation(8).astype(np.int64)
        seq = _periodic_session(order, n_blocks=5)
        sequences = {"1": seq}
        result = analyze_stimulus_schedules(
            session_ids=np.array([1]),
            frame_sequences=sequences,
            session_metadata=[{"donor_id": "A"}],
        )
        s = result["per_session"][0]
        assert s["is_block_partitionable"] is True
        assert s["n_blocks"] == 5
        assert s["n_complete_blocks"] == 5
        assert s["block_completeness_fraction"] == 1.0
        assert s["within_session_block_order_identical"] is True
        assert s["n_distinct_block_orderings"] == 1

    def test_block_randomized_session(self):
        rng = np.random.default_rng(0)
        seq = _randomized_blocks(rng, block_size=8, n_blocks=5)
        result = analyze_stimulus_schedules(
            session_ids=np.array([1]),
            frame_sequences={"1": seq},
            session_metadata=[{"donor_id": "A"}],
        )
        s = result["per_session"][0]
        assert s["is_block_partitionable"] is True
        assert s["n_complete_blocks"] == 5
        # 5 random permutations of size 8 are overwhelmingly distinct.
        assert s["n_distinct_block_orderings"] >= 4
        assert s["within_session_block_order_identical"] is False

    def test_non_divisible_sequence_flagged(self):
        seq = np.array([0, 1, 2, 3, 4], dtype=np.int64)  # 5 frames, 5 unique -> block_size=5 -> 1 block
        # Use block_size override that does NOT divide evenly.
        result = analyze_stimulus_schedules(
            session_ids=np.array([1]),
            frame_sequences={"1": seq},
            session_metadata=[{"donor_id": "A"}],
            block_size=3,
        )
        s = result["per_session"][0]
        assert s["is_block_partitionable"] is False
        assert s["n_blocks"] == 0

    def test_blanks_dropped_by_default(self):
        rng = np.random.default_rng(0)
        order = rng.permutation(4).astype(np.int64)
        seq = _periodic_session(order, n_blocks=3)
        # Splice in blanks (frame == -1) at the start and middle.
        with_blanks = np.concatenate([np.array([-1, -1]), seq, np.array([-1])]).astype(np.int64)
        result = analyze_stimulus_schedules(
            session_ids=np.array([1]),
            frame_sequences={"1": with_blanks},
            session_metadata=[{"donor_id": "A"}],
        )
        s = result["per_session"][0]
        assert s["n_blanks"] == 3
        assert s["n_non_blank"] == 12
        assert s["within_session_block_order_identical"] is True


class TestPairwiseAgreement:
    def test_identical_sessions_have_1_0_off_diagonal(self):
        seq = np.array([0, 1, 2, 3], dtype=np.int64)
        mat = _pairwise_identical([seq, seq.copy(), seq.copy()])
        assert mat.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert mat[i, j] == 1.0

    def test_disjoint_sessions_have_zero_off_diagonal(self):
        a = np.array([0, 1, 2, 3], dtype=np.int64)
        b = np.array([3, 2, 1, 0], dtype=np.int64)
        mat = _pairwise_identical([a, b])
        assert mat[0, 1] == 0.0
        assert mat[1, 0] == 0.0
        assert mat[0, 0] == 1.0

    def test_partial_overlap_intermediate_value(self):
        a = np.array([0, 1, 2, 3], dtype=np.int64)
        b = np.array([0, 1, 9, 9], dtype=np.int64)
        mat = _pairwise_identical([a, b])
        assert 0.49 < mat[0, 1] < 0.51


class TestScheduleClustering:
    def test_all_sessions_identical_one_cluster(self):
        rng = np.random.default_rng(1)
        order = rng.permutation(8).astype(np.int64)
        seq = _periodic_session(order, n_blocks=4)
        sequences = {str(i): seq.copy() for i in range(3)}
        result = analyze_stimulus_schedules(
            session_ids=np.array([0, 1, 2]),
            frame_sequences=sequences,
            session_metadata=[{"donor_id": "A"}, {"donor_id": "B"}, {"donor_id": "C"}],
        )
        assert result["n_unique_schedules"] == 1
        assert result["schedule_groups"][0]["size"] == 3
        assert result["pairwise_agreement"]["min"] == 1.0
        assert result["within_session"]["all_strictly_periodic"] is True

    def test_all_sessions_distinct_n_clusters(self):
        rng = np.random.default_rng(2)
        # Each session has a totally different random schedule.
        sequences = {str(i): rng.permutation(40).astype(np.int64) for i in range(3)}
        result = analyze_stimulus_schedules(
            session_ids=np.array([0, 1, 2]),
            frame_sequences=sequences,
            session_metadata=[{"donor_id": "A"}, {"donor_id": "B"}, {"donor_id": "C"}],
        )
        assert result["n_unique_schedules"] == 3
        assert result["pairwise_agreement"]["max"] < 1.0

    def test_donor_grouped_schedules(self):
        rng = np.random.default_rng(3)
        order_a = rng.permutation(10).astype(np.int64)
        order_b = rng.permutation(10).astype(np.int64)
        seq_a = _periodic_session(order_a, n_blocks=3)
        seq_b = _periodic_session(order_b, n_blocks=3)
        sequences = {
            "1": seq_a, "2": seq_a.copy(),    # donor A: same schedule
            "3": seq_b, "4": seq_b.copy(),    # donor B: same schedule
        }
        session_ids = np.array([1, 2, 3, 4])
        metadata = [
            {"donor_id": "A"}, {"donor_id": "A"},
            {"donor_id": "B"}, {"donor_id": "B"},
        ]
        result = analyze_stimulus_schedules(
            session_ids=session_ids,
            frame_sequences=sequences,
            session_metadata=metadata,
        )
        assert result["n_unique_schedules"] == 2
        assert result["n_donors"] == 2
        # Within-donor: every pair within a donor is identical -> mean 1.0
        assert result["within_donor_mean_agreement"] == 1.0
        # Cross-donor: A's schedule != B's schedule -> < 1.0
        assert result["cross_donor_mean_agreement"] < 1.0
        donor_a = next(d for d in result["donor_breakdown"] if d["donor_id"] == "A")
        assert donor_a["within_donor_all_identical"] is True
        assert donor_a["n_sessions"] == 2


class TestInterpretSummary:
    def test_all_periodic_one_schedule_message(self):
        rng = np.random.default_rng(4)
        order = rng.permutation(8).astype(np.int64)
        seq = _periodic_session(order, n_blocks=3)
        sequences = {str(i): seq.copy() for i in range(2)}
        result = analyze_stimulus_schedules(
            session_ids=np.array([0, 1]),
            frame_sequences=sequences,
            session_metadata=[{"donor_id": "A"}, {"donor_id": "B"}],
        )
        summary = result["summary"]
        assert "Same order each trial? YES" in summary
        assert "Same schedule for all animals? YES" in summary

    def test_block_randomized_distinct_schedules_message(self):
        rng = np.random.default_rng(5)
        sequences = {str(i): _randomized_blocks(rng, block_size=8, n_blocks=4) for i in range(3)}
        result = analyze_stimulus_schedules(
            session_ids=np.array([0, 1, 2]),
            frame_sequences=sequences,
            session_metadata=[{"donor_id": "A"}, {"donor_id": "B"}, {"donor_id": "C"}],
        )
        summary = result["summary"]
        # Each block is a complete permutation but per-block order varies.
        assert "Same order each trial? NO" in summary
        assert "block-randomized" in summary
        # And every session has its own schedule.
        assert "Same schedule for all animals? NO" in summary

    def test_empty_returns_no_sessions_message(self):
        msg = interpret_schedule(
            {
                "n_sessions": 0,
                "n_donors": 0,
                "within_session": {
                    "n_block_partitionable": 0,
                    "n_strictly_periodic": 0,
                    "all_strictly_periodic": False,
                },
                "n_unique_schedules": 0,
                "within_donor_mean_agreement": float("nan"),
                "cross_donor_mean_agreement": float("nan"),
            }
        )
        assert "No sessions" in msg
