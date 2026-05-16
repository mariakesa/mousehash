"""Tests for tools/comparison/group_comparison.py."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mousehash.tools.comparison import (
    GROUP_COMPARISON_CONTRACT,
    compare_groups_by_label,
    interpret_comparison,
    make_comparison_plot,
)


# ---------- pure math ----------

class TestCompareGroupsByLabel:
    def _identical_groups(self) -> dict:
        # Two groups drawn from identical distributions -> Cohen's d ≈ 0, p high.
        rng = np.random.default_rng(0)
        n = 60
        labels = (np.arange(n) % 2 == 0).astype(np.int8)  # 30 / 30
        features = rng.normal(loc=10.0, scale=2.0, size=(n, 3)).astype(np.float32)
        return compare_groups_by_label(
            features=features, labels=labels,
            label_names=("animate", "inanimate"),
            feature_names=["q10", "q50", "q90"],
            units="kB",
        )

    def test_identical_groups_small_effect(self):
        result = self._identical_groups()
        # Cohen's d magnitudes should be small (< ~0.5 typical for noise n=30+30)
        for f in result["per_feature"]:
            assert f["sufficient_samples"] is True
            assert abs(f["cohens_d"]) < 0.6

    def test_per_feature_keys_present(self):
        result = self._identical_groups()
        required = {
            "feature_name", "n_animate", "n_inanimate",
            "mean_animate", "mean_inanimate",
            "median_animate", "median_inanimate",
            "std_animate", "std_inanimate",
            "delta_mean", "delta_mean_pct",
            "cohens_d", "p_welch", "p_mannwhitneyu",
            "sufficient_samples",
        }
        for f in result["per_feature"]:
            assert required <= set(f), f"missing keys: {required - set(f)}"

    def test_known_large_effect_direction(self):
        rng = np.random.default_rng(0)
        n_a, n_b = 50, 50
        a = rng.normal(loc=20.0, scale=1.0, size=n_a).astype(np.float32)
        b = rng.normal(loc=10.0, scale=1.0, size=n_b).astype(np.float32)
        features = np.concatenate([a, b]).reshape(-1, 1)
        labels = np.concatenate([np.ones(n_a, dtype=np.int8), np.zeros(n_b, dtype=np.int8)])
        result = compare_groups_by_label(
            features=features, labels=labels,
            label_names=("hi", "lo"),
        )
        f = result["per_feature"][0]
        # Should detect "hi" larger with large positive Cohen's d and tiny p
        assert f["cohens_d"] > 5.0
        assert f["p_welch"] < 1e-20
        assert result["overall"]["direction"] == "hi_larger"

    def test_small_group_marks_insufficient(self):
        # One label has only 2 samples -> sufficient_samples False, NaN out tests
        features = np.array([[1.0], [2.0], [3.0], [10.0], [11.0]])
        labels = np.array([1, 1, 1, 0, 0], dtype=np.int8)  # 3 vs 2 -> insufficient
        result = compare_groups_by_label(features=features, labels=labels)
        f = result["per_feature"][0]
        assert f["sufficient_samples"] is False
        assert math.isnan(f["p_welch"])
        assert math.isnan(f["cohens_d"])

    def test_bonferroni_correction_caps_at_one(self):
        # Many features, large effect -> raw min_p tiny, Bonferroni stays <=1.
        rng = np.random.default_rng(1)
        k = 5
        a = rng.normal(loc=10.0, scale=1.0, size=(30, k))
        b = rng.normal(loc=15.0, scale=1.0, size=(30, k))
        features = np.vstack([a, b])
        labels = np.concatenate([np.ones(30), np.zeros(30)]).astype(np.int8)
        result = compare_groups_by_label(features=features, labels=labels)
        overall = result["overall"]
        assert 0 <= overall["min_p_welch_bonferroni"] <= 1.0
        assert overall["min_p_welch_bonferroni"] >= overall["min_p_welch"]

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="match"):
            compare_groups_by_label(
                features=np.zeros((5, 3)),
                labels=np.zeros(7, dtype=np.int8),
            )

    def test_rejects_non_2d_features(self):
        with pytest.raises(ValueError, match="2D"):
            compare_groups_by_label(features=np.zeros(10), labels=np.zeros(10))

    def test_feature_names_length_check(self):
        with pytest.raises(ValueError, match="feature_names"):
            compare_groups_by_label(
                features=np.zeros((4, 3)),
                labels=np.array([1, 1, 0, 0]),
                feature_names=["only", "two"],
            )

    def test_summary_string_present_and_non_empty(self):
        result = self._identical_groups()
        assert isinstance(result["summary"], str) and len(result["summary"]) > 20

    def test_summary_mentions_direction_for_clear_effect(self):
        rng = np.random.default_rng(0)
        n_a, n_b = 40, 40
        a = rng.normal(loc=20.0, scale=1.0, size=n_a)
        b = rng.normal(loc=10.0, scale=1.0, size=n_b)
        features = np.concatenate([a, b]).reshape(-1, 1)
        labels = np.concatenate([np.ones(n_a, dtype=np.int8), np.zeros(n_b, dtype=np.int8)])
        result = compare_groups_by_label(
            features=features, labels=labels,
            label_names=("animate", "inanimate"),
            feature_names=["q50"], units="kB",
            feature_axis="jpeg_quality_levels",
        )
        # animate (loc=20) larger than inanimate (loc=10)
        assert "animate" in result["summary"]
        assert "larger" in result["summary"]


# ---------- interpret_comparison standalone ----------

class TestInterpretComparison:
    def test_handles_zero_tests(self):
        # Mock a result where every feature is insufficient
        result = {
            "label_names": ["a", "b"],
            "units": "kB",
            "feature_axis": "qlevels",
            "per_feature": [{
                "feature_name": "x", "sufficient_samples": False,
                "delta_mean_pct": float("nan"), "cohens_d": float("nan"),
                "p_welch": float("nan"),
            }],
            "overall": {"n_tests": 0, "median_cohens_d": float("nan"),
                         "min_p_welch": float("nan"), "min_p_welch_bonferroni": float("nan"),
                         "direction": "mixed"},
        }
        s = interpret_comparison(result)
        assert "Insufficient" in s


# ---------- contract ----------

class TestContract:
    def test_contract_name_and_family(self):
        assert GROUP_COMPARISON_CONTRACT.name == "compare_groups_by_label"
        assert GROUP_COMPARISON_CONTRACT.family == "comparison"

    def test_contract_consumes_observation_by_feature(self):
        from mousehash.core.analysis_view import AnalysisViewKind
        assert GROUP_COMPARISON_CONTRACT.consumes_views["features"] == AnalysisViewKind.OBSERVATION_BY_FEATURE


# ---------- plot smoke test ----------

class TestMakeComparisonPlot:
    def test_writes_non_empty_html(self, tmp_path: Path):
        rng = np.random.default_rng(0)
        n_a, n_b = 20, 20
        a = rng.normal(loc=10.0, scale=2.0, size=(n_a, 3))
        b = rng.normal(loc=12.0, scale=2.0, size=(n_b, 3))
        features = np.vstack([a, b])
        labels = np.concatenate([np.ones(n_a, dtype=bool), np.zeros(n_b, dtype=bool)])
        out = make_comparison_plot(
            features=features,
            labels=labels,
            label_names=("animate", "inanimate"),
            feature_names=["q10", "q50", "q90"],
            units="kB",
            output_path=tmp_path / "plot.html",
            title="smoke",
        )
        assert out.exists() and out.stat().st_size > 1024
        text = out.read_text()
        assert "smoke" in text
        # plotly content marker
        assert "plotly" in text.lower()
