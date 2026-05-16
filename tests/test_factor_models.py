"""Tests for tools/factor_models/{pca,nmf}.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.artifacts.io import save_npy
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.errors import ViewKindMismatchError
from mousehash.core.ids import ManifestId
from mousehash.tools.factor_models.nmf import (
    NMF_CONTRACT,
    apply_probability_temperature,
    run_nmf,
)
from mousehash.tools.factor_models.pca import PCA_CONTRACT, run_pca


# ---------- view fixtures ----------

def _make_feature_view(tmp_path: Path, logits: np.ndarray, probabilities: np.ndarray) -> AnalysisView:
    """Save logits/probabilities under tmp_path and build an AnalysisView pointing at it."""
    save_npy(tmp_path / "logits.npy", logits)
    save_npy(tmp_path / "probabilities.npy", probabilities)
    return AnalysisView.new(
        kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
        manifest_id=ManifestId("mf_test"),
        shape=list(logits.shape),
        axes={"observations": "stimuli", "features": "features"},
        source_roles=["stimuli"],
        transformation_lineage=["test_stub"],
        artifact_path=str(tmp_path),
    )


def _make_wrong_kind_view(tmp_path: Path) -> AnalysisView:
    return AnalysisView.new(
        kind=AnalysisViewKind.RDM,
        manifest_id=ManifestId("mf_test"),
        shape=[10, 10],
        axes={"i": "x", "j": "x"},
        source_roles=["stimuli"],
        transformation_lineage=["stub"],
        artifact_path=str(tmp_path),
    )


# ---------- PCA ----------

class TestRunPCA:
    def test_basic_run(self, data_root_tmp: Path, tmp_path: Path):
        rng = np.random.default_rng(0)
        # Generate data with clear PC1 direction
        n, d = 50, 20
        logits = rng.normal(size=(n, d)).astype(np.float32)
        logits[:, 0] *= 10  # boost first feature
        probs = np.abs(logits) / (np.abs(logits).sum(axis=1, keepdims=True) + 1e-9)
        probs = probs.astype(np.float32)
        view = _make_feature_view(tmp_path, logits, probs)

        out_view, summary = run_pca(view=view, n_components=5)
        assert out_view.kind == AnalysisViewKind.OBSERVATION_BY_FEATURE
        assert out_view.axes["features"] == "pca_components"
        assert summary["method"] == "pca"
        assert summary["n_components"] == 5
        assert summary["n_images"] == n
        assert 0.0 < summary["explained_variance_ratio_total"] <= 1.0
        assert Path(summary["artifacts"]["scores"]).exists()
        assert Path(summary["artifacts"]["components"]).exists()
        assert Path(summary["artifacts"]["component_stats"]).exists()
        # cache writes view.json + summary.json + spec.json automatically
        assert (Path(out_view.artifact_path) / "view.json").exists()
        assert (Path(out_view.artifact_path) / "spec.json").exists()
        assert summary["view_id"] == out_view.view_id

    def test_scores_shape(self, data_root_tmp: Path, tmp_path: Path):
        rng = np.random.default_rng(1)
        n, d, k = 40, 30, 5
        logits = rng.normal(size=(n, d)).astype(np.float32)
        view = _make_feature_view(tmp_path, logits, logits)
        out_view, summary = run_pca(view=view, n_components=k)
        scores = np.load(summary["artifacts"]["scores"])
        assert scores.shape == (n, k)
        assert out_view.shape == [n, k]

    def test_rejects_wrong_view_kind(self, data_root_tmp: Path, tmp_path: Path):
        view = _make_wrong_kind_view(tmp_path)
        with pytest.raises(ViewKindMismatchError):
            run_pca(view=view, n_components=2)

    def test_rejects_view_without_artifact_path(self, data_root_tmp: Path):
        view = AnalysisView.new(
            kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
            manifest_id=ManifestId("mf_x"),
            shape=[10, 5], axes={"o": "x", "f": "y"},
            source_roles=["stimuli"], transformation_lineage=["stub"],
            artifact_path=None,
        )
        with pytest.raises(ValueError, match="no artifact_path"):
            run_pca(view=view, n_components=2)

    def test_custom_input_array_name(self, data_root_tmp: Path, tmp_path: Path):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(20, 10)).astype(np.float32)
        probs = np.abs(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        view = _make_feature_view(tmp_path, logits, probs.astype(np.float32))
        # PCA on probabilities instead of logits
        _, summary = run_pca(view=view, n_components=3, input_array_name="probabilities")
        assert summary["input_array_name"] == "probabilities"

    def test_idempotent_cache_hit(self, data_root_tmp: Path, tmp_path: Path):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(20, 12)).astype(np.float32)
        view = _make_feature_view(tmp_path, logits, logits)
        v1, s1 = run_pca(view=view, n_components=4)
        v2, s2 = run_pca(view=view, n_components=4)
        assert s1["from_cache"] is False
        assert s2["from_cache"] is True
        assert v1.view_id == v2.view_id

    def test_label_does_not_invalidate_cache(self, data_root_tmp: Path, tmp_path: Path):
        rng = np.random.default_rng(8)
        logits = rng.normal(size=(20, 12)).astype(np.float32)
        view = _make_feature_view(tmp_path, logits, logits)
        _, s1 = run_pca(view=view, n_components=4, label="run_alpha")
        _, s2 = run_pca(view=view, n_components=4, label="run_beta")
        assert s1["from_cache"] is False
        assert s2["from_cache"] is True

    def test_pca_contract_name(self):
        assert PCA_CONTRACT.name == "run_pca"
        assert PCA_CONTRACT.family == "factor_models"


# ---------- NMF ----------

class TestApplyProbabilityTemperature:
    def test_temperature_one_is_identity(self):
        rng = np.random.default_rng(0)
        P = rng.dirichlet(alpha=np.ones(10), size=20).astype(np.float64)
        P_t = apply_probability_temperature(P, temperature=1.0)
        np.testing.assert_allclose(P_t, P, atol=1e-10)

    def test_row_sums_to_one_after_temperature(self):
        rng = np.random.default_rng(1)
        P = rng.dirichlet(alpha=np.ones(8), size=15)
        for T in [0.5, 1.0, 2.0, 5.0]:
            P_t = apply_probability_temperature(P, temperature=T)
            np.testing.assert_allclose(P_t.sum(axis=1), 1.0, atol=1e-9)

    def test_rejects_zero_or_negative_temperature(self):
        P = np.ones((2, 3)) / 3
        with pytest.raises(ValueError, match="temperature"):
            apply_probability_temperature(P, temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            apply_probability_temperature(P, temperature=-1.0)

    def test_rejects_negative_probabilities(self):
        P = np.array([[0.5, -0.5, 1.0]])
        with pytest.raises(ValueError, match="non-negative"):
            apply_probability_temperature(P, temperature=1.0)

    def test_high_temperature_smooths(self):
        # Sharp distribution: one class near 1
        P = np.array([[0.98, 0.01, 0.01]])
        smoothed = apply_probability_temperature(P, temperature=10.0)
        # Entropy should rise (distribution closer to uniform)
        assert -np.sum(smoothed * np.log(smoothed)) > -np.sum(P * np.log(P + 1e-12))


class TestRunNMF:
    def _probs(self, n: int = 30, d: int = 20, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        P = rng.dirichlet(alpha=np.ones(d) * 0.5, size=n)
        return P.astype(np.float32)

    def test_basic_run(self, data_root_tmp: Path, tmp_path: Path):
        probs = self._probs()
        view = _make_feature_view(tmp_path, np.log(probs + 1e-12), probs)
        out_view, summary = run_nmf(view=view, n_components=5, temperature=1.0)
        assert out_view.kind == AnalysisViewKind.OBSERVATION_BY_FEATURE
        assert out_view.axes["features"] == "nmf_components"
        assert summary["method"] == "nmf"
        assert summary["n_components"] == 5
        assert summary["reconstruction_err"] >= 0.0
        assert summary["n_iter"] >= 1
        assert Path(summary["artifacts"]["scores"]).exists()
        assert (Path(out_view.artifact_path) / "view.json").exists()
        assert summary["view_id"] == out_view.view_id

    def test_scores_shape(self, data_root_tmp: Path, tmp_path: Path):
        probs = self._probs(n=24, d=15)
        view = _make_feature_view(tmp_path, np.log(probs + 1e-12), probs)
        out_view, summary = run_nmf(view=view, n_components=4)
        scores = np.load(summary["artifacts"]["scores"])
        assert scores.shape == (24, 4)
        assert out_view.shape == [24, 4]

    def test_scores_nonnegative(self, data_root_tmp: Path, tmp_path: Path):
        probs = self._probs()
        view = _make_feature_view(tmp_path, np.log(probs + 1e-12), probs)
        _, summary = run_nmf(view=view, n_components=3)
        scores = np.load(summary["artifacts"]["scores"])
        assert (scores >= 0).all()

    def test_rejects_wrong_view_kind(self, data_root_tmp: Path, tmp_path: Path):
        view = _make_wrong_kind_view(tmp_path)
        with pytest.raises(ViewKindMismatchError):
            run_nmf(view=view, n_components=2)

    def test_temperature_propagated_to_summary(self, data_root_tmp: Path, tmp_path: Path):
        probs = self._probs()
        view = _make_feature_view(tmp_path, np.log(probs + 1e-12), probs)
        _, summary = run_nmf(view=view, n_components=3, temperature=2.5)
        assert summary["temperature"] == 2.5

    def test_idempotent_cache_hit(self, data_root_tmp: Path, tmp_path: Path):
        probs = self._probs(n=20, d=15)
        view = _make_feature_view(tmp_path, np.log(probs + 1e-12), probs)
        v1, s1 = run_nmf(view=view, n_components=4)
        v2, s2 = run_nmf(view=view, n_components=4)
        assert s1["from_cache"] is False
        assert s2["from_cache"] is True
        assert v1.view_id == v2.view_id

    def test_temperature_change_invalidates_cache(self, data_root_tmp: Path, tmp_path: Path):
        probs = self._probs(n=20, d=15)
        view = _make_feature_view(tmp_path, np.log(probs + 1e-12), probs)
        _, s1 = run_nmf(view=view, n_components=3, temperature=1.0)
        _, s2 = run_nmf(view=view, n_components=3, temperature=2.0)
        assert s1["from_cache"] is False
        assert s2["from_cache"] is False

    def test_nmf_contract_name(self):
        assert NMF_CONTRACT.name == "run_nmf"
        assert NMF_CONTRACT.family == "factor_models"
