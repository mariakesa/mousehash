"""Tests for tools.decoders.logistic_decoder.

We build a synthetic learnable problem: 30 images, 5 neurons. Neuron 0's
activity is just a noisy copy of the label, so any reasonable classifier
should hit near-perfect CV accuracy. Sklearn is exercised for real (it's
cheap and worth the coverage).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.artifacts.io import save_npy
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.errors import ViewKindMismatchError
from mousehash.tools.decoders.logistic_decoder import (
    DecoderInputError,
    run_logistic_decoder,
)


N_IMAGES = 30
N_NEURONS = 5


def _make_learnable_views(data_root: Path, rng_seed: int = 0) -> tuple[AnalysisView, AnalysisView]:
    """Build (neural_view, labels_view) where the label is learnable from neuron 0."""
    rng = np.random.default_rng(rng_seed)
    y = np.tile(np.array([0, 1]), N_IMAGES // 2).astype(np.int8)  # balanced 0/1
    # neuron 0 = label + small noise; other neurons = pure noise
    neural = rng.normal(scale=0.1, size=(N_NEURONS, N_IMAGES)).astype(np.float32)
    neural[0] = y.astype(np.float32) + rng.normal(scale=0.1, size=N_IMAGES).astype(np.float32)

    neural_dir = data_root / "fake_neural"
    neural_dir.mkdir(parents=True, exist_ok=True)
    save_npy(neural_dir / "event_probabilities.npy", neural)
    labels_dir = data_root / "fake_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    save_npy(labels_dir / "animate_inanimate.npy", y)

    neural_view = AnalysisView.new(
        kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
        manifest_id="mf_test",
        shape=[N_NEURONS, N_IMAGES],
        axes={"observations": "neurons_across_sessions", "features": "natural_scene_images"},
        source_roles=["neural_data", "stimuli"],
        transformation_lineage=["fake_neural"],
        artifact_path=str(neural_dir),
    )
    labels_view = AnalysisView.new(
        kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
        manifest_id="mf_test",
        shape=[N_IMAGES, 1000],
        axes={"observations": "stimulus_presentations", "features": "imagenet_classifier_output"},
        source_roles=["stimuli"],
        transformation_lineage=["fake_vit"],
        artifact_path=str(labels_dir),
    )
    return neural_view, labels_view


class TestSplitStrategies:
    def test_stratified_kfold(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        view, summary = run_logistic_decoder(
            nv, lv, split_strategy="stratified_kfold", k_folds=5,
        )
        assert view.kind == AnalysisViewKind.METRIC_TABLE
        assert summary["cv_balanced_accuracy"] >= 0.8
        assert summary["chance_accuracy"] == pytest.approx(0.5)
        assert summary["n_features"] == N_NEURONS
        assert summary["n_images"] == N_IMAGES

    def test_loo(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        view, summary = run_logistic_decoder(nv, lv, split_strategy="loo")
        assert view.kind == AnalysisViewKind.METRIC_TABLE
        assert summary["cv_balanced_accuracy"] >= 0.8
        per_fold = np.load(Path(view.artifact_path) / "per_fold_accuracy.npy")
        assert per_fold.shape == (N_IMAGES,)

    def test_holdout(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        view, summary = run_logistic_decoder(
            nv, lv, split_strategy="holdout", holdout_fraction=0.25,
        )
        assert summary["cv_balanced_accuracy"] >= 0.8
        per_fold = np.load(Path(view.artifact_path) / "per_fold_accuracy.npy")
        assert per_fold.shape == (1,)


class TestArtifacts:
    def test_all_artifacts_written(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        view, _ = run_logistic_decoder(nv, lv, split_strategy="stratified_kfold", k_folds=5)
        art = Path(view.artifact_path)
        for name in (
            "cv_predictions.npy", "cv_probabilities.npy", "per_fold_accuracy.npy",
            "final_model_coef.npy", "final_model_intercept.npy",
        ):
            assert (art / name).exists(), name
        coef = np.load(art / "final_model_coef.npy")
        assert coef.shape == (N_NEURONS,)

    def test_coef_concentrates_on_informative_neuron(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        view, _ = run_logistic_decoder(nv, lv, split_strategy="stratified_kfold", k_folds=5)
        coef = np.load(Path(view.artifact_path) / "final_model_coef.npy")
        # Neuron 0 is the only informative feature; it should dominate the weights.
        assert np.abs(coef[0]) > np.abs(coef[1:]).max()


class TestHyperparamSearch:
    def test_search_hyperparams_records_best_C(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        view, summary = run_logistic_decoder(
            nv, lv, split_strategy="stratified_kfold", k_folds=5,
            search_hyperparams=True, C_grid=(0.1, 1.0, 10.0),
        )
        assert summary["search_hyperparams"] is True
        assert summary["best_C_mean"] in (0.1, 1.0, 10.0) or 0.1 <= summary["best_C_mean"] <= 10.0


class TestPermutationNull:
    def test_permutation_null_produces_p_value(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        view, summary = run_logistic_decoder(
            nv, lv, split_strategy="stratified_kfold", k_folds=5,
            n_permutations=20, random_state=42,
        )
        assert summary["p_value"] is not None
        # Learnable task: p-value should be small.
        assert summary["p_value"] <= 0.20
        perm = np.load(Path(view.artifact_path) / "permutation_accuracies.npy")
        assert perm.shape == (20,)


class TestCacheIdempotency:
    def test_second_call_is_cache_hit(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        v1, s1 = run_logistic_decoder(nv, lv, split_strategy="stratified_kfold", k_folds=5)
        v2, s2 = run_logistic_decoder(nv, lv, split_strategy="stratified_kfold", k_folds=5)
        assert s1["from_cache"] is False
        assert s2["from_cache"] is True
        assert v1.view_id == v2.view_id
        assert v1.artifact_path == v2.artifact_path


class TestValidation:
    def test_wrong_kind_raises(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        bad = AnalysisView.new(
            kind=AnalysisViewKind.RDM,
            manifest_id="mf_test", shape=[N_IMAGES, N_IMAGES],
            axes={"x": "y"}, source_roles=["stimuli"],
            transformation_lineage=["bad"], artifact_path=str(data_root_tmp / "fake_neural"),
        )
        with pytest.raises(ViewKindMismatchError):
            run_logistic_decoder(bad, lv, split_strategy="stratified_kfold")

    def test_length_mismatch_raises(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        # Overwrite labels with the wrong length
        bad_labels = np.zeros(N_IMAGES + 5, dtype=np.int8)
        save_npy(Path(lv.artifact_path) / "animate_inanimate.npy", bad_labels)
        with pytest.raises(DecoderInputError, match="n_images"):
            run_logistic_decoder(nv, lv, split_strategy="stratified_kfold")

    def test_unknown_split_strategy_raises(self, data_root_tmp: Path):
        nv, lv = _make_learnable_views(data_root_tmp)
        with pytest.raises(ValueError, match="Unknown split_strategy"):
            run_logistic_decoder(nv, lv, split_strategy="bogus")
