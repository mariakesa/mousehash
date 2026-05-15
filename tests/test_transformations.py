"""Tests for transformations/labeling.py and feature_extraction.py.

ViT inference is monkeypatched so tests run fast — the math/glue around it
(view construction, artifact saving, label derivation) is what we verify.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.core.analysis_view import AnalysisViewKind
from mousehash.core.ids import ManifestId
from mousehash.transformations.feature_extraction import extract_vit_features_view
from mousehash.transformations.labeling import (
    ANIMATE_MAX_CLASS_IDX,
    derive_animate_inanimate,
    derive_top1,
    load_imagenet_labels,
)


# ---------- labeling ----------

class TestLabelingLabels:
    def test_label_count(self):
        labels = load_imagenet_labels()
        assert len(labels) == 1000

    def test_label_count_cached(self):
        # Two calls should return the same object (lru_cache)
        assert load_imagenet_labels() is load_imagenet_labels()

    def test_first_and_last_labels_sane(self):
        labels = load_imagenet_labels()
        # cls 0 = tench (a fish); cls 999 = toilet tissue
        assert "tench" in labels[0]
        assert "toilet" in labels[999] or "tissue" in labels[999]

    def test_animate_threshold_constant(self):
        assert ANIMATE_MAX_CLASS_IDX == 397


class TestDeriveTop1:
    def test_basic_argmax(self):
        probs = np.array([
            [0.1, 0.8, 0.1],
            [0.5, 0.3, 0.2],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        top1 = derive_top1(probs)
        np.testing.assert_array_equal(top1, [1, 0, 2])
        assert top1.dtype == np.int32

    def test_one_image_one_class(self):
        probs = np.array([[1.0]])
        np.testing.assert_array_equal(derive_top1(probs), [0])


class TestDeriveAnimateInanimate:
    def test_dtype_is_int8(self):
        top1 = np.array([0, 397, 398, 999], dtype=np.int32)
        labels = derive_animate_inanimate(top1)
        assert labels.dtype == np.int8

    def test_threshold_boundary_397_is_animate(self):
        top1 = np.array([397], dtype=np.int32)
        assert derive_animate_inanimate(top1)[0] == 1

    def test_threshold_boundary_398_is_inanimate(self):
        top1 = np.array([398], dtype=np.int32)
        assert derive_animate_inanimate(top1)[0] == 0

    def test_custom_threshold(self):
        top1 = np.array([100, 200, 300], dtype=np.int32)
        # Custom threshold 150: classes <= 150 are "animate"
        labels = derive_animate_inanimate(top1, threshold_max_class_idx=150)
        np.testing.assert_array_equal(labels, [1, 0, 0])

    def test_mixed_classes(self):
        top1 = np.array([0, 397, 398, 999, 50], dtype=np.int32)
        np.testing.assert_array_equal(derive_animate_inanimate(top1), [1, 1, 0, 0, 1])


# ---------- feature_extraction.extract_vit_features_view ----------

def _patch_vit(monkeypatch, n_images: int, n_classes: int = 1000, seed: int = 0):
    """Replace run_vit_on_frames with a cheap deterministic fake."""
    rng = np.random.default_rng(seed)
    fake_logits = rng.normal(size=(n_images, n_classes)).astype(np.float32)
    # Make sure class 0 is the argmax for image 0 so we can check top1 trivially.
    fake_logits[0, 0] = 100.0

    def fake_run(frames, model_name=None, batch_size=None, device=None):
        assert frames.shape[0] == n_images
        # Simple softmax: matches the real function's output structure
        from scipy.special import softmax
        probs = softmax(fake_logits, axis=1).astype(np.float32)
        return fake_logits, probs

    monkeypatch.setattr("mousehash.transformations.feature_extraction.run_vit_on_frames", fake_run)
    return fake_logits


class TestExtractVitFeaturesView:
    def test_returns_observation_by_feature_view(self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch):
        n = 8
        _patch_vit(monkeypatch, n)
        frames = np.zeros((n, 64, 64), dtype=np.uint8)

        view, bundle = extract_vit_features_view(
            frames=frames,
            manifest_id=ManifestId("mf_test"),
            scene_set_id="stub",
        )
        assert view.kind == AnalysisViewKind.OBSERVATION_BY_FEATURE
        assert view.shape == [n, 1000]
        assert view.axes == {"observations": "stimulus_presentations", "features": "imagenet_classifier_output"}
        assert "stimuli" in view.source_roles

    def test_view_id_deterministic(self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch):
        n = 4
        _patch_vit(monkeypatch, n)
        frames = np.zeros((n, 64, 64), dtype=np.uint8)
        view1, _ = extract_vit_features_view(frames, ManifestId("mf_x"), scene_set_id="a",
                                              representation_spec_id="rep1")
        view2, _ = extract_vit_features_view(frames, ManifestId("mf_x"), scene_set_id="a",
                                              representation_spec_id="rep1")
        assert view1.view_id == view2.view_id

    def test_saves_all_four_npy_files(self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch):
        n = 4
        _patch_vit(monkeypatch, n)
        frames = np.zeros((n, 64, 64), dtype=np.uint8)
        _, bundle = extract_vit_features_view(frames, ManifestId("mf_q"), scene_set_id="setQ")
        out_dir = Path(bundle["out_dir"])
        for fname in ["logits.npy", "probabilities.npy", "top1.npy", "animate_inanimate.npy", "summary.json"]:
            assert (out_dir / fname).exists(), f"missing {fname}"

    def test_summary_counts_match_bundle(self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch):
        n = 6
        _patch_vit(monkeypatch, n)
        frames = np.zeros((n, 64, 64), dtype=np.uint8)
        _, bundle = extract_vit_features_view(frames, ManifestId("mf_z"), scene_set_id="setZ")
        s = bundle["summary"]
        assert s["n_images"] == n
        assert s["n_classes"] == 1000
        assert s["n_animate"] + s["n_inanimate"] == n

    def test_lineage_records_model_name(self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch):
        _patch_vit(monkeypatch, 4)
        frames = np.zeros((4, 64, 64), dtype=np.uint8)
        view, _ = extract_vit_features_view(
            frames, ManifestId("mf_l"), scene_set_id="setL", model_name="google/vit-base-patch16-224"
        )
        assert any("vit:" in step for step in view.transformation_lineage)
        assert "softmax" in view.transformation_lineage
