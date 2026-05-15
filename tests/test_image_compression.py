"""Tests for transformations/image_compression.py — JPEG-size feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.artifacts.cache import cache_dir_for, ComputationSpec, fingerprint_array
from mousehash.core.analysis_view import AnalysisViewKind
from mousehash.core.ids import ManifestId
from mousehash.transformations.image_compression import (
    DEFAULT_JPEG_QUALITIES,
    extract_jpeg_size_view,
    jpeg_size_bytes,
)


def _stub_frames(n: int = 6, size: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, size, size), dtype=np.uint8)


# ---------- jpeg_size_bytes ----------

class TestJpegSizeBytes:
    def test_returns_positive_int(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        size = jpeg_size_bytes(img, quality=75)
        assert isinstance(size, int) and size > 0

    def test_larger_quality_larger_size(self):
        """Higher quality should not yield a smaller JPEG (almost always strictly larger for non-trivial images)."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        low = jpeg_size_bytes(img, quality=10)
        high = jpeg_size_bytes(img, quality=90)
        assert high > low

    def test_constant_image_smaller_than_random(self):
        const = np.full((64, 64), 128, dtype=np.uint8)
        rng = np.random.default_rng(0)
        rand = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        assert jpeg_size_bytes(const, quality=75) < jpeg_size_bytes(rand, quality=75)


# ---------- extract_jpeg_size_view ----------

class TestExtractJpegSizeView:
    def test_view_shape_and_kind(self, data_root_tmp: Path):
        frames = _stub_frames(n=5)
        view, _ = extract_jpeg_size_view(
            frames=frames,
            manifest_id=ManifestId("mf_jpeg"),
            scene_set_id="set1",
            qualities=[10, 50, 90],
        )
        assert view.kind == AnalysisViewKind.OBSERVATION_BY_FEATURE
        assert view.shape == [5, 3]
        assert view.axes["features"] == "jpeg_quality_levels"

    def test_default_qualities(self, data_root_tmp: Path):
        frames = _stub_frames(n=4)
        view, summary = extract_jpeg_size_view(
            frames=frames,
            manifest_id=ManifestId("mf_def"),
            scene_set_id="set2",
        )
        assert tuple(summary["qualities"]) == tuple(sorted(DEFAULT_JPEG_QUALITIES))
        assert view.shape == [4, len(DEFAULT_JPEG_QUALITIES)]

    def test_qualities_are_sorted_in_summary_and_lineage(self, data_root_tmp: Path):
        frames = _stub_frames(n=3)
        view, summary = extract_jpeg_size_view(
            frames=frames,
            manifest_id=ManifestId("mf_sort"),
            scene_set_id="set3",
            qualities=[90, 10, 50],  # passed unsorted
        )
        assert summary["qualities"] == [10, 50, 90]
        lineage_str = " ".join(view.transformation_lineage)
        assert "10,50,90" in lineage_str

    def test_writes_bytes_kb_mb_npy_files(self, data_root_tmp: Path):
        frames = _stub_frames(n=4)
        view, summary = extract_jpeg_size_view(
            frames=frames,
            manifest_id=ManifestId("mf_io"),
            scene_set_id="set_io",
            qualities=[25, 75],
        )
        out = Path(view.artifact_path)
        bytes_path = out / "sizes_bytes.npy"
        kb_path = out / "sizes_kb.npy"
        mb_path = out / "sizes_mb.npy"
        assert bytes_path.exists() and kb_path.exists() and mb_path.exists()

        b = np.load(bytes_path)
        kb = np.load(kb_path)
        mb = np.load(mb_path)
        assert b.shape == (4, 2)
        assert b.dtype == np.int64
        np.testing.assert_allclose(kb, b / 1024.0, rtol=1e-5)
        np.testing.assert_allclose(mb, b / (1024.0 * 1024.0), rtol=1e-5)

    def test_mean_size_increases_with_quality(self, data_root_tmp: Path):
        """Across many images, mean JPEG size should be monotone non-decreasing in quality."""
        frames = _stub_frames(n=10, size=48)
        _, summary = extract_jpeg_size_view(
            frames=frames,
            manifest_id=ManifestId("mf_mono"),
            scene_set_id="set_mono",
            qualities=[10, 25, 50, 75, 90],
        )
        means = summary["mean_size_kb_by_quality"]
        for a, b in zip(means, means[1:]):
            assert b >= a, f"mean size dropped from q -> q+ : {a} -> {b}"

    def test_idempotent_cache_hit(self, data_root_tmp: Path):
        frames = _stub_frames(n=4)
        kwargs = dict(
            manifest_id=ManifestId("mf_idem"),
            scene_set_id="set_idem",
            qualities=[20, 80],
        )
        view1, summary1 = extract_jpeg_size_view(frames=frames, **kwargs)
        assert summary1["from_cache"] is False

        view2, summary2 = extract_jpeg_size_view(frames=frames, **kwargs)
        assert summary2["from_cache"] is True
        assert view1.view_id == view2.view_id
        assert view1.artifact_path == view2.artifact_path

    def test_pixel_change_invalidates_cache(self, data_root_tmp: Path):
        frames = _stub_frames(n=4)
        kwargs = dict(
            manifest_id=ManifestId("mf_inv"),
            scene_set_id="set_inv",
            qualities=[25, 75],
        )
        v1, _ = extract_jpeg_size_view(frames=frames, **kwargs)

        # Modify one pixel
        frames_mod = frames.copy()
        frames_mod[0, 0, 0] = (int(frames_mod[0, 0, 0]) + 1) % 256
        v2, summary2 = extract_jpeg_size_view(frames=frames_mod, **kwargs)
        assert v1.view_id != v2.view_id
        assert summary2["from_cache"] is False

    def test_label_does_not_invalidate_cache(self, data_root_tmp: Path):
        frames = _stub_frames(n=3)
        kwargs = dict(
            manifest_id=ManifestId("mf_lab"),
            scene_set_id="set_lab",
            qualities=[20, 80],
        )
        _, s1 = extract_jpeg_size_view(frames=frames, label="run_alpha", **kwargs)
        _, s2 = extract_jpeg_size_view(frames=frames, label="run_beta", **kwargs)
        assert s1["from_cache"] is False
        assert s2["from_cache"] is True

    def test_rejects_invalid_quality(self, data_root_tmp: Path):
        frames = _stub_frames(n=2)
        with pytest.raises(ValueError, match="quality"):
            extract_jpeg_size_view(frames=frames, manifest_id=ManifestId("x"),
                                    scene_set_id="s", qualities=[0])
        with pytest.raises(ValueError, match="quality"):
            extract_jpeg_size_view(frames=frames, manifest_id=ManifestId("x"),
                                    scene_set_id="s", qualities=[100])

    def test_rejects_non_3d_frames(self, data_root_tmp: Path):
        with pytest.raises(ValueError, match="image stack"):
            extract_jpeg_size_view(
                frames=np.zeros((32, 32), dtype=np.uint8),
                manifest_id=ManifestId("x"),
                scene_set_id="s",
            )

    def test_handles_float_frames(self, data_root_tmp: Path):
        """Float-dtype frames should be normalized to uint8 internally."""
        rng = np.random.default_rng(0)
        frames = rng.normal(size=(4, 32, 32)).astype(np.float32)
        view, summary = extract_jpeg_size_view(
            frames=frames,
            manifest_id=ManifestId("mf_f"),
            scene_set_id="set_f",
            qualities=[50],
        )
        assert view.shape == [4, 1]
        assert summary["mean_size_kb_by_quality"][0] > 0
