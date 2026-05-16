"""End-to-end MCP test for compare_jpeg_by_animate_inanimate.

Chains the real wrappers: allen_build_manifest -> extract_vit_features ->
extract_jpeg_sizes -> compare_jpeg_by_animate_inanimate. AllenSDK + ViT
are mocked (deterministic fake logits + animate/inanimate labels by image).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.mcp.analysis_tools import compare_jpeg_by_animate_inanimate
from mousehash.mcp.target_tools import allen_build_manifest
from mousehash.mcp.transformation_tools import extract_jpeg_sizes, extract_vit_features


def _bootstrap_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    n_images: int = 20,
) -> tuple[str, str]:
    """Set up Allen + ViT stubs, run extract_vit_features + extract_jpeg_sizes,
    return (jpeg_view_id, vit_view_id).
    """
    monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
    from mousehash.artifacts import paths as _paths
    _paths._load_dotenv_once.cache_clear()

    rng = np.random.default_rng(0)
    # Random image stack — the wrapper test exercises the plumbing, not the science.
    stub_stack = rng.integers(0, 256, size=(n_images, 64, 64), dtype=np.uint16).astype(np.float32)

    monkeypatch.setattr(
        "mousehash.targets.allen.manifest.fetch_natural_scene_template",
        lambda manifest_path=None: stub_stack,
    )
    monkeypatch.setattr(
        "mousehash.targets.allen.loaders.fetch_natural_scene_template",
        lambda scene_set_id, manifest_path=None: stub_stack,
    )

    # ViT stub: deterministic logits, half "animate" (cls < 397), half "inanimate"
    def fake_vit(frames, model_name=None, batch_size=None, device=None):
        from scipy.special import softmax
        n = len(frames)
        logits = np.zeros((n, 1000), dtype=np.float32)
        # First half: argmax = 100 (animate); second half: argmax = 500 (inanimate)
        for i in range(n):
            target = 100 if i < n // 2 else 500
            logits[i, target] = 50.0
        return logits, softmax(logits, axis=1).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.transformations.feature_extraction.run_vit_on_frames", fake_vit
    )

    manifest_id = allen_build_manifest(scene_set_id="cmp_test")["manifest_id"]
    vit = extract_vit_features(manifest_id=manifest_id)
    jpeg = extract_jpeg_sizes(manifest_id=manifest_id, qualities=[10, 50, 90])
    return jpeg["view_id"], vit["view_id"]


class TestCompareJpegByAnimateInanimate:
    def test_returns_full_result_dict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        jpeg_id, vit_id = _bootstrap_chain(monkeypatch, tmp_path, data_root_tmp)
        result = compare_jpeg_by_animate_inanimate(jpeg_view_id=jpeg_id, vit_view_id=vit_id)
        assert "per_feature" in result and len(result["per_feature"]) == 3
        assert "overall" in result
        assert "summary" in result and isinstance(result["summary"], str)
        assert result["units"] == "kB"
        assert result["feature_axis"] == "jpeg_quality_levels"
        assert {f["feature_name"] for f in result["per_feature"]} == {"10", "50", "90"}

    def test_n_animate_n_inanimate_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        jpeg_id, vit_id = _bootstrap_chain(monkeypatch, tmp_path, data_root_tmp, n_images=20)
        result = compare_jpeg_by_animate_inanimate(jpeg_view_id=jpeg_id, vit_view_id=vit_id)
        assert result["n_animate"] == 10
        assert result["n_inanimate"] == 10
        assert result["n_animate"] + result["n_inanimate"] == 20

    def test_per_feature_has_stats(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        jpeg_id, vit_id = _bootstrap_chain(monkeypatch, tmp_path, data_root_tmp)
        result = compare_jpeg_by_animate_inanimate(jpeg_view_id=jpeg_id, vit_view_id=vit_id)
        for f in result["per_feature"]:
            assert f["sufficient_samples"] is True
            assert "p_welch" in f and "p_mannwhitneyu" in f
            assert "cohens_d" in f
            assert "delta_mean" in f and "delta_mean_pct" in f

    def test_plot_written_by_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        jpeg_id, vit_id = _bootstrap_chain(monkeypatch, tmp_path, data_root_tmp)
        result = compare_jpeg_by_animate_inanimate(jpeg_view_id=jpeg_id, vit_view_id=vit_id)
        assert "plot_html" in result["artifacts"]
        plot_path = Path(result["artifacts"]["plot_html"])
        assert plot_path.exists() and plot_path.stat().st_size > 1024

    def test_plot_skipped_when_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        jpeg_id, vit_id = _bootstrap_chain(monkeypatch, tmp_path, data_root_tmp)
        result = compare_jpeg_by_animate_inanimate(
            jpeg_view_id=jpeg_id, vit_view_id=vit_id, plot=False,
        )
        assert "plot_html" not in result["artifacts"]

    def test_idempotent_cache_hit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        jpeg_id, vit_id = _bootstrap_chain(monkeypatch, tmp_path, data_root_tmp)
        r1 = compare_jpeg_by_animate_inanimate(jpeg_view_id=jpeg_id, vit_view_id=vit_id)
        r2 = compare_jpeg_by_animate_inanimate(jpeg_view_id=jpeg_id, vit_view_id=vit_id)
        assert r1["from_cache"] is False
        assert r2["from_cache"] is True
        assert r1["view_id"] == r2["view_id"]

    def test_unknown_view_id_returns_structured_error(self, data_root_tmp: Path):
        result = compare_jpeg_by_animate_inanimate(
            jpeg_view_id="view_nonexistent_jpeg",
            vit_view_id="view_nonexistent_vit",
        )
        assert "error" in result
        assert result["type"] == "ViewNotFoundError"

    def test_result_is_json_serializable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        import json, math

        jpeg_id, vit_id = _bootstrap_chain(monkeypatch, tmp_path, data_root_tmp)
        result = compare_jpeg_by_animate_inanimate(jpeg_view_id=jpeg_id, vit_view_id=vit_id)

        # The result contains float('nan') in some fields when groups are tiny; here all
        # groups are 10/10 so we shouldn't have NaN. But json.dumps fails on NaN unless
        # allow_nan is True (the default). Just verify it doesn't crash.
        s = json.dumps(result, default=str)
        assert len(s) > 200
