"""Integration test for mcp/pipeline_tools.py — run_allen_natural_scenes_v0 over MCP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.mcp.pipeline_tools import run_allen_natural_scenes_v0


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, n_images: int = 10) -> None:
    """Mock Allen fetch + ViT for fast end-to-end exercise."""
    rng = np.random.default_rng(0)
    stub_stack = rng.integers(0, 256, size=(n_images, 64, 64), dtype=np.uint16).astype(np.float32)

    monkeypatch.setattr(
        "mousehash.targets.allen.manifest.fetch_natural_scene_template",
        lambda manifest_path=None: stub_stack,
    )
    monkeypatch.setattr(
        "mousehash.targets.allen.loaders.fetch_natural_scene_template",
        lambda scene_set_id, manifest_path=None: stub_stack,
    )

    rng2 = np.random.default_rng(1)
    fake_logits = rng2.normal(size=(n_images, 1000)).astype(np.float32)
    def fake_run_vit(frames, model_name=None, batch_size=None, device=None):
        from scipy.special import softmax
        return fake_logits, softmax(fake_logits, axis=1).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.transformations.feature_extraction.run_vit_on_frames", fake_run_vit
    )


class TestRunPipelineV0OverMcp:
    def test_full_chain_via_mcp_wrapper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_pipeline(monkeypatch, n_images=10)

        result = run_allen_natural_scenes_v0(
            scene_set_id="mcp_pipe_v0",
            pca_n_components=3,
            nmf_n_components=3,
        )
        assert "manifest_id" in result and result["manifest_id"].startswith("mf_")
        for key in ["view_id", "jpeg_view_id", "pca_view_id", "nmf_view_id"]:
            assert key in result and result[key].startswith("view_")

        # Report files materialized
        for k in ("index", "pca", "nmf"):
            p = Path(result["report"]["reports"][k])
            assert p.exists() and p.stat().st_size > 0

    def test_idempotent_via_mcp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_pipeline(monkeypatch, n_images=8)

        r1 = run_allen_natural_scenes_v0(scene_set_id="idem", pca_n_components=3, nmf_n_components=3)
        r2 = run_allen_natural_scenes_v0(scene_set_id="idem", pca_n_components=3, nmf_n_components=3)
        # All view ids stable across calls (content-addressed)
        for key in ["view_id", "jpeg_view_id", "pca_view_id", "nmf_view_id"]:
            assert r1[key] == r2[key]

    def test_custom_jpeg_qualities_propagated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_pipeline(monkeypatch, n_images=6)

        result = run_allen_natural_scenes_v0(
            scene_set_id="jq_test",
            pca_n_components=2, nmf_n_components=2,
            jpeg_qualities=[20, 60],
        )
        assert result["jpeg_summary"]["qualities"] == [20, 60]
