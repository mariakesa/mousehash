"""Tests for mcp/analysis_tools.py — run_pca + run_nmf MCP wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.mcp.analysis_tools import run_nmf, run_pca
from mousehash.mcp.target_tools import allen_build_manifest
from mousehash.mcp.transformation_tools import extract_vit_features


def _bootstrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path) -> str:
    """Stub Allen + ViT and produce a ViT feature view; return its view_id."""
    monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
    from mousehash.artifacts import paths as _paths
    _paths._load_dotenv_once.cache_clear()

    rng = np.random.default_rng(0)
    stub_stack = rng.integers(0, 256, size=(20, 64, 64), dtype=np.uint16).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.targets.allen.manifest.fetch_natural_scene_template",
        lambda manifest_path=None: stub_stack,
    )
    monkeypatch.setattr(
        "mousehash.targets.allen.loaders.fetch_natural_scene_template",
        lambda scene_set_id, manifest_path=None: stub_stack,
    )

    def fake_vit(frames, model_name=None, batch_size=None, device=None):
        from scipy.special import softmax
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(len(frames), 1000)).astype(np.float32)
        return logits, softmax(logits, axis=1).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.transformations.feature_extraction.run_vit_on_frames", fake_vit
    )

    manifest_id = allen_build_manifest(scene_set_id="ana_test")["manifest_id"]
    vit_result = extract_vit_features(manifest_id=manifest_id)
    return vit_result["view_id"]


class TestRunPCA:
    def test_basic(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path):
        view_id = _bootstrap(monkeypatch, tmp_path, data_root_tmp)
        result = run_pca(view_id=view_id, n_components=4)
        assert result["view_id"].startswith("view_")
        assert result["view_id"] != view_id  # PCA produces a new view
        assert result["summary"]["method"] == "pca"
        assert result["summary"]["n_components"] == 4
        assert Path(result["artifact_path"]).exists()

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path):
        view_id = _bootstrap(monkeypatch, tmp_path, data_root_tmp)
        r1 = run_pca(view_id=view_id, n_components=3)
        r2 = run_pca(view_id=view_id, n_components=3)
        assert r1["from_cache"] is False
        assert r2["from_cache"] is True
        assert r1["view_id"] == r2["view_id"]

    def test_unknown_view_returns_structured_error(self, data_root_tmp: Path):
        result = run_pca(view_id="view_doesnotexist")
        assert result["type"] == "ViewNotFoundError"


class TestRunNMF:
    def test_basic(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path):
        view_id = _bootstrap(monkeypatch, tmp_path, data_root_tmp)
        result = run_nmf(view_id=view_id, n_components=4, temperature=1.0)
        assert result["view_id"].startswith("view_")
        assert result["summary"]["method"] == "nmf"
        assert result["summary"]["temperature"] == 1.0
        assert Path(result["artifact_path"]).exists()

    def test_temperature_change_invalidates_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        view_id = _bootstrap(monkeypatch, tmp_path, data_root_tmp)
        r1 = run_nmf(view_id=view_id, n_components=3, temperature=1.0)
        r2 = run_nmf(view_id=view_id, n_components=3, temperature=2.0)
        assert r1["view_id"] != r2["view_id"]

    def test_unknown_view_returns_structured_error(self, data_root_tmp: Path):
        result = run_nmf(view_id="view_doesnotexist")
        assert result["type"] == "ViewNotFoundError"
