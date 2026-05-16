"""Tests for mcp/report_tools.py — generate_structure_report wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.mcp.analysis_tools import run_nmf, run_pca
from mousehash.mcp.report_tools import generate_structure_report
from mousehash.mcp.target_tools import allen_build_manifest
from mousehash.mcp.transformation_tools import extract_vit_features


def _bootstrap_full_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
) -> tuple[str, str, str, str]:
    """Build manifest -> ViT -> PCA -> NMF and return their ids."""
    monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
    from mousehash.artifacts import paths as _paths
    _paths._load_dotenv_once.cache_clear()

    rng = np.random.default_rng(0)
    stub = rng.integers(0, 256, size=(8, 64, 64), dtype=np.uint16).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.targets.allen.manifest.fetch_natural_scene_template",
        lambda manifest_path=None: stub,
    )
    monkeypatch.setattr(
        "mousehash.targets.allen.loaders.fetch_natural_scene_template",
        lambda scene_set_id, manifest_path=None: stub,
    )

    def fake_vit(frames, model_name=None, batch_size=None, device=None):
        from scipy.special import softmax
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(len(frames), 1000)).astype(np.float32)
        return logits, softmax(logits, axis=1).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.transformations.feature_extraction.run_vit_on_frames", fake_vit
    )

    manifest_id = allen_build_manifest(scene_set_id="rep_test")["manifest_id"]
    vit = extract_vit_features(manifest_id=manifest_id)
    pca = run_pca(view_id=vit["view_id"], n_components=3)
    nmf = run_nmf(view_id=vit["view_id"], n_components=3, temperature=1.0)
    return manifest_id, vit["view_id"], pca["view_id"], nmf["view_id"]


class TestGenerateStructureReport:
    def test_full_chain_produces_three_html_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id, vit_id, pca_id, nmf_id = _bootstrap_full_chain(
            monkeypatch, tmp_path, data_root_tmp
        )
        result = generate_structure_report(
            manifest_id=manifest_id,
            pca_view_id=pca_id,
            nmf_view_id=nmf_id,
            vit_view_id=vit_id,
        )
        for key in ["index", "pca", "nmf"]:
            p = Path(result["reports"][key])
            assert p.exists() and p.stat().st_size > 0

    def test_unknown_manifest_returns_structured_error(self, data_root_tmp: Path):
        result = generate_structure_report(
            manifest_id="mf_doesnotexist",
            pca_view_id="view_x", nmf_view_id="view_y", vit_view_id="view_z",
        )
        assert result["type"] == "ManifestNotFoundError"

    def test_unknown_view_returns_structured_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        # Build a valid manifest so the manifest check passes, then pass bad view ids.
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        rng = np.random.default_rng(0)
        stub = rng.integers(0, 256, size=(4, 32, 32), dtype=np.uint16).astype(np.float32)
        monkeypatch.setattr(
            "mousehash.targets.allen.manifest.fetch_natural_scene_template",
            lambda manifest_path=None: stub,
        )
        manifest_id = allen_build_manifest(scene_set_id="err_test")["manifest_id"]
        result = generate_structure_report(
            manifest_id=manifest_id,
            pca_view_id="view_nope", nmf_view_id="view_nope2", vit_view_id="view_nope3",
        )
        assert result["type"] == "ViewNotFoundError"

    def test_returns_summaries_alongside_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id, vit_id, pca_id, nmf_id = _bootstrap_full_chain(
            monkeypatch, tmp_path, data_root_tmp
        )
        result = generate_structure_report(
            manifest_id=manifest_id,
            pca_view_id=pca_id, nmf_view_id=nmf_id, vit_view_id=vit_id,
        )
        assert result["pca_summary"]["method"] == "pca"
        assert result["nmf_summary"]["method"] == "nmf"
