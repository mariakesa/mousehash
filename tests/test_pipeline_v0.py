"""End-to-end integration test of run_allen_natural_scenes_v0.

AllenSDK fetch and ViT inference are both monkeypatched. Everything else —
sklearn PCA/NMF, plotly HTML, manifest YAML, file I/O — is real. This is
the test that catches breakage in the pipeline glue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.pipelines import run_allen_natural_scenes_v0


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, n_images: int = 10) -> None:
    """Replace both the AllenSDK image fetch and the ViT inference with cheap stubs."""
    rng = np.random.default_rng(0)
    stub_stack = rng.integers(0, 256, size=(n_images, 64, 64), dtype=np.uint16).astype(np.float32)

    def fake_fetch(manifest_path=None):
        return stub_stack

    # Allen fetch in two callers
    monkeypatch.setattr("mousehash.targets.allen.manifest.fetch_natural_scene_template", fake_fetch)
    monkeypatch.setattr("mousehash.targets.allen.loaders.fetch_natural_scene_template", fake_fetch)

    # ViT: produce deterministic synthetic logits + probs
    rng2 = np.random.default_rng(1)
    fake_logits = rng2.normal(size=(n_images, 1000)).astype(np.float32)

    def fake_run_vit(frames, model_name=None, batch_size=None, device=None):
        from scipy.special import softmax
        return fake_logits, softmax(fake_logits, axis=1).astype(np.float32)

    monkeypatch.setattr("mousehash.transformations.feature_extraction.run_vit_on_frames", fake_run_vit)


class TestPipelineV0:
    def test_end_to_end_returns_full_result(
        self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_pipeline(monkeypatch, n_images=12)

        result = run_allen_natural_scenes_v0(
            scene_set_id="pipe_test",
            pca_n_components=4,
            nmf_n_components=4,
        )

        assert "manifest_id" in result and result["manifest_id"].startswith("mf_")
        assert "view_id" in result and result["view_id"].startswith("view_")
        for key in ["pca_summary", "nmf_summary", "report", "vit_summary"]:
            assert key in result

    def test_all_artifact_files_exist(
        self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_pipeline(monkeypatch, n_images=10)

        result = run_allen_natural_scenes_v0(
            scene_set_id="art_test",
            pca_n_components=3,
            nmf_n_components=3,
        )

        for path_key in ["scores", "components", "component_stats"]:
            assert Path(result["pca_summary"]["artifacts"][path_key]).exists()
            assert Path(result["nmf_summary"]["artifacts"][path_key]).exists()
        for report_key in ["index", "pca", "nmf"]:
            p = Path(result["report"]["reports"][report_key])
            assert p.exists()
            assert p.stat().st_size > 0

    def test_pipeline_idempotent_on_manifest_id(
        self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_pipeline(monkeypatch, n_images=8)

        r1 = run_allen_natural_scenes_v0(scene_set_id="idem", pca_n_components=3, nmf_n_components=3)
        r2 = run_allen_natural_scenes_v0(scene_set_id="idem", pca_n_components=3, nmf_n_components=3)
        # Same scene_set_id should hash to the same manifest_id and view_id
        assert r1["manifest_id"] == r2["manifest_id"]
        assert r1["view_id"] == r2["view_id"]

    def test_pca_components_count_matches_request(
        self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_pipeline(monkeypatch, n_images=10)

        result = run_allen_natural_scenes_v0(
            scene_set_id="comp_test",
            pca_n_components=5,
            nmf_n_components=4,
        )
        assert result["pca_summary"]["n_components"] == 5
        assert result["nmf_summary"]["n_components"] == 4
