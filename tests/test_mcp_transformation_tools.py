"""Tests for mcp/transformation_tools.py — extract_vit_features + extract_jpeg_sizes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.mcp.target_tools import allen_build_manifest
from mousehash.mcp.transformation_tools import (
    extract_event_responses,
    extract_jpeg_sizes,
    extract_vit_features,
)


def _bootstrap_allen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
                      scene_set_id: str = "tf_test", n_images: int = 6) -> str:
    """Stub Allen + ViT, build a manifest via the MCP wrapper, return manifest_id."""
    monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
    from mousehash.artifacts import paths as _paths
    _paths._load_dotenv_once.cache_clear()

    rng = np.random.default_rng(0)
    stub = rng.integers(0, 256, size=(n_images, 64, 64), dtype=np.uint16).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.targets.allen.manifest.fetch_natural_scene_template",
        lambda manifest_path=None: stub,
    )
    monkeypatch.setattr(
        "mousehash.targets.allen.loaders.fetch_natural_scene_template",
        lambda scene_set_id, manifest_path=None: stub,
    )

    # ViT stub for transformation tests
    def fake_vit(frames, model_name=None, batch_size=None, device=None):
        from scipy.special import softmax
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(len(frames), 1000)).astype(np.float32)
        return logits, softmax(logits, axis=1).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.transformations.feature_extraction.run_vit_on_frames", fake_vit
    )

    return allen_build_manifest(scene_set_id=scene_set_id)["manifest_id"]


class TestExtractVitFeatures:
    def test_returns_view_id_and_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_allen(monkeypatch, tmp_path, data_root_tmp)
        result = extract_vit_features(manifest_id=manifest_id)
        assert result["view_id"].startswith("view_")
        assert Path(result["artifact_path"]).exists()
        assert result["summary"]["n_images"] == 6
        assert result["summary"]["n_classes"] == 1000

    def test_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_allen(monkeypatch, tmp_path, data_root_tmp)
        r1 = extract_vit_features(manifest_id=manifest_id)
        r2 = extract_vit_features(manifest_id=manifest_id)
        assert r1["from_cache"] is False
        assert r2["from_cache"] is True
        assert r1["view_id"] == r2["view_id"]

    def test_unknown_manifest_returns_structured_error(self, data_root_tmp: Path):
        result = extract_vit_features(manifest_id="mf_doesnotexist")
        assert result["type"] == "ManifestNotFoundError"


class TestExtractJpegSizes:
    def test_returns_view_id_and_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_allen(monkeypatch, tmp_path, data_root_tmp)
        result = extract_jpeg_sizes(manifest_id=manifest_id, qualities=[25, 75])
        assert result["view_id"].startswith("view_")
        assert result["summary"]["n_images"] == 6
        assert result["summary"]["qualities"] == [25, 75]
        assert result["summary"]["n_qualities"] == 2

    def test_default_qualities_used_when_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        from mousehash.transformations.image_compression import DEFAULT_JPEG_QUALITIES
        manifest_id = _bootstrap_allen(monkeypatch, tmp_path, data_root_tmp)
        result = extract_jpeg_sizes(manifest_id=manifest_id)
        assert tuple(result["summary"]["qualities"]) == tuple(DEFAULT_JPEG_QUALITIES)

    def test_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_allen(monkeypatch, tmp_path, data_root_tmp)
        r1 = extract_jpeg_sizes(manifest_id=manifest_id, qualities=[50])
        r2 = extract_jpeg_sizes(manifest_id=manifest_id, qualities=[50])
        assert r1["from_cache"] is False
        assert r2["from_cache"] is True
        assert r1["view_id"] == r2["view_id"]

    def test_unknown_manifest_returns_structured_error(self, data_root_tmp: Path):
        result = extract_jpeg_sizes(manifest_id="mf_doesnotexist")
        assert result["type"] == "ManifestNotFoundError"


class TestExtractEventResponses:
    def _stub_core(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Replace extract_event_response_view with a fake (view, summary) pair."""
        from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind

        fake_dir = tmp_path / "fake_events"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake_view = AnalysisView.new(
            kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
            manifest_id="mf_fake",
            shape=[10, 118],
            axes={"observations": "neurons_across_sessions", "features": "natural_scene_images"},
            source_roles=["neural_data", "stimuli"],
            transformation_lineage=["input:abc", "event_probability:max>0.0"],
            artifact_path=str(fake_dir),
            summary={"n_total_neurons": 10, "n_images": 118},
        )
        fake_summary = {
            "n_total_neurons": 10, "n_sessions_kept": 2, "n_sessions_skipped": 0,
            "n_images": 118, "n_donors": 2, "n_containers": 2,
            "from_cache": False,
            "artifacts": {"event_probabilities": str(fake_dir / "event_probabilities.npy")},
        }
        monkeypatch.setattr(
            "mousehash.mcp.transformation_tools.extract_event_response_view",
            lambda manifest, **kw: (fake_view, fake_summary),
        )

    def test_returns_view_id_and_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_allen(monkeypatch, tmp_path, data_root_tmp)
        self._stub_core(monkeypatch, tmp_path)
        result = extract_event_responses(manifest_id=manifest_id)
        assert result["view_id"].startswith("view_")
        assert Path(result["artifact_path"]).exists()
        assert result["summary"]["n_total_neurons"] == 10
        assert result["summary"]["n_sessions_kept"] == 2
        assert result["from_cache"] is False

    def test_unknown_manifest_returns_structured_error(self, data_root_tmp: Path):
        result = extract_event_responses(manifest_id="mf_doesnotexist")
        assert result["type"] == "ManifestNotFoundError"
