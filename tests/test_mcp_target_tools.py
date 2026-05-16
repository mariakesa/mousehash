"""Tests for mcp/target_tools.py with AllenSDK fetch monkeypatched."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.mcp.target_tools import allen_build_manifest, allen_list_datasets


def _patch_allen_fetch(monkeypatch: pytest.MonkeyPatch, n_images: int = 6) -> None:
    """Stub Allen fetch so tests don't need network / AllenSDK."""
    rng = np.random.default_rng(0)
    stub = rng.integers(0, 256, size=(n_images, 64, 64), dtype=np.uint16).astype(np.float32)

    def fake_fetch(manifest_path=None):
        return stub

    monkeypatch.setattr("mousehash.targets.allen.manifest.fetch_natural_scene_template", fake_fetch)


class TestAllenListDatasets:
    def test_returns_one_dataset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        result = allen_list_datasets()
        assert result["target"] == "allen"
        assert len(result["datasets"]) == 1
        ds = result["datasets"][0]
        assert ds["target"] == "allen"
        # dataset_id should be a string (e.g. allen_natural_scenes_v1)
        assert isinstance(ds["dataset_id"], str) and ds["dataset_id"]

    def test_json_serializable(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        import json
        json.dumps(allen_list_datasets())  # must not raise


class TestAllenBuildManifest:
    def test_returns_manifest_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_allen_fetch(monkeypatch, n_images=4)
        result = allen_build_manifest(scene_set_id="t1")
        assert result["manifest_id"].startswith("mf_")
        assert "stimuli" in result["satisfied_roles"]
        assert result["manifest_yaml_path"].endswith(".yaml")
        assert Path(result["manifest_yaml_path"]).exists()

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_allen_fetch(monkeypatch, n_images=4)
        a = allen_build_manifest(scene_set_id="idem")
        b = allen_build_manifest(scene_set_id="idem")
        assert a["manifest_id"] == b["manifest_id"]

    def test_explicit_path_arg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        _patch_allen_fetch(monkeypatch, n_images=4)
        # No env, explicit path
        result = allen_build_manifest(scene_set_id="t2", allen_manifest_path=str(tmp_path / "boc.json"))
        assert result["manifest_id"].startswith("mf_")

    def test_missing_env_returns_structured_error(self, monkeypatch: pytest.MonkeyPatch, data_root_tmp: Path):
        # No ALLEN_MANIFEST_PATH / ALLEN_DATA env, no arg -> structured error via @mcp_safe.
        result = allen_build_manifest(scene_set_id="oops")
        assert "error" in result
        assert "manifest path" in result["error"].lower()
