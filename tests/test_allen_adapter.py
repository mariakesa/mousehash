"""Tests for the Allen adapter with AllenSDK mocked.

We do not call live AllenSDK in the test suite. `fetch_natural_scene_template`
and the BrainObservatoryCache calls are monkeypatched to return tiny stub data
so we can exercise the manifest/catalog plumbing without network or HDF5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mousehash.core.manifests import RoleManifest
from mousehash.core.role_bundle import RoleStatus
from mousehash.targets.allen.client import resolve_manifest_path
from mousehash.core.errors import MouseHashError
from mousehash.targets.allen.adapter import AllenAdapter
from mousehash.targets.allen.manifest import (
    build_natural_scenes_manifest,
    load_image_catalog,
)
from mousehash.targets.base import TargetAdapter


# ---------- resolve_manifest_path precedence ----------

class TestResolveManifestPath:
    def test_explicit_arg_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "from_env.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        explicit = tmp_path / "from_arg.json"
        assert resolve_manifest_path(explicit) == explicit.resolve()

    def test_env_used_when_no_arg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        env_path = tmp_path / "from_env.json"
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(env_path))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        assert resolve_manifest_path() == env_path.resolve()

    def test_raises_when_neither_set(self):
        with pytest.raises(MouseHashError, match="manifest path"):
            resolve_manifest_path()

    def test_allen_data_accepted_as_alias(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # Only ALLEN_DATA is set; ALLEN_MANIFEST_PATH unset.
        env_path = tmp_path / "alias.json"
        monkeypatch.setenv("ALLEN_DATA", str(env_path))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        assert resolve_manifest_path() == env_path.resolve()

    def test_allen_manifest_path_preferred_over_alias(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        # Both set; ALLEN_MANIFEST_PATH wins.
        primary = tmp_path / "primary.json"
        alias = tmp_path / "alias.json"
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(primary))
        monkeypatch.setenv("ALLEN_DATA", str(alias))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        assert resolve_manifest_path() == primary.resolve()


# ---------- AllenAdapter Protocol compliance ----------

class TestAllenAdapterProtocol:
    def test_satisfies_target_adapter_protocol(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        adapter = AllenAdapter()
        assert isinstance(adapter, TargetAdapter)

    def test_target_name_is_allen(self):
        assert AllenAdapter.target_name == "allen"

    def test_default_list_datasets_returns_natural_scenes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        adapter = AllenAdapter()
        from mousehash.targets.base import DatasetQuery
        refs = adapter.list_datasets(DatasetQuery())
        assert len(refs) == 1
        assert refs[0].target == "allen"


# ---------- build_natural_scenes_manifest with mocked fetch ----------

def _patch_allen_fetch(monkeypatch: pytest.MonkeyPatch, n_images: int = 8) -> np.ndarray:
    """Replace fetch_natural_scene_template with a deterministic stub stack."""
    rng = np.random.default_rng(0)
    stub_stack = rng.integers(0, 256, size=(n_images, 64, 64), dtype=np.uint16).astype(np.float32)

    def fake_fetch(manifest_path=None):
        return stub_stack

    # Patch in BOTH places it's imported from
    monkeypatch.setattr("mousehash.targets.allen.manifest.fetch_natural_scene_template", fake_fetch)
    return stub_stack


class TestBuildNaturalScenesManifest:
    def test_returns_role_manifest_with_stimuli_present(
        self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_allen_fetch(monkeypatch, n_images=6)
        mf = build_natural_scenes_manifest(scene_set_id="t1")
        assert isinstance(mf, RoleManifest)
        assert mf.roles.stimuli.status == RoleStatus.PRESENT
        assert mf.roles.time_organization.status == RoleStatus.DERIVABLE
        assert mf.roles.metadata.status == RoleStatus.PRESENT
        assert mf.dataset.target == "allen"
        assert mf.dataset.dataset_id == "t1"

    def test_manifest_id_deterministic_per_scene_set(
        self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_allen_fetch(monkeypatch, n_images=4)
        a = build_natural_scenes_manifest(scene_set_id="repeat")
        b = build_natural_scenes_manifest(scene_set_id="repeat")
        assert a.manifest_id == b.manifest_id

    def test_writes_image_thumbnails_and_catalog(
        self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_allen_fetch(monkeypatch, n_images=5)
        build_natural_scenes_manifest(scene_set_id="cat_test")

        from mousehash.artifacts.paths import stimuli_root
        img_dir = stimuli_root() / "cat_test" / "images"
        thumbs = sorted(img_dir.glob("scene_*.png"))
        assert len(thumbs) == 5

        catalog = load_image_catalog("cat_test")
        assert catalog["n_images"] == 5
        assert catalog["dataset_name"] == "allen_brain_observatory"
        assert len(catalog["images"]) == 5
        # Every entry has a sha1
        for row in catalog["images"]:
            assert len(row["image_sha1"]) == 40

    def test_writes_manifest_yaml_to_manifests_root(
        self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_allen_fetch(monkeypatch, n_images=4)
        mf = build_natural_scenes_manifest(scene_set_id="yaml_test")
        from mousehash.artifacts.paths import manifests_root
        yaml_path = manifests_root() / f"{mf.manifest_id}.yaml"
        assert yaml_path.exists()
        # Round-trip the file
        rebuilt = RoleManifest.from_yaml(yaml_path.read_text())
        assert rebuilt.manifest_id == mf.manifest_id

    def test_reuses_existing_thumbnails(
        self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        """Calling twice should not re-save PNGs (idempotent)."""
        monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        _patch_allen_fetch(monkeypatch, n_images=4)
        build_natural_scenes_manifest(scene_set_id="idem")
        from mousehash.artifacts.paths import stimuli_root
        img_dir = stimuli_root() / "idem" / "images"
        first_mtime = (img_dir / "scene_0000.png").stat().st_mtime

        # Second call — thumbnails already exist, should be reused
        build_natural_scenes_manifest(scene_set_id="idem")
        assert (img_dir / "scene_0000.png").stat().st_mtime == first_mtime
