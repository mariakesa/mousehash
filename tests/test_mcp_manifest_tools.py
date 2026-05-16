"""Tests for mcp/manifest_tools.py — get_manifest, list_runnable_tools, explain_tool_readiness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.mcp.manifest_tools import (
    explain_tool_readiness,
    get_manifest,
    list_runnable_tools,
)
from mousehash.mcp.target_tools import allen_build_manifest


def _bootstrap_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path, scene_set_id: str = "rd_test",
) -> str:
    """Build a real Allen manifest via the MCP wrapper; return manifest_id."""
    monkeypatch.setenv("ALLEN_MANIFEST_PATH", str(tmp_path / "boc.json"))
    from mousehash.artifacts import paths as _paths
    _paths._load_dotenv_once.cache_clear()

    rng = np.random.default_rng(0)
    stub = rng.integers(0, 256, size=(4, 64, 64), dtype=np.uint16).astype(np.float32)
    monkeypatch.setattr(
        "mousehash.targets.allen.manifest.fetch_natural_scene_template",
        lambda manifest_path=None: stub,
    )
    return allen_build_manifest(scene_set_id=scene_set_id)["manifest_id"]


class TestGetManifest:
    def test_returns_json_serializable_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_manifest(monkeypatch, tmp_path, data_root_tmp)
        result = get_manifest(manifest_id)
        assert result["manifest_id"] == manifest_id
        assert "dataset" in result
        assert "roles" in result
        # Roundtrip via json
        import json
        json.dumps(result)

    def test_unknown_id_returns_structured_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        _bootstrap_manifest(monkeypatch, tmp_path, data_root_tmp)
        result = get_manifest("mf_doesnotexist")
        assert "error" in result
        assert result["type"] == "ManifestNotFoundError"


class TestListRunnableTools:
    def test_lists_both_pca_and_nmf(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_manifest(monkeypatch, tmp_path, data_root_tmp)
        result = list_runnable_tools(manifest_id)
        assert result["manifest_id"] == manifest_id
        tool_names = {r["tool_name"] for r in result["readiness"]}
        assert {"run_pca", "run_nmf", "compare_groups_by_label"} <= tool_names

    def test_readiness_includes_family_and_produces(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_manifest(monkeypatch, tmp_path, data_root_tmp)
        result = list_runnable_tools(manifest_id)
        # Each readiness entry has family + produces; specific families vary.
        for entry in result["readiness"]:
            assert "family" in entry and isinstance(entry["family"], str)
            assert isinstance(entry["produces"], list) and len(entry["produces"]) > 0
        # factor_models still represented
        families = {r["family"] for r in result["readiness"]}
        assert "factor_models" in families
        assert "comparison" in families

    def test_pca_runnable_against_stimuli_only_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        # Allen natural-scenes manifest has stimuli + time_organization + metadata.
        # PCA contract requires only stimuli -> runnable.
        manifest_id = _bootstrap_manifest(monkeypatch, tmp_path, data_root_tmp)
        result = list_runnable_tools(manifest_id)
        pca_entry = next(r for r in result["readiness"] if r["tool_name"] == "run_pca")
        assert pca_entry["runnable"] is True


class TestExplainToolReadiness:
    def test_returns_contract_detail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_manifest(monkeypatch, tmp_path, data_root_tmp)
        result = explain_tool_readiness("run_pca", manifest_id)
        assert result["tool_name"] == "run_pca"
        assert result["runnable"] is True
        assert result["contract"]["family"] == "factor_models"
        assert "stimuli" in result["contract"]["required_roles"]
        assert "X" in result["contract"]["consumes_views"]
        assert result["contract"]["consumes_views"]["X"] == "observation_by_feature"

    def test_unknown_tool_returns_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_root_tmp: Path,
    ):
        manifest_id = _bootstrap_manifest(monkeypatch, tmp_path, data_root_tmp)
        result = explain_tool_readiness("run_nonexistent", manifest_id)
        assert result["type"] == "UnknownToolError"
        assert "known_tools" in result["details"]
