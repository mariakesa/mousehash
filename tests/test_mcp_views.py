"""Tests for mcp/views.py — find_view_by_id + list_view_records."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousehash.artifacts.io import save_json
from mousehash.artifacts.paths import artifact_root
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.ids import ManifestId
from mousehash.mcp.errors import ViewNotFoundError
from mousehash.mcp.views import find_view_by_id, list_view_records


def _write_view(out_dir: Path, manifest_id: str, lineage: list[str]) -> AnalysisView:
    """Persist a synthetic view.json at out_dir to mimic a cached computation."""
    view = AnalysisView.new(
        kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
        manifest_id=ManifestId(manifest_id),
        shape=[10, 5],
        axes={"observations": "stim", "features": "imagenet"},
        source_roles=["stimuli"],
        transformation_lineage=lineage,
        artifact_path=str(out_dir),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "view.json", view.model_dump(mode="json"))
    return view


class TestFindViewById:
    def test_returns_matching_view(self, data_root_tmp: Path):
        v1 = _write_view(artifact_root() / "representations" / "set_a" / "h1", "mf_a", ["x"])
        found = find_view_by_id(v1.view_id)
        assert found.view_id == v1.view_id
        assert found.shape == [10, 5]

    def test_raises_when_no_match(self, data_root_tmp: Path):
        _write_view(artifact_root() / "representations" / "set_a" / "h1", "mf_a", ["x"])
        with pytest.raises(ViewNotFoundError, match="view_nonexistent"):
            find_view_by_id("view_nonexistent")

    def test_searches_across_multiple_directories(self, data_root_tmp: Path):
        # Three views in three different families/scopes
        v1 = _write_view(artifact_root() / "representations" / "set_a" / "h1", "mf", ["a"])
        v2 = _write_view(artifact_root() / "compression" / "set_b" / "h2", "mf", ["b"])
        v3 = _write_view(artifact_root() / "decompositions" / "src_hash" / "h3", "mf", ["c"])
        for v in (v1, v2, v3):
            assert find_view_by_id(v.view_id).view_id == v.view_id

    def test_empty_cache_raises_cleanly(self, data_root_tmp: Path):
        with pytest.raises(ViewNotFoundError):
            find_view_by_id("view_anything")

    def test_skips_malformed_view_json(self, data_root_tmp: Path):
        malformed = artifact_root() / "broken" / "view.json"
        malformed.parent.mkdir(parents=True, exist_ok=True)
        malformed.write_text("{not json", encoding="utf-8")
        v = _write_view(artifact_root() / "representations" / "ok" / "h1", "mf", ["x"])
        # Malformed file is silently skipped; real one is still found.
        assert find_view_by_id(v.view_id).view_id == v.view_id


class TestListViewRecords:
    def test_lists_all_views(self, data_root_tmp: Path):
        v1 = _write_view(artifact_root() / "representations" / "set_a" / "h1", "mf_a", ["x"])
        v2 = _write_view(artifact_root() / "compression" / "set_b" / "h2", "mf_b", ["y"])
        records = list_view_records()
        ids = {r["view_id"] for r in records}
        assert v1.view_id in ids and v2.view_id in ids

    def test_record_shape(self, data_root_tmp: Path):
        v = _write_view(artifact_root() / "representations" / "set_a" / "h1", "mf_a", ["x"])
        record = next(r for r in list_view_records() if r["view_id"] == v.view_id)
        assert set(record.keys()) >= {"view_id", "kind", "manifest_id", "lineage_hash", "shape", "artifact_path"}
        assert record["kind"] == "observation_by_feature"

    def test_empty_cache_returns_empty_list(self, data_root_tmp: Path):
        assert list_view_records() == []
