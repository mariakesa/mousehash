"""Tests for mcp/view_tools.py — list_views + inspect_view."""

from __future__ import annotations

from pathlib import Path

from mousehash.artifacts.io import save_json
from mousehash.artifacts.paths import artifact_root
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.ids import ManifestId
from mousehash.mcp.view_tools import inspect_view, list_views


def _stub_view(scope: str, lineage: list[str]) -> AnalysisView:
    out = artifact_root() / "representations" / scope / "hash_x"
    out.mkdir(parents=True, exist_ok=True)
    view = AnalysisView.new(
        kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
        manifest_id=ManifestId("mf_stub"),
        shape=[5, 10],
        axes={"observations": "x", "features": "y"},
        source_roles=["stimuli"],
        transformation_lineage=lineage,
        artifact_path=str(out),
    )
    save_json(out / "view.json", view.model_dump(mode="json"))
    return view


class TestListViews:
    def test_empty_cache(self, data_root_tmp: Path):
        result = list_views()
        assert result == {"n_views": 0, "views": []}

    def test_one_view(self, data_root_tmp: Path):
        v = _stub_view("set_a", ["transform_a"])
        result = list_views()
        assert result["n_views"] == 1
        assert result["views"][0]["view_id"] == v.view_id

    def test_multiple_views(self, data_root_tmp: Path):
        v1 = _stub_view("set_a", ["a"])
        v2 = _stub_view("set_b", ["b"])
        result = list_views()
        ids = {v["view_id"] for v in result["views"]}
        assert v1.view_id in ids and v2.view_id in ids
        assert result["n_views"] == 2


class TestInspectView:
    def test_returns_full_view_payload(self, data_root_tmp: Path):
        v = _stub_view("set_a", ["transform_x"])
        result = inspect_view(v.view_id)
        assert result["view_id"] == v.view_id
        assert result["kind"] == "observation_by_feature"
        assert result["shape"] == [5, 10]
        assert result["axes"] == {"observations": "x", "features": "y"}
        assert "transform_x" in result["transformation_lineage"]

    def test_unknown_id_returns_structured_error(self, data_root_tmp: Path):
        result = inspect_view("view_doesnotexist")
        assert "error" in result
        assert result["type"] == "ViewNotFoundError"
