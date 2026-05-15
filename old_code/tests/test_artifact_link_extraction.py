from __future__ import annotations

from pathlib import Path

from mousehash.agents.ui_helpers import (
    extract_all_artifact_paths,
    format_artifact_links_markdown,
)


def test_extracts_html_and_png_from_plot_tool_output() -> None:
    text = (
        "Cell plot complete. cell_specimen_id=1, "
        "plot=/data/reports/cell_1.html, plot_png=/data/reports/cell_1.png"
    )
    paths = extract_all_artifact_paths(text)
    assert Path("/data/reports/cell_1.html") in paths
    assert Path("/data/reports/cell_1.png") in paths


def test_extracts_paths_from_blahml_run_multiline_output() -> None:
    text = (
        "BlahML dispatched run_pca.\n"
        "  tool_run_spec_id      = abc123\n"
        "  decomposition_spec_id = blahml_abc123\n"
        "  components_path = /home/maria/data/decompositions/x/components.npy\n"
        "  scores_path = /home/maria/data/decompositions/x/scores.npy\n"
        "  summary_path = /home/maria/data/decompositions/x/summary.json\n"
        "  summary = {\"method\": \"pca\"}"
    )
    paths = extract_all_artifact_paths(text)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".json", ".npy", ".npy"]


def test_deduplicates_paths_seen_more_than_once() -> None:
    text = "report at /tmp/r.html, also see /tmp/r.html for details"
    paths = extract_all_artifact_paths(text)
    assert paths == [Path("/tmp/r.html")]


def test_format_artifact_links_renders_clickable_markdown() -> None:
    paths = [
        Path("/data/reports/nmf_explorer.html"),
        Path("/data/reports/summary.json"),
    ]
    md = format_artifact_links_markdown(paths)
    assert "**Generated files:**" in md
    assert "[nmf_explorer.html](/file=/data/reports/nmf_explorer.html)" in md
    assert "[summary.json](/file=/data/reports/summary.json)" in md
    assert "report" in md      # html → report kind
    assert "summary" in md     # json → summary kind


def test_format_artifact_links_marks_missing_files() -> None:
    md = format_artifact_links_markdown([Path("/no/such/file.html")])
    assert "_(not found)_" in md


def test_format_artifact_links_returns_empty_string_for_no_paths() -> None:
    assert format_artifact_links_markdown([]) == ""


def test_extracts_paths_from_run_reports_multiline_output() -> None:
    text = (
        "Reports built:\n"
        "  /data/reports/scene_v1/spec_a/decomp_a/pca_explorer.html\n"
        "  /data/reports/scene_v1/spec_a/decomp_b/nmf_explorer.html"
    )
    paths = extract_all_artifact_paths(text)
    names = sorted(p.name for p in paths)
    assert names == ["nmf_explorer.html", "pca_explorer.html"]
