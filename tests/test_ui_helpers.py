from __future__ import annotations

from pathlib import Path

from PIL import Image

from mousehash.agents.ui_helpers import (
    extract_plot_path,
    extract_plot_png_path,
    normalize_image_path,
    render_plot_iframe,
)


def test_extract_plot_path_returns_html_path() -> None:
    path = extract_plot_path(
        "Vanilla cell plot complete. cell_specimen_id=1, experiment_id=2, plot=/tmp/demo_plot.html"
    )

    assert path == Path("/tmp/demo_plot.html")


def test_extract_plot_path_returns_none_when_missing() -> None:
    assert extract_plot_path("No plot was generated") is None


def test_extract_plot_path_finds_html_path_in_prose() -> None:
    path = extract_plot_path(
        "Interactive HTML: /tmp/demo_plot.html\nStatic PNG: /tmp/demo_plot.png"
    )

    assert path == Path("/tmp/demo_plot.html")


def test_extract_plot_png_path_returns_image_path() -> None:
    path = extract_plot_png_path(
        "Vanilla cell plot complete. plot=/tmp/demo_plot.html, plot_png=/tmp/demo_plot.png"
    )

    assert path == Path("/tmp/demo_plot.png")


def test_extract_plot_png_path_finds_image_path_in_prose() -> None:
    path = extract_plot_png_path(
        "Interactive HTML: /tmp/demo_plot.html\nStatic PNG: /tmp/demo_plot.png"
    )

    assert path == Path("/tmp/demo_plot.png")


def test_render_plot_iframe_embeds_existing_html(tmp_path: Path) -> None:
    plot_path = tmp_path / "plot.html"
    plot_path.write_text("<html><body><div>demo plot</div></body></html>", encoding="utf-8")

    iframe = render_plot_iframe(plot_path)

    assert "iframe" in iframe
    assert "demo plot" in iframe


def test_render_plot_iframe_handles_missing_file(tmp_path: Path) -> None:
    iframe = render_plot_iframe(tmp_path / "missing.html")

    assert "Plot file not found" in iframe


def test_normalize_image_path_returns_existing_path(tmp_path: Path) -> None:
    image_path = tmp_path / "plot.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)

    image = normalize_image_path(image_path)

    assert image is not None
    assert image.size == (4, 4)