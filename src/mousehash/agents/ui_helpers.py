from __future__ import annotations

import html
import re
from pathlib import Path

_PLOT_PATH_RE = re.compile(r"plot=([^\s,]+\.html)\b")
_PLOT_PNG_PATH_RE = re.compile(r"plot_png=([^\s,]+\.(?:png|jpg|jpeg|webp))\b")
_ABSOLUTE_HTML_PATH_RE = re.compile(r"(/[^\s<>'\",]+\.html)\b")
_ABSOLUTE_IMAGE_PATH_RE = re.compile(r"(/[^\s<>'\",]+\.(?:png|jpg|jpeg|webp))\b")


def extract_plot_path(text: str) -> Path | None:
    match = _PLOT_PATH_RE.search(text)
    if match:
        return Path(match.group(1))

    fallback_match = _ABSOLUTE_HTML_PATH_RE.search(text)
    if fallback_match:
        return Path(fallback_match.group(1))

    return None


def extract_plot_png_path(text: str) -> Path | None:
    match = _PLOT_PNG_PATH_RE.search(text)
    if match:
        return Path(match.group(1))

    fallback_match = _ABSOLUTE_IMAGE_PATH_RE.search(text)
    if fallback_match:
        return Path(fallback_match.group(1))

    return None


def render_plot_iframe(plot_path: Path | str | None) -> str:
    if plot_path is None:
        return "<div style='padding:1rem;color:#666;'>No plot generated yet.</div>"

    path = Path(plot_path)
    if not path.exists():
        return (
            "<div style='padding:1rem;color:#a33;'>"
            f"Plot file not found: {html.escape(str(path))}"
            "</div>"
        )

    srcdoc = html.escape(path.read_text(encoding="utf-8"))
    return (
        "<iframe "
        "style='width:100%;height:760px;border:1px solid #ddd;border-radius:8px;background:white;' "
        f"srcdoc='{srcdoc}'></iframe>"
    )


def normalize_image_path(plot_png_path: Path | str | None) -> str | None:
    if plot_png_path is None:
        return None

    path = Path(plot_png_path)
    if not path.exists():
        return None

    from PIL import Image

    return Image.open(path).copy()