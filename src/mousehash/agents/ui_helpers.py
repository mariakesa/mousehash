from __future__ import annotations

import html
import re
from pathlib import Path

_PLOT_PATH_RE = re.compile(r"plot=([^\s,]+\.html)\b")
_PLOT_PNG_PATH_RE = re.compile(r"plot_png=([^\s,]+\.(?:png|jpg|jpeg|webp))\b")
_ABSOLUTE_HTML_PATH_RE = re.compile(r"(/[^\s<>'\",]+\.html)\b")
_ABSOLUTE_IMAGE_PATH_RE = re.compile(r"(/[^\s<>'\",]+\.(?:png|jpg|jpeg|webp))\b")

# Every absolute path with one of these suffixes is treated as a generated
# artifact and surfaced to the user as a clickable link. Order doesn't matter
# but longer suffixes must precede their prefixes (e.g. .json before .js) if
# that ever becomes relevant.
_ARTIFACT_SUFFIXES = ("html", "png", "jpg", "jpeg", "webp", "svg", "json", "npy", "csv", "parquet")
_ARTIFACT_PATH_RE = re.compile(
    r"(/[^\s<>'\",]+?\.(?:" + "|".join(_ARTIFACT_SUFFIXES) + r"))\b"
)


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


def extract_all_artifact_paths(text: str) -> list[Path]:
    """Pull every absolute path with a known artifact suffix out of a tool reply.

    Returns paths in first-seen order with duplicates removed. Existence is
    not checked here — the caller decides what to do with missing files.
    """
    seen: dict[str, Path] = {}
    for match in _ARTIFACT_PATH_RE.finditer(text):
        path = Path(match.group(1))
        seen.setdefault(str(path), path)
    return list(seen.values())


_FRIENDLY_KIND = {
    "html": "report",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "webp": "image",
    "svg": "image",
    "json": "summary",
    "npy": "array",
    "csv": "table",
    "parquet": "table",
}


def format_artifact_links_markdown(paths: list[Path]) -> str:
    """Render a markdown bullet list of clickable links to gradio's file route.

    Gradio rewrites ``/file=<abs>`` and ``/gradio_api/file=<abs>`` links when
    the directory is in ``allowed_paths``. We use the older ``/file=`` form
    since both gradio 4.x and 5.x accept it.

    Returns an empty string when there are no paths.
    """
    if not paths:
        return ""

    lines = ["", "**Generated files:**"]
    for path in paths:
        suffix = path.suffix.lower().lstrip(".")
        kind = _FRIENDLY_KIND.get(suffix, "file")
        exists_marker = "" if path.exists() else " _(not found)_"
        lines.append(
            f"- [{path.name}](/file={path}) — {kind} at `{path}`{exists_marker}"
        )
    return "\n".join(lines)