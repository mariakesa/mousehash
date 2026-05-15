"""Natural-scene stimulus access for the Allen Brain Observatory.

This module is the Allen-side of the stimulus path: fetch the 118 natural
scene templates from AllenSDK, normalize them globally, and write small
grayscale thumbnails to disk for downstream ViT / report use.

The math + I/O here is a port of the old `tools/allen/stimulus_fetch.py`,
with three changes:
  - paths come from `mousehash.artifacts.paths` rather than `mousehash.config`,
  - AllenSDK import is via `targets.allen.client.require_allensdk`,
  - the function returns a small dict the manifest builder can consume instead
    of inserting DataJoint rows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mousehash.targets.allen.client import get_brain_observatory_cache, resolve_manifest_path

logger = logging.getLogger(__name__)

NATURAL_SCENES_EXPERIMENT_ID = 501559087
STIMULUS_NAME = "natural_scenes"
THUMBNAIL_MAX_EDGE_PX = 160


def fetch_natural_scene_template(manifest_path: Path | str | None = None) -> np.ndarray:
    """Return the Allen natural scene template as a (n_images, H, W) numpy array.

    Tries the modern `boc.get_stimulus_template(name)` API first; falls back to
    fetching one experiment and asking it for the template, which works on
    older AllenSDK versions.
    """
    resolved = resolve_manifest_path(manifest_path)
    boc = get_brain_observatory_cache(str(resolved))

    exps = boc.get_ophys_experiments(stimuli=[STIMULUS_NAME])
    if not exps:
        raise ValueError(f"No natural-scene experiments in AllenSDK manifest {resolved}")
    if NATURAL_SCENES_EXPERIMENT_ID not in {exp["id"] for exp in exps}:
        raise ValueError(
            f"Reference experiment {NATURAL_SCENES_EXPERIMENT_ID} not present in manifest {resolved}"
        )

    if hasattr(boc, "get_stimulus_template"):
        template = boc.get_stimulus_template(STIMULUS_NAME)
    else:
        dataset = boc.get_ophys_experiment_data(NATURAL_SCENES_EXPERIMENT_ID)
        template = dataset.get_stimulus_template(STIMULUS_NAME)

    template = np.asarray(template)
    if template.ndim != 3:
        raise ValueError(f"Expected (n_images, H, W) template, got shape {template.shape}")
    return template


def image_to_uint8_global(img: np.ndarray, global_lo: float, global_hi: float) -> np.ndarray:
    """Normalize one grayscale frame to uint8 using global template min/max.

    Global (rather than per-image) normalization preserves cross-image
    luminance differences, which matters for downstream ViT consistency.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {img.shape}")
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if global_hi == global_lo:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - global_lo) / (global_hi - global_lo)
    img = np.clip(img, 0.0, 1.0)
    return np.round(img * 255.0).astype(np.uint8)


def save_natural_scene_images(
    template: np.ndarray,
    output_dir: Path,
    prefix: str = "scene",
    thumbnail_max_edge_px: int = THUMBNAIL_MAX_EDGE_PX,
) -> tuple[int, int]:
    """Save each frame as a globally-normalized grayscale PNG. Returns (height, width)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save natural scene images. Install '.[allen]'.") from exc

    template = np.asarray(template)
    if template.ndim != 3:
        raise ValueError(f"Expected (n_images, H, W) template, got shape {template.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_lo = float(np.nanmin(template))
    global_hi = float(np.nanmax(template))
    saved_size: tuple[int, int] | None = None

    for i, img in enumerate(template):
        img_u8 = image_to_uint8_global(img, global_lo=global_lo, global_hi=global_hi)
        out_path = output_dir / f"{prefix}_{i:04d}.png"
        pil_img = Image.fromarray(img_u8).convert("L")
        pil_img.thumbnail((thumbnail_max_edge_px, thumbnail_max_edge_px), Image.Resampling.LANCZOS)
        pil_img.save(out_path)
        if saved_size is None:
            saved_size = (pil_img.height, pil_img.width)

    if saved_size is None:
        raise ValueError("Cannot save thumbnails from an empty template")
    return saved_size
