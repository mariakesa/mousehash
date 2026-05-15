from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


NATURAL_SCENES_EXPERIMENT_ID = 501559087
STIMULUS_NAME = "natural_scenes"
THUMBNAIL_MAX_EDGE_PX = 160


def fetch_natural_scene_template(manifest_path: Path) -> np.ndarray:
    """Return raw Allen natural scene template.

    Shape is usually (n_images, height, width).
    Dtype/range depends on AllenSDK/cache version, so normalize before saving.

    Keeps the same API contract as the previous MouseHash pipeline:
        fetch_natural_scene_template(manifest_path: Path) -> np.ndarray
    """
    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    except ImportError as e:
        raise ImportError(
            "allensdk is required for Allen ingestion.\n"
            "Install it with: pip install allensdk"
        ) from e

    manifest_path = Path(manifest_path)

    boc = BrainObservatoryCache(manifest_file=str(manifest_path))
    exps = boc.get_ophys_experiments(stimuli=[STIMULUS_NAME])

    if not exps:
        raise ValueError(
            f"No natural-scene experiments found in manifest at {manifest_path}"
        )

    exp_ids = {exp["id"] for exp in exps}

    if NATURAL_SCENES_EXPERIMENT_ID not in exp_ids:
        raise ValueError(
            "Configured natural-scenes experiment "
            f"{NATURAL_SCENES_EXPERIMENT_ID} is unavailable in manifest at {manifest_path}"
        )

    # AllenSDK-version-compatible path:
    #
    # Some AllenSDK versions expose:
    #     boc.get_stimulus_template("natural_scenes")
    #
    # Your installed version does not, so we fall back to the dataset-level method.
    if hasattr(boc, "get_stimulus_template"):
        template = boc.get_stimulus_template(STIMULUS_NAME)
    else:
        dataset = boc.get_ophys_experiment_data(NATURAL_SCENES_EXPERIMENT_ID)
        template = dataset.get_stimulus_template(STIMULUS_NAME)

    template = np.asarray(template)

    if template.ndim != 3:
        raise ValueError(
            f"Expected natural-scenes template with shape (n_images, H, W), "
            f"got {template.shape}"
        )

    return template


def image_to_uint8_global(
    img: np.ndarray,
    global_lo: float,
    global_hi: float,
) -> np.ndarray:
    """Convert one Allen natural-scene image using global template min/max.

    This matches the test script's `global_normalized` output.
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
) -> tuple[int, int]:
    """Save natural scene template as globally normalized grayscale PNGs.

    Returns the saved image size as ``(height, width)``.
    """

    template = np.asarray(template)

    if template.ndim != 3:
        raise ValueError(
            f"Expected template with shape (n_images, H, W), got {template.shape}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_lo = float(np.nanmin(template))
    global_hi = float(np.nanmax(template))
    saved_size: tuple[int, int] | None = None

    for i, img in enumerate(template):
        img_u8 = image_to_uint8_global(
            img,
            global_lo=global_lo,
            global_hi=global_hi,
        )

        out_path = output_dir / f"{prefix}_{i:04d}.png"

        pil_img = Image.fromarray(img_u8).convert("L")
        pil_img.thumbnail(
            (THUMBNAIL_MAX_EDGE_PX, THUMBNAIL_MAX_EDGE_PX),
            Image.Resampling.LANCZOS,
        )
        pil_img.save(out_path)

        if saved_size is None:
            saved_size = (pil_img.height, pil_img.width)

    if saved_size is None:
        raise ValueError("Cannot save thumbnails from an empty template")

    return saved_size