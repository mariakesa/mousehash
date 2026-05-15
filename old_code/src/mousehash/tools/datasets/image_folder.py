from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PIL import Image

from mousehash.artifacts.paths import stimuli_root
from mousehash.schema.stimuli import StimulusImage, StimulusSet
from mousehash.utils.hashing import sha1_file

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
THUMBNAIL_MAX_EDGE_PX = 160


def _iter_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _copy_thumbnail(source_path: Path, dest_path: Path) -> tuple[int, int]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as img:
        thumb = img.convert("RGB")
        thumb.thumbnail(
            (THUMBNAIL_MAX_EDGE_PX, THUMBNAIL_MAX_EDGE_PX),
            Image.Resampling.LANCZOS,
        )
        thumb.save(dest_path)
        return thumb.height, thumb.width


def ingest_image_folder(
    image_dir: Path,
    scene_set_id: str,
    dataset_name: str,
    stimulus_name: str = "image_folder",
    notes: str = "",
) -> dict:
    """Ingest a folder of image files into the generic MouseHash stimulus schema."""
    image_dir = Path(image_dir).expanduser().resolve()
    source_paths = _iter_image_paths(image_dir)
    if not source_paths:
        raise ValueError(f"No supported image files found in {image_dir}")

    managed_dir = stimuli_root() / scene_set_id / "images"
    managed_dir.mkdir(parents=True, exist_ok=True)

    image_records = []
    for idx, source_path in enumerate(source_paths):
        item_id = source_path.stem
        managed_path = managed_dir / f"{item_id}{source_path.suffix.lower()}"
        if not managed_path.exists():
            height, width = _copy_thumbnail(source_path, managed_path)
        else:
            with Image.open(managed_path) as img:
                width, height = img.size

        image_records.append(
            dict(
                scene_set_id=scene_set_id,
                image_idx=idx,
                image_path=str(managed_path),
                height=height,
                width=width,
                image_sha1=sha1_file(managed_path),
            )
        )

    StimulusSet.insert1(
        dict(
            scene_set_id=scene_set_id,
            dataset_name=dataset_name,
            stimulus_name=stimulus_name,
            n_images=len(image_records),
            notes=notes,
        ),
        skip_duplicates=True,
    )
    StimulusImage.insert(image_records, skip_duplicates=True)
    logger.info("Registered image-folder stimulus set '%s' with %d images.", scene_set_id, len(image_records))
    return dict(
        scene_set_id=scene_set_id,
        n_images=len(image_records),
        image_dir=str(managed_dir),
    )