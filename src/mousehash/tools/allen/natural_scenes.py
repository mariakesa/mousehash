from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from mousehash.artifacts.paths import stimuli_root
from mousehash.utils.hashing import sha1_file

logger = logging.getLogger(__name__)

_DATASET_NAME = "allen_brain_observatory"
_STIMULUS_NAME = "natural_scenes"


def ingest_natural_scenes(
    manifest_path: Path,
    scene_set_id: str,
    notes: str = "",
) -> dict:
    """Fetch Allen natural scenes and register them in DataJoint.

    Saves each stimulus frame as a PNG under
    ``<DATA_ROOT>/stimuli/<scene_set_id>/images/``, then inserts records into
    ``AllenNaturalSceneSet`` and ``AllenNaturalSceneImage``.  Safe to re-run:
    existing files are not overwritten and DataJoint inserts use
    ``skip_duplicates=True``.

    Args:
        manifest_path: Path to the AllenSDK BrainObservatoryCache manifest JSON.
        scene_set_id:  Identifier stored as the primary key in DataJoint.
        notes:         Free-text notes attached to the scene-set record.

    Returns:
        Summary dict with keys ``scene_set_id``, ``n_images``, ``image_dir``.
    """
    from mousehash.tools.allen.stimulus_fetch import fetch_natural_scene_template
    from mousehash.schema.stimuli import AllenNaturalSceneSet, AllenNaturalSceneImage

    logger.info("Fetching natural scene template via AllenSDK …")
    template = fetch_natural_scene_template(manifest_path)
    n_images, height, width = template.shape
    logger.info("Got %d images (%d × %d px)", n_images, height, width)

    image_dir = stimuli_root() / scene_set_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_records = []
    for idx in range(n_images):
        img_path = image_dir / f"scene_{idx:04d}.png"
        if not img_path.exists():
            Image.fromarray(template[idx], mode="L").save(img_path)
        sha1 = sha1_file(img_path)
        image_records.append(
            dict(
                scene_set_id=scene_set_id,
                image_idx=idx,
                image_path=str(img_path),
                height=height,
                width=width,
                image_sha1=sha1,
            )
        )
        if idx % 20 == 0:
            logger.info("  Processed image %d / %d", idx, n_images)

    logger.info("Inserting scene-set record into DataJoint …")
    AllenNaturalSceneSet.insert1(
        dict(
            scene_set_id=scene_set_id,
            dataset_name=_DATASET_NAME,
            stimulus_name=_STIMULUS_NAME,
            n_images=n_images,
            notes=notes,
        ),
        skip_duplicates=True,
    )
    AllenNaturalSceneImage.insert(image_records, skip_duplicates=True)
    logger.info(
        "Registered scene set '%s' with %d images.", scene_set_id, n_images
    )

    return dict(
        scene_set_id=scene_set_id,
        n_images=n_images,
        image_dir=str(image_dir),
    )
