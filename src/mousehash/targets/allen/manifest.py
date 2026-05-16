"""Build a MouseHash RoleManifest for an Allen natural-scenes session.

Identity that used to live in DataJoint tables (`AllenNaturalSceneSet`,
`AllenNaturalSceneImage`) is captured two ways:
  - high-level role evidence on the `RoleManifest`,
  - per-image SHA-1 catalog as a sidecar JSON next to the saved thumbnails.

Both live under `MOUSEHASH_*` paths and can be re-derived from disk
deterministically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mousehash.artifacts.hashes import sha1_file
from mousehash.artifacts.io import save_json
from mousehash.artifacts.paths import manifests_root, stimuli_root
from mousehash.core.ids import DatasetId, TargetName
from mousehash.core.manifests import DatasetRef, RoleManifest
from mousehash.core.role_bundle import (
    MetadataRole,
    NeuralDataRole,
    RoleBundle,
    RoleConfidence,
    RoleEvidence,
    RoleStatus,
    StimuliRole,
    TimeOrganizationRole,
)
from mousehash.targets.allen.stimuli import (
    NATURAL_SCENES_EXPERIMENT_ID,
    STIMULUS_NAME,
    fetch_natural_scene_template,
    save_natural_scene_images,
)

logger = logging.getLogger(__name__)

ALLEN_TARGET = TargetName("allen")
ALLEN_DATASET_NAME = "allen_brain_observatory"


def _image_catalog_path(scene_set_id: str) -> Path:
    return stimuli_root() / scene_set_id / "image_catalog.json"


def build_natural_scenes_manifest(
    scene_set_id: str,
    manifest_path: Path | str | None = None,
    notes: str | None = None,
) -> RoleManifest:
    """Fetch the Allen natural-scene template, save thumbnails, write the role manifest.

    Idempotent on disk: if thumbnails for `scene_set_id` already exist they are
    reused and only the manifest + catalog JSON are (re)written.
    """
    template = fetch_natural_scene_template(manifest_path)
    n_images, raw_h, raw_w = template.shape
    logger.info("Fetched Allen natural scenes: %d images (%d x %d)", n_images, raw_h, raw_w)

    image_dir = stimuli_root() / scene_set_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(image_dir.glob("scene_*.png"))
    if existing and len(existing) == n_images:
        from PIL import Image  # type: ignore[import]
        with Image.open(existing[0]) as first:
            thumb_w, thumb_h = first.size
        logger.info("Reusing %d existing thumbnails under %s", n_images, image_dir)
    else:
        thumb_h, thumb_w = save_natural_scene_images(template, image_dir)
        logger.info("Saved %d thumbnails (%d x %d) to %s", n_images, thumb_h, thumb_w, image_dir)

    catalog_rows = []
    for idx in range(n_images):
        img_path = image_dir / f"scene_{idx:04d}.png"
        catalog_rows.append(
            {
                "image_idx": idx,
                "image_path": str(img_path),
                "image_sha1": sha1_file(img_path),
                "height": thumb_h,
                "width": thumb_w,
            }
        )
    catalog_path = _image_catalog_path(scene_set_id)
    save_json(
        catalog_path,
        {
            "scene_set_id": scene_set_id,
            "dataset_name": ALLEN_DATASET_NAME,
            "stimulus_name": STIMULUS_NAME,
            "n_images": n_images,
            "reference_experiment_id": NATURAL_SCENES_EXPERIMENT_ID,
            "raw_shape": [int(n_images), int(raw_h), int(raw_w)],
            "thumb_shape": [int(thumb_h), int(thumb_w)],
            "images": catalog_rows,
        },
    )

    roles = RoleBundle(
        stimuli=StimuliRole(
            status=RoleStatus.PRESENT,
            confidence=RoleConfidence.HIGH,
            evidence=[
                RoleEvidence(
                    status=RoleStatus.PRESENT,
                    confidence=RoleConfidence.HIGH,
                    source="allensdk",
                    path=f"natural_scenes:experiment_id={NATURAL_SCENES_EXPERIMENT_ID}",
                    notes=f"{n_images} grayscale natural scene templates, saved as PNG thumbnails.",
                ),
                RoleEvidence(
                    status=RoleStatus.PRESENT,
                    confidence=RoleConfidence.HIGH,
                    source="filesystem",
                    path=str(catalog_path),
                    notes="Per-image SHA-1 catalog.",
                ),
            ],
        ),
        time_organization=TimeOrganizationRole(
            status=RoleStatus.DERIVABLE,
            confidence=RoleConfidence.HIGH,
            evidence=[
                RoleEvidence(
                    status=RoleStatus.DERIVABLE,
                    confidence=RoleConfidence.HIGH,
                    source="allensdk",
                    path="get_stimulus_table('natural_scenes')",
                    notes="Per-experiment stimulus table with start/end/frame columns.",
                ),
            ],
        ),
        neural_data=NeuralDataRole(
            status=RoleStatus.DERIVABLE,
            confidence=RoleConfidence.HIGH,
            evidence=[
                RoleEvidence(
                    status=RoleStatus.DERIVABLE,
                    confidence=RoleConfidence.HIGH,
                    source="allensdk",
                    path="boc.get_ophys_experiment_events(session_id)",
                    notes="Per-session L0 event arrays, shape (n_cells, n_timestamps).",
                ),
            ],
        ),
        metadata=MetadataRole(
            status=RoleStatus.PRESENT,
            confidence=RoleConfidence.HIGH,
            evidence=[
                RoleEvidence(
                    status=RoleStatus.PRESENT,
                    confidence=RoleConfidence.HIGH,
                    source="allensdk",
                    path="boc.get_experiment_containers / get_ophys_experiments",
                    notes="Experiment / container / cell metadata available via AllenSDK.",
                ),
            ],
        ),
    )

    dataset = DatasetRef(
        target=ALLEN_TARGET,
        dataset_id=DatasetId(scene_set_id),
        dataset_version=ALLEN_DATASET_NAME,
        label=f"Allen Brain Observatory natural scenes — {scene_set_id}",
    )
    manifest = RoleManifest.new(
        dataset=dataset,
        roles=roles,
        parser_version="0.1.0",
        notes=notes,
    )

    manifest_path_out = manifests_root() / f"{manifest.manifest_id}.yaml"
    manifest_path_out.write_text(manifest.to_yaml(), encoding="utf-8")
    logger.info("Wrote manifest YAML to %s", manifest_path_out)
    return manifest


def load_image_catalog(scene_set_id: str) -> dict:
    """Load the on-disk catalog JSON written by `build_natural_scenes_manifest`."""
    from mousehash.artifacts.io import load_json

    return load_json(_image_catalog_path(scene_set_id))
