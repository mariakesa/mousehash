"""Ingest Allen natural scenes into DataJoint-backed stimulus artifacts.

Usage:
    python scripts/ingest_natural_scenes.py \\
        --manifest /path/to/boc_manifest.json \\
        --scene-set-id allen_natural_scenes_v1

The BrainObservatoryCache manifest is created automatically by AllenSDK the
first time it is used; point --manifest at wherever you want it to live.
AllenSDK will download session data into the same directory on first run.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to AllenSDK BrainObservatoryCache manifest JSON",
    )
    parser.add_argument(
        "--scene-set-id",
        default="allen_natural_scenes_v1",
        help="Primary key for this scene set in DataJoint (default: allen_natural_scenes_v1)",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-text notes stored with the scene-set record",
    )
    args = parser.parse_args()

    from mousehash.tools.allen.natural_scenes import ingest_natural_scenes

    summary = ingest_natural_scenes(
        manifest_path=args.manifest,
        scene_set_id=args.scene_set_id,
        notes=args.notes,
    )

    print("\nIngestion complete:")
    print(f"  scene_set_id : {summary['scene_set_id']}")
    print(f"  n_images     : {summary['n_images']}")
    print(f"  image_dir    : {summary['image_dir']}")


if __name__ == "__main__":
    main()
