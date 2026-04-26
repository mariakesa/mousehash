"""Ingest a folder of image files into MouseHash's generic stimulus schema.

Usage:
    python scripts/ingest_image_folder.py \
        --image-dir /path/to/images \
        --scene-set-id macaque_itbench_v1 \
        --dataset-name macaque_it_bench
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
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--scene-set-id", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--stimulus-name", default="image_folder")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    from mousehash.tools.datasets.image_folder import ingest_image_folder

    summary = ingest_image_folder(
        image_dir=args.image_dir,
        scene_set_id=args.scene_set_id,
        dataset_name=args.dataset_name,
        stimulus_name=args.stimulus_name,
        notes=args.notes,
    )

    print("\nImage-folder ingestion complete:")
    print(f"  scene_set_id : {summary['scene_set_id']}")
    print(f"  n_images     : {summary['n_images']}")
    print(f"  image_dir    : {summary['image_dir']}")


if __name__ == "__main__":
    main()