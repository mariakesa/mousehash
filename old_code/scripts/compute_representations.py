"""Compute ViT ImageNet representations for an ingested Allen natural scene set.

Usage:
    python scripts/compute_representations.py \\
        --scene-set-id allen_natural_scenes_v1 \\
        --spec-id vit_b16_imagenet_cpu \\
        --rule-id imagenet_top1_leq_397

Prerequisite: run setup_schema.py, seed_lookup_tables.py, and
ingest_natural_scenes.py first.
"""
from __future__ import annotations

import argparse
import logging

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
        "--scene-set-id",
        default="allen_natural_scenes_v1",
        help="Primary key of the ingested AllenNaturalSceneSet",
    )
    parser.add_argument(
        "--spec-id",
        default="vit_b16_imagenet_cpu",
        help="RepresentationSpec primary key (default: vit_b16_imagenet_cpu)",
    )
    parser.add_argument(
        "--rule-id",
        default="imagenet_top1_leq_397",
        help="AnimateInanimateRule primary key (default: imagenet_top1_leq_397)",
    )
    args = parser.parse_args()

    from mousehash.tools.representations.vit_imagenet import (
        compute_stimulus_representation,
    )

    summary = compute_stimulus_representation(
        scene_set_id=args.scene_set_id,
        representation_spec_id=args.spec_id,
        rule_id=args.rule_id,
    )

    print("\nRepresentation computation complete:")
    print(f"  scene_set_id           : {summary['scene_set_id']}")
    print(f"  representation_spec_id : {summary['representation_spec_id']}")
    print(f"  n_images               : {summary['n_images']}")
    print(f"  n_animate              : {summary['n_animate']}")
    print(f"  n_inanimate            : {summary['n_inanimate']}")


if __name__ == "__main__":
    main()
