"""Print a human-readable pipeline status for a given scene set.

Usage:
    python scripts/inspect_schema.py
    python scripts/inspect_schema.py --scene-set-id allen_natural_scenes_v1
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-set-id", default="allen_natural_scenes_v1")
    parser.add_argument("--spec-id", default="vit_b16_imagenet_cpu")
    parser.add_argument("--rule-id", default="imagenet_top1_leq_397")
    args = parser.parse_args()

    from mousehash.schema.queries import format_status, pipeline_status

    status = pipeline_status(
        scene_set_id=args.scene_set_id,
        representation_spec_id=args.spec_id,
        rule_id=args.rule_id,
    )
    print(format_status(status))


if __name__ == "__main__":
    main()
