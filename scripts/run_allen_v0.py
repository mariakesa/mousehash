#!/usr/bin/env python
"""Run the Allen natural-scenes v0 structure-discovery pipeline end-to-end.

Reads paths from environment (MOUSEHASH_DATA_ROOT, ALLEN_MANIFEST_PATH).

Example:
    .venv/bin/python scripts/run_allen_v0.py \\
        --scene-set-id allen_natural_scenes_v1 \\
        --pca-components 10 --nmf-components 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mousehash.pipelines import run_allen_natural_scenes_v0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-set-id", default="allen_natural_scenes_v1")
    p.add_argument("--allen-manifest", type=Path, default=None,
                   help="Path to AllenSDK BrainObservatoryCache manifest JSON. Defaults to $ALLEN_MANIFEST_PATH.")
    p.add_argument("--representation-spec-id", default="vit_base_imagenet_v0")
    p.add_argument("--vit-model", default="google/vit-base-patch16-224")
    p.add_argument("--vit-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--vit-batch-size", type=int, default=16)
    p.add_argument("--pca-components", type=int, default=10)
    p.add_argument("--nmf-components", type=int, default=10)
    p.add_argument("--nmf-temperature", type=float, default=1.0)
    p.add_argument("--animate-threshold", type=int, default=397)
    p.add_argument("--report-output-dir", type=Path, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    result = run_allen_natural_scenes_v0(
        scene_set_id=args.scene_set_id,
        allen_manifest_path=args.allen_manifest,
        representation_spec_id=args.representation_spec_id,
        pca_n_components=args.pca_components,
        nmf_n_components=args.nmf_components,
        nmf_temperature=args.nmf_temperature,
        vit_model_name=args.vit_model,
        vit_batch_size=args.vit_batch_size,
        vit_device=args.vit_device,
        animate_threshold=args.animate_threshold,
        report_output_dir=args.report_output_dir,
    )
    print(json.dumps(
        {
            "manifest_id": result["manifest_id"],
            "view_id": result["view_id"],
            "report_index": result["report"]["reports"]["index"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
