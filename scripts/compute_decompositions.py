"""Run PCA and NMF decompositions on stored ViT representations.

Runs both decompositions seeded in DecompositionSpec by default.

Usage:
    python scripts/compute_decompositions.py \\
        --scene-set-id allen_natural_scenes_v1 \\
        --spec-id vit_b16_imagenet_cpu \\
        --rule-id imagenet_top1_leq_397

Prerequisite: compute_representations.py must have completed successfully.
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_DECOMP_SPECS = [
    "pca_logits_10_exploratory",
    "nmf_probs_10_exploratory",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene-set-id", default="allen_natural_scenes_v1")
    parser.add_argument("--spec-id", default="vit_b16_imagenet_cpu")
    parser.add_argument("--rule-id", default="imagenet_top1_leq_397")
    parser.add_argument(
        "--decomp-spec-ids",
        nargs="+",
        default=DEFAULT_DECOMP_SPECS,
        metavar="DECOMP_SPEC_ID",
        help="One or more DecompositionSpec IDs to run (default: both PCA and NMF)",
    )
    args = parser.parse_args()

    from mousehash.tools.decompositions.compute import compute_stimulus_decomposition

    for decomp_spec_id in args.decomp_spec_ids:
        print(f"\n--- {decomp_spec_id} ---")
        summary = compute_stimulus_decomposition(
            scene_set_id=args.scene_set_id,
            representation_spec_id=args.spec_id,
            rule_id=args.rule_id,
            decomposition_spec_id=decomp_spec_id,
        )
        print(f"  method       : {summary['method']}")
        print(f"  input_kind   : {summary['input_kind']}")
        print(f"  n_components : {summary['n_components']}")
        print(f"  n_images     : {summary['n_images']}")
        if "explained_variance_ratio_total" in summary:
            print(f"  variance explained : {summary['explained_variance_ratio_total']:.1%}")
        if "reconstruction_err" in summary:
            print(f"  reconstruction err : {summary['reconstruction_err']:.4f}")


if __name__ == "__main__":
    main()
