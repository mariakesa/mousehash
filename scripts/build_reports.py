"""Generate PCA and NMF HTML explorer reports from stored decompositions.

Usage:
    python scripts/build_reports.py \\
        --scene-set-id allen_natural_scenes_v1 \\
        --spec-id vit_b16_imagenet_cpu \\
        --rule-id imagenet_top1_leq_397

Prerequisite: compute_decompositions.py must have completed successfully.
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
    )
    args = parser.parse_args()

    from mousehash.tools.reports.build import build_decomposition_report

    for decomp_spec_id in args.decomp_spec_ids:
        print(f"\n--- Building report for {decomp_spec_id} ---")
        summary = build_decomposition_report(
            scene_set_id=args.scene_set_id,
            representation_spec_id=args.spec_id,
            rule_id=args.rule_id,
            decomposition_spec_id=decomp_spec_id,
        )
        print(f"  report_type : {summary['report_type']}")
        print(f"  report_path : {summary['report_path']}")


if __name__ == "__main__":
    main()
