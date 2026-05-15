"""Plot one Allen cell's dF/F trace against animate/inanimate natural scenes.

Usage:
    python scripts/plot_cell_dff_natural_scenes.py \
        --manifest /path/to/brain_observatory_manifest.json \
        --cell-specimen-id 517510587
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
        help="Path to the AllenSDK BrainObservatoryCache manifest JSON",
    )
    parser.add_argument(
        "--cell-specimen-id",
        required=True,
        type=int,
        help="Allen cell_specimen_id to fetch and plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output HTML path (default: reports/cell_activity/...)",
    )
    parser.add_argument(
        "--model-name",
        default="google/vit-base-patch16-224",
        help="HuggingFace image-classification model used for scene labels",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Retained for compatibility with the ViT runner (default: 16)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for ViT inference, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--threshold-max-class-idx",
        type=int,
        default=397,
        help="Top-1 ImageNet class threshold for animate vs inanimate",
    )
    args = parser.parse_args()

    from mousehash.tools.allen.cell_activity import (
        analyze_cell_dff_against_animate_inanimate,
    )

    summary = analyze_cell_dff_against_animate_inanimate(
        manifest_path=args.manifest,
        cell_specimen_id=args.cell_specimen_id,
        threshold_max_class_idx=args.threshold_max_class_idx,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )

    print("\nCell dF/F plot complete:")
    print(f"  cell_specimen_id      : {summary['cell_specimen_id']}")
    print(f"  experiment_id         : {summary['experiment_id']}")
    print(f"  experiment_container  : {summary['experiment_container_id']}")
    print(f"  n_timepoints          : {summary['n_timepoints']}")
    print(f"  animate timepoints    : {summary['n_animate_timepoints']}")
    print(f"  inanimate timepoints  : {summary['n_inanimate_timepoints']}")
    print(f"  outside/blank points  : {summary['n_other_timepoints']}")
    print(f"  plot_path             : {summary['plot_path']}")


if __name__ == "__main__":
    main()