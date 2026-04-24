"""Launch the MouseHash coordinator agent to advance the analysis pipeline.

The coordinator inspects DataJoint schema state and calls the appropriate
pipeline tools (ingestion → representations → decompositions → reports) until
the pipeline is complete for the given scene set.

Usage:
    python scripts/run_coordinator.py \\
        --scene-set-id allen_natural_scenes_v1 \\
        --manifest /path/to/boc_manifest.json

    # Use a different model (any LiteLLM-compatible string):
    python scripts/run_coordinator.py \\
        --scene-set-id allen_natural_scenes_v1 \\
        --manifest /path/to/boc_manifest.json \\
        --model gpt-4o

Requires ANTHROPIC_API_KEY (default model) or the appropriate key for the
chosen model to be set in the environment.
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
        help="Scene set to process end-to-end",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to BrainObservatoryCache manifest (required if ingestion not yet done)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="LiteLLM model string (default: claude-sonnet-4-6)",
    )
    args = parser.parse_args()

    from mousehash.agents.coordinator import make_coordinator

    coordinator = make_coordinator(model_id=args.model)

    task = f"Run the full MouseHash pipeline for scene_set_id='{args.scene_set_id}'."
    if args.manifest:
        task += f" The AllenSDK manifest is at '{args.manifest}'."
    task += (
        " Check what is already done, then run only the missing stages in order. "
        "When everything is complete, report the paths to the HTML reports."
    )

    print(f"\nTask: {task}\n")
    result = coordinator.run(task)
    print("\n--- Coordinator result ---")
    print(result)


if __name__ == "__main__":
    main()
