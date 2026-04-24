"""Launch the MouseHash chat UI (Gradio).

Usage:
    python scripts/ui.py
    python scripts/ui.py --model gpt-4o
    python scripts/ui.py --share          # public Gradio link

Requires ANTHROPIC_API_KEY (default model) or the appropriate key for the
chosen model.  Install the agents extra first:
    pip install -e ".[agents]"
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="LiteLLM model string (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port to serve the UI on (default: 7860)",
    )
    args = parser.parse_args()

    from smolagents import GradioUI
    from mousehash.agents.coordinator import make_coordinator

    coordinator = make_coordinator(model_id=args.model)

    print(f"\nStarting MouseHash UI (model={args.model}) on http://localhost:{args.port}")
    print("Ask the coordinator to run the pipeline or query your analysis results.\n")

    GradioUI(coordinator).launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
