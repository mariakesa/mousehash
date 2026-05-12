"""Launch the MouseHash DANDI chat agent.

Two front-ends:

1. ``--ui gradio`` — browser chat window via smolagents.GradioUI.
2. ``--ui terminal`` — simple REPL loop in the terminal (default).

Examples:
    # Terminal REPL
    python scripts/run_dandi_agent.py

    # Gradio chat in browser
    python scripts/run_dandi_agent.py --ui gradio

    # One-shot, non-interactive (good for screenshots / CI smoke tests)
    python scripts/run_dandi_agent.py --task "What can I analyze in 000011?"

Requires ANTHROPIC_API_KEY (default model: claude-sonnet-4-6) or the
appropriate key for the chosen ``--model``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` from the repo root before anything that reads env vars
# (`ANTHROPIC_API_KEY`, `MOUSEHASH_DATA_ROOT`, DJ_* etc.) — mirrors the
# tests/conftest.py bootstrap so this script works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


PREAMBLE = """\
MouseHash DANDI chat agent.

What it can do:
  * inspect_dandiset(dandiset_id, metadata_path)
  * parse_nwb_manifest(dandiset_id, nwb_path)  -> EvidenceBackedRoleManifest
  * show_role_manifest(manifest_path)
  * suggest_analyses(manifest_path, top_k=10)  -> ranked AnalysisMoves
  * propose_transformation_plan(manifest_path, tool_id)
  * explain_blocked_tools(manifest_path)

Try asking, for example:

  > I have an NWB at /home/maria/mousehash/src/DANDI-agent/dandi_nwb_cache/
    light_data/000011/0.220126.1907/sub-291064/sub-291064_ses-20150907_
    behavior+ecephys+ogen.nwb from dandiset 000011. What can I analyze, and
    which analysis should I run first?

Type "exit" or Ctrl-D to quit (terminal mode).
"""


def _check_api_key(model: str) -> None:
    if "claude" in model and not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "WARNING: ANTHROPIC_API_KEY is not set. The agent needs it to call "
            "Claude. Export it before chatting, e.g. "
            "`export ANTHROPIC_API_KEY=sk-ant-...`.",
            file=sys.stderr,
        )


def run_terminal(agent) -> int:
    print(PREAMBLE)
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.lower() in {"exit", "quit", ":q"}:
            return 0
        if not line:
            continue
        try:
            answer = agent.run(line)
        except Exception as exc:  # noqa: BLE001 — top-level UI handler
            print(f"agent error: {exc}")
            continue
        print(f"\nagent> {answer}")


def run_gradio(agent, share: bool = False) -> int:
    from smolagents import GradioUI

    GradioUI(
        agent,
        file_upload_folder=None,
    ).launch(
        share=share,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ui",
        choices=["terminal", "gradio"],
        default="terminal",
        help="Front-end to use (default: terminal).",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="LiteLLM model string (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="If supplied, run this single task and exit instead of starting a UI.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="(gradio only) create a public gradio.live URL.",
    )
    args = parser.parse_args()

    _check_api_key(args.model)

    from mousehash.agents.dandi_agent import make_dandi_coordinator

    agent = make_dandi_coordinator(model_id=args.model)

    if args.task:
        print(f"\nTask: {args.task}\n")
        result = agent.run(args.task)
        print("\n--- agent result ---")
        print(result)
        return 0

    if args.ui == "gradio":
        return run_gradio(agent, share=args.share)
    return run_terminal(agent)


if __name__ == "__main__":
    raise SystemExit(main())
