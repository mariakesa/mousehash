from __future__ import annotations

from smolagents import LiteLLMModel, ToolCallingAgent

from mousehash.agents.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are the MouseHash coordinator agent.

Your job is to advance the Allen natural scenes analysis pipeline by inspecting
DataJoint schema state and triggering the appropriate bounded computation tools.

Pipeline order:
  1. ingest        → run_ingestion
  2. representations → run_representations
  3. decompositions  → run_decompositions
  4. reports         → run_reports

Rules you must follow:
- Always call check_pipeline_status first to see what is already done.
- Never skip a stage: ingestion must exist before representations, representations
  before decompositions, decompositions before reports.
- Never re-run a stage that is already marked complete.
- Do not invent paths, scene_set_ids, or spec_ids that were not given to you.
- After completing a stage, re-check status before deciding what to do next.
- When the pipeline is fully complete, report the paths to the HTML reports.
"""


def make_coordinator(model_id: str = "claude-sonnet-4-6") -> ToolCallingAgent:
    """Return a ToolCallingAgent wired up with all MouseHash pipeline tools.

    Args:
        model_id: Any LiteLLM-compatible model string, e.g.
                  ``"claude-sonnet-4-6"`` or ``"gpt-4o"``.
                  Requires the corresponding API key in the environment.
    """
    model = LiteLLMModel(model_id=model_id)
    return ToolCallingAgent(
        tools=ALL_TOOLS,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        max_steps=12,
    )
