from __future__ import annotations

import importlib.resources

import yaml
from smolagents import LiteLLMModel, ToolCallingAgent

from mousehash.agents.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are the MouseHash coordinator agent for Allen Brain Observatory \
natural scenes analysis.

You help users in two ways:
1. **Pipeline management** — advance the analysis pipeline by checking DataJoint
   schema state and running missing stages in order.
2. **Data analysis Q&A** — answer questions about the results using stored
   summary files (animate/inanimate counts, PCA variance, NMF error, top-1
   class distributions, report locations).

Pipeline order (never skip stages):
  1. ingest           → run_ingestion        (needs manifest path)
  2. representations  → run_representations
  3. decompositions   → run_decompositions
  4. reports          → run_reports

Rules:
- Treat a bare manifest path or a message that only provides a manifest path as context, not as a request to run or summarize the pipeline.
- Only use pipeline-management tools when the user explicitly asks about pipeline status, missing stages, ingestion, representations, decompositions, reports, or to run the pipeline.
- For cell plotting or cell activity questions, do not call pipeline-management tools first. Use the dedicated cell plot tools directly.
- If the user asks for a generic dF/F plot or cell activity plot and does not mention animate or inanimate labeling, use the vanilla cell plot tool.
- Only use the animate/inanimate cell plot tool when the user explicitly asks for animate/inanimate coloring, labels, or overlays.
- If the user requests specific colors, pass them through to the plotting tool instead of using defaults.
- For pipeline questions, always call check_pipeline_status first.
- Never re-run a stage already marked complete.
- For data questions, call get_analysis_summary to read stored results.
- For cell plotting questions, use the dedicated cell plot tools.
- Do not invent paths, scene_set_ids, or spec_ids not given to you.
- When all pipeline stages are done, tell the user where the HTML reports are.
- Default scene_set_id is 'allen_natural_scenes_v1' unless told otherwise.
- If a manifest path is not provided for cell plotting, use the configured default.
"""


def make_coordinator(model_id: str = "claude-sonnet-4-6") -> ToolCallingAgent:
    """Return a ToolCallingAgent wired up with all MouseHash pipeline tools.

    Args:
        model_id: Any LiteLLM-compatible model string, e.g.
                  ``"claude-sonnet-4-6"`` or ``"gpt-4o"``.
                  Requires the corresponding API key in the environment.
    """
    prompt_templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts")
        .joinpath("toolcalling_agent.yaml")
        .read_text()
    )
    prompt_templates["system_prompt"] = SYSTEM_PROMPT

    model = LiteLLMModel(model_id=model_id)
    return ToolCallingAgent(
        tools=ALL_TOOLS,
        model=model,
        prompt_templates=prompt_templates,
        max_steps=12,
    )


__all__ = ["SYSTEM_PROMPT", "make_coordinator"]
