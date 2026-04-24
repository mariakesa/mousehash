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
- For pipeline questions, always call check_pipeline_status first.
- Never re-run a stage already marked complete.
- For data questions, call get_analysis_summary to read stored results.
- Do not invent paths, scene_set_ids, or spec_ids not given to you.
- When all pipeline stages are done, tell the user where the HTML reports are.
- Default scene_set_id is 'allen_natural_scenes_v1' unless told otherwise.
"""


def make_coordinator(model_id: str = "claude-sonnet-4-6") -> ToolCallingAgent:
    """Return a ToolCallingAgent wired up with all MouseHash pipeline tools.

    Args:
        model_id: Any LiteLLM-compatible model string, e.g.
                  ``"claude-sonnet-4-6"`` or ``"gpt-4o"``.
                  Requires the corresponding API key in the environment.
    """
    # Load the default toolcalling prompt templates and patch only system_prompt.
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
