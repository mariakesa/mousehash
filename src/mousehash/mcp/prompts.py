"""Canned MCP prompts for common MouseHash scientific interactions.

Prompts are reusable conversation starters. They're templated strings that
embed argument values; the agent calls them with placeholders filled in.

Keeping these short and instructive — they should nudge the agent toward the
correct tool sequence, not replace its reasoning.
"""

from __future__ import annotations


def explain_dataset_readiness(manifest_id: str) -> str:
    """Prompt the agent to summarize what MouseHash can run on a dataset."""
    return (
        f"For the MouseHash role manifest `{manifest_id}`:\n"
        f"  1. Call `get_manifest(\"{manifest_id}\")` to see which roles are present and at what confidence.\n"
        f"  2. Call `list_runnable_tools(\"{manifest_id}\")` to enumerate every registered tool's readiness.\n"
        f"  3. For each tool that is NOT runnable, call `explain_tool_readiness(tool_name, \"{manifest_id}\")` "
        f"to identify the missing required roles or unsatisfied `any_of` groups.\n"
        f"  4. Report: which analyses can run today, which roles are missing for the rest, and what would have "
        f"to change in the dataset (or what transformations would have to be added) to unlock them.\n"
        f"\n"
        f"Do not invent tool names or roles. Only reference what the readiness reports return."
    )


def design_analysis_plan(scientific_goal: str) -> str:
    """Prompt the agent to map a research question onto MouseHash tools."""
    return (
        f"Scientific goal: {scientific_goal}\n"
        f"\n"
        f"Design a MouseHash analysis plan in five steps:\n"
        f"  1. Identify which roles (conditions, stimuli, behavior, neural_data, time_organization, metadata) "
        f"are required to answer this question.\n"
        f"  2. List candidate target datasets (call `allen_list_datasets` for v0; DANDI / IBL come in later phases).\n"
        f"  3. For each candidate, call `allen_build_manifest(scene_set_id)` then `list_runnable_tools(manifest_id)`. "
        f"Pick the dataset whose readiness covers the most of the required roles.\n"
        f"  4. Propose an AnalysisMove: required roles, transformation plan, tool, validation controls, expected artifacts. "
        f"Reference the architecture doc's `AnalysisMove` shape (§14-15).\n"
        f"  5. Note risks: leakage paths, control comparisons needed, where the result would be ambiguous.\n"
        f"\n"
        f"Be specific. Reference tools by their registered names, not invented ones."
    )
