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


def design_decoder_fit(neural_view_id: str = "", labels_view_id: str = "") -> str:
    """Prompt the agent to walk the BlahML decoder question loop with the user."""
    seed = {}
    if neural_view_id:
        seed["neural_view_id"] = neural_view_id
    if labels_view_id:
        seed["labels_view_id"] = labels_view_id
    seed_json = "{}" if not seed else "{" + ", ".join(f'\"{k}\": \"{v}\"' for k, v in seed.items()) + "}"
    return (
        f"Drive the logistic-decoder fit configuration with the user using BlahML.\n"
        f"\n"
        f"  1. Call `decoder_next_question(answers_json='{seed_json}')` to get the first pending question.\n"
        f"  2. For each returned `question`:\n"
        f"       - Read `question.ask` and `question.explain` aloud to the user.\n"
        f"       - If `question.choices` exists, present them; if `question.default` exists, mention it as a safe fallback.\n"
        f"       - Wait for the user's answer. Coerce it into the right type from `question.type`.\n"
        f"       - Add the answer under `question.name` in your running answers dict; JSON-encode and pass back as the new `answers_json`.\n"
        f"  3. Loop until `decoder_next_question` returns `resolved: true`. Then call:\n"
        f"       `decode_animate_inanimate(neural_view_id=..., labels_view_id=..., split_strategy=..., k_folds=..., ...)`\n"
        f"     passing every answered key. Omit keys the user did not answer — the tool will use its default.\n"
        f"  4. Report `summary.cv_balanced_accuracy` vs `summary.chance_accuracy`, plus `summary.p_value` if a permutation null was requested. Mention the artifact dir so the user can inspect coefficients.\n"
        f"\n"
        f"Do not invent parameter names. Only use the keys returned by `decoder_next_question`."
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
