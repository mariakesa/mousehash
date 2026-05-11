"""SmoLAgents @tool wrappers for the DANDI agent."""

from __future__ import annotations

from smolagents import tool

from mousehash.agents.dandi_agent.agent_tools import (
    explain_blocked_tools as _explain_blocked_tools,
    inspect_dandiset as _inspect_dandiset,
    list_paper_evidence as _list_paper_evidence,
    parse_nwb_manifest as _parse_nwb_manifest,
    propose_transformation_plan as _propose_transformation_plan,
    show_role_manifest as _show_role_manifest,
    suggest_analyses as _suggest_analyses,
)

inspect_dandiset = tool(_inspect_dandiset)
parse_nwb_manifest = tool(_parse_nwb_manifest)
show_role_manifest = tool(_show_role_manifest)
suggest_analyses = tool(_suggest_analyses)
propose_transformation_plan = tool(_propose_transformation_plan)
explain_blocked_tools = tool(_explain_blocked_tools)
list_paper_evidence = tool(_list_paper_evidence)

DANDI_AGENT_TOOLS = [
    inspect_dandiset,
    parse_nwb_manifest,
    show_role_manifest,
    suggest_analyses,
    propose_transformation_plan,
    explain_blocked_tools,
    list_paper_evidence,
]

__all__ = [
    "DANDI_AGENT_TOOLS",
    "inspect_dandiset",
    "parse_nwb_manifest",
    "show_role_manifest",
    "suggest_analyses",
    "propose_transformation_plan",
    "explain_blocked_tools",
    "list_paper_evidence",
]
