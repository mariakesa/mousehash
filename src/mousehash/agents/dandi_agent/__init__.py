"""DANDI-aware MouseHash chat agent.

Pipeline: DANDI metadata + NWB structure -> EvidenceBackedRoleManifest ->
ToolReadinessReport -> ranked AnalysisMoves.
"""

from mousehash.agents.dandi_agent.agent import make_dandi_coordinator
from mousehash.agents.dandi_agent.parser import parse_mousehash_roles
from mousehash.agents.dandi_agent.readiness import (
    rank_analysis_suggestions,
    suggest_analysis_moves,
)
from mousehash.agents.dandi_agent.tools import DANDI_AGENT_TOOLS

__all__ = [
    "DANDI_AGENT_TOOLS",
    "make_dandi_coordinator",
    "parse_mousehash_roles",
    "rank_analysis_suggestions",
    "suggest_analysis_moves",
]
