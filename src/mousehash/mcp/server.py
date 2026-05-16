"""FastMCP server instance + registration of every tool / resource / prompt.

This module is a **pure registration shim** — no tool logic lives here. Every
callable comes from `mcp/<domain>_tools.py` (tools), `mcp/resources.py`
(resources), or `mcp/prompts.py` (prompts).

Launch: `python -m mousehash.mcp` (see `mcp/__main__.py`).
"""

from __future__ import annotations

from fastmcp import FastMCP

from mousehash.mcp.analysis_tools import (
    compare_jpeg_by_animate_inanimate,
    run_nmf,
    run_pca,
)
from mousehash.mcp.manifest_tools import (
    explain_tool_readiness,
    get_manifest,
    list_runnable_tools,
)
from mousehash.mcp.pipeline_tools import run_allen_natural_scenes_v0
from mousehash.mcp.prompts import design_analysis_plan, explain_dataset_readiness
from mousehash.mcp.report_tools import generate_structure_report
from mousehash.mcp.resources import (
    allen_datasets_resource,
    manifest_resource,
    targets_resource,
    tool_contract_resource,
)
from mousehash.mcp.schedule_tools import analyze_stimulus_schedule, extract_stimulus_schedule
from mousehash.mcp.target_tools import allen_build_manifest, allen_list_datasets
from mousehash.mcp.transformation_tools import (
    extract_event_responses,
    extract_jpeg_sizes,
    extract_vit_features,
)
from mousehash.mcp.view_tools import inspect_view, list_views

mcp = FastMCP("mousehash")


# ---------- Tools ----------
# Tool name = function name; namespace comes from the server name "mousehash".
# Targets + manifests + readiness
mcp.tool()(allen_list_datasets)
mcp.tool()(allen_build_manifest)
mcp.tool()(get_manifest)
mcp.tool()(list_runnable_tools)
mcp.tool()(explain_tool_readiness)
# Views
mcp.tool()(list_views)
mcp.tool()(inspect_view)
# Transformations
mcp.tool()(extract_vit_features)
mcp.tool()(extract_jpeg_sizes)
mcp.tool()(extract_stimulus_schedule)
mcp.tool()(extract_event_responses)
# Analysis
mcp.tool()(run_pca)
mcp.tool()(run_nmf)
mcp.tool()(compare_jpeg_by_animate_inanimate)
mcp.tool()(analyze_stimulus_schedule)
# Reports
mcp.tool()(generate_structure_report)
# All-in-one pipeline
mcp.tool()(run_allen_natural_scenes_v0)


# ---------- Resources ----------

mcp.resource("mousehash://targets")(targets_resource)
mcp.resource("mousehash://targets/allen/datasets")(allen_datasets_resource)
mcp.resource("mousehash://manifests/{manifest_id}")(manifest_resource)
mcp.resource("mousehash://tools/{tool_name}/contract")(tool_contract_resource)


# ---------- Prompts ----------

mcp.prompt()(explain_dataset_readiness)
mcp.prompt()(design_analysis_plan)


__all__ = ["mcp"]
