from __future__ import annotations

from smolagents import tool

from mousehash.agents.agent_tools import (
    check_pipeline_status as _check_pipeline_status,
    get_analysis_summary as _get_analysis_summary,
    run_decompositions as _run_decompositions,
    run_ingestion as _run_ingestion,
    run_nmf_at_temperature as _run_nmf_at_temperature,
    run_reports as _run_reports,
    run_representations as _run_representations,
)

check_pipeline_status = tool(_check_pipeline_status)
get_analysis_summary = tool(_get_analysis_summary)
run_ingestion = tool(_run_ingestion)
run_representations = tool(_run_representations)
run_decompositions = tool(_run_decompositions)
run_nmf_at_temperature = tool(_run_nmf_at_temperature)
run_reports = tool(_run_reports)

ALL_TOOLS = [
    check_pipeline_status,
    get_analysis_summary,
    run_ingestion,
    run_representations,
    run_decompositions,
    run_nmf_at_temperature,
    run_reports,
]

__all__ = [
    "ALL_TOOLS",
    "check_pipeline_status",
    "get_analysis_summary",
    "run_ingestion",
    "run_representations",
    "run_decompositions",
    "run_nmf_at_temperature",
    "run_reports",
]
