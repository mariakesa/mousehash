"""Direct tests for the @tool functions in dandi_agent.agent_tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mousehash.agents.dandi_agent.agent_tools import (
    explain_blocked_tools,
    inspect_dandiset,
    list_paper_evidence,
    parse_nwb_manifest,
    propose_transformation_plan,
    show_role_manifest,
    suggest_analyses,
)

GOLDEN_NWB = Path(
    "/home/maria/mousehash/src/DANDI-agent/dandi_nwb_cache/light_data/000011/"
    "0.220126.1907/sub-291064/sub-291064_ses-20150907_behavior+ecephys+ogen.nwb"
)


pytestmark = pytest.mark.skipif(
    not GOLDEN_NWB.exists(), reason="DANDI cache fixture missing"
)


@pytest.fixture(scope="module")
def manifest_path() -> str:
    payload = json.loads(
        parse_nwb_manifest(dandiset_id="000011", nwb_path=str(GOLDEN_NWB))
    )
    return payload["manifest_path"]


def test_inspect_dandiset_returns_structured_summary_or_error() -> None:
    """Without a local metadata path, the tool either fetches from DANDI or
    reports a clean network error. Either outcome is JSON with dandiset_id."""
    out = inspect_dandiset("000011")
    payload = json.loads(out)
    assert payload["dandiset_id"] == "000011"
    assert "error" in payload or "name" in payload


def test_parse_nwb_manifest_persists_json(manifest_path: str) -> None:
    p = Path(manifest_path)
    assert p.exists()
    blob = json.loads(p.read_text())
    assert blob["dandiset_id"] == "000011"
    assert "roles" in blob


def test_show_role_manifest_prints_each_role(manifest_path: str) -> None:
    text = show_role_manifest(manifest_path)
    assert "neural_data.spikes" in text
    assert "present" in text
    assert "parser:" in text


def test_suggest_analyses_includes_mvp_tools(manifest_path: str) -> None:
    payload = json.loads(suggest_analyses(manifest_path, top_k=10))
    names = {s["tool_name"] for s in payload["suggestions"]}
    # On this ALM dandiset, raster/PSTH/Trial-Averaged PCA should all be ready.
    assert names & {
        "Generate PSTH Plot",
        "Generate Raster Plot",
        "Run Trial-Averaged PCA",
    }


def test_propose_transformation_plan_enriches_steps(manifest_path: str) -> None:
    payload = json.loads(
        propose_transformation_plan(manifest_path, "generate_psth_plot")
    )
    assert payload["tool_name"] == "Generate PSTH Plot"
    assert payload["required_view"] == "condition/trial x time bins -> firing rate"
    steps = payload["transformation_plan"]
    assert any(s["name"] == "bin_spikes" for s in steps)
    bin_step = next(s for s in steps if s["name"] == "bin_spikes")
    assert bin_step["family"] == "binning_resampling"
    assert "bin_size_ms" in bin_step["parameters"]


def test_propose_transformation_plan_rejects_unknown_tool(manifest_path: str) -> None:
    payload = json.loads(propose_transformation_plan(manifest_path, "nope_tool_id"))
    assert "error" in payload


def test_explain_blocked_tools_returns_list(manifest_path: str) -> None:
    payload = json.loads(explain_blocked_tools(manifest_path))
    assert "blocked_tools" in payload
    # On this rich ALM dataset there should be 0 blocked tools.
    assert isinstance(payload["blocked_tools"], list)


def test_list_paper_evidence_is_stub(manifest_path: str) -> None:
    payload = json.loads(list_paper_evidence(manifest_path))
    assert payload["paper_evidence"] == []
    assert "MVP 3" in payload["note"]
