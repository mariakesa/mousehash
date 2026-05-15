"""End-to-end test: NWB -> manifest -> readiness -> AnalysisMove."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mousehash.agents.dandi_agent import (
    parse_mousehash_roles,
    suggest_analysis_moves,
)
from mousehash.agents.dandi_agent.catalogs.loaders import (
    load_tools,
    load_transformations,
)

GOLDEN_NWB = Path(
    "/home/maria/mousehash/src/DANDI-agent/dandi_nwb_cache/light_data/000011/"
    "0.220126.1907/sub-291064/sub-291064_ses-20150907_behavior+ecephys+ogen.nwb"
)


@pytest.mark.skipif(not GOLDEN_NWB.exists(), reason="DANDI cache fixture missing")
def test_full_pipeline_produces_pca_and_psth_moves() -> None:
    """The canonical demo: dandiset 000011 -> ranked moves with PSTH + Trial-Averaged PCA."""
    manifest = parse_mousehash_roles(nwb_path=str(GOLDEN_NWB), dandiset_id="000011")

    # Parser produces all six top-level roles for this dataset.
    for role in ("neural_data", "stimuli", "behavior", "conditions", "time_organization", "metadata"):
        assert manifest.status(role) in (
            "present",
            "likely_present",
            "derived_possible",
        ), f"role {role} should not be missing"

    pairs = suggest_analysis_moves(manifest, top_k=10)
    move_names = [m.tool_name for _, m in pairs]
    assert "Generate PSTH Plot" in move_names
    assert "Run Trial-Averaged PCA" in move_names
    assert "Generate Raster Plot" in move_names

    # Trial-Averaged PCA should be ready and its plan should reference
    # specific transformations that exist in the catalog.
    tx_catalog = load_transformations()
    psth_move = next(m for _, m in pairs if m.tool_name == "Run Trial-Averaged PCA")
    assert psth_move.transformation_plan
    assert all(step in tx_catalog for step in psth_move.transformation_plan)


@pytest.mark.skipif(not GOLDEN_NWB.exists(), reason="DANDI cache fixture missing")
def test_manifest_persistence_roundtrip(tmp_path) -> None:
    """Manifest JSON is faithful: writing + reading restores statuses."""
    manifest = parse_mousehash_roles(nwb_path=str(GOLDEN_NWB), dandiset_id="000011")
    out = tmp_path / "manifest.json"
    out.write_text(manifest.model_dump_json(indent=2))
    raw = json.loads(out.read_text())
    assert raw["dandiset_id"] == "000011"
    # Spot-check that one core role survived the round-trip with status intact.
    spikes = raw["roles"]["neural_data.spikes"]
    assert spikes["status"] == "present"


def test_catalogs_load_and_every_tool_path_maps_to_real_transforms() -> None:
    """No tool can reference a transformation that isn't catalogued."""
    tools = load_tools()
    tx = load_transformations()
    missing: list[tuple[str, str]] = []
    for tool in tools.values():
        for step in tool.default_transformation_path:
            if step not in tx:
                missing.append((tool.tool_id, step))
    assert not missing, f"unknown transformation steps: {missing}"


def test_all_mvp14_tools_present() -> None:
    expected = {
        "generate_raster_plot",
        "generate_psth_plot",
        "run_pca",
        "run_trial_averaged_pca",
        "fit_logistic_decoder",
        "cv_decoding_eval",
        "fit_ridge_encoding",
        "cv_encoding_eval",
        "compute_rsm",
        "run_rsa",
        "compute_noise_correlations",
        "run_permutation_test",
        "generate_latent_trajectory_plot",
        "generate_structure_discovery_report",
    }
    assert set(load_tools()) == expected
