"""Tests for the on-demand DANDI fetcher.

Most tests stay offline — only the marked ``network`` test actually hits
the DANDI archive.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from mousehash.agents.dandi_agent.fetcher import (
    AssetChoice,
    _ephys_score,
    _extract_named_list,
    _is_units_only,
    _normalize_dandiset_id,
    select_representative_asset,
)


def test_normalize_accepts_common_formats() -> None:
    assert _normalize_dandiset_id("000011") == "000011"
    assert _normalize_dandiset_id("DANDI:000011") == "000011"
    assert _normalize_dandiset_id("DANDI:000011/draft") == "000011"
    assert _normalize_dandiset_id("11") == "000011"


def test_normalize_rejects_non_dandiset_strings() -> None:
    with pytest.raises(ValueError):
        _normalize_dandiset_id("not_a_dandiset")


def test_is_units_only_strict_set() -> None:
    assert _is_units_only(["Units", "SpikeEventSeries"])
    assert _is_units_only(["Units"])
    assert not _is_units_only(["Units", "BehavioralEvents"])
    assert not _is_units_only([])


def test_ephys_score_prefers_ecephys_paths() -> None:
    a = _ephys_score("sub-1/sub-1_ses-1_behavior+ecephys+ogen.nwb")
    b = _ephys_score("sub-1/sub-1_ses-1_behavior.nwb")
    c = _ephys_score("sub-1/sub-1_ses-1_image+movie.nwb")
    assert a > b > c


def test_extract_named_list_handles_dicts_and_objects() -> None:
    md = {
        "variableMeasured": [
            {"value": "Units"},
            {"value": "BehavioralEvents"},
        ],
        "approach": [SimpleNamespace(name="electrophysiological approach")],
    }
    assert _extract_named_list(md, "variableMeasured") == [
        "Units",
        "BehavioralEvents",
    ]
    assert _extract_named_list(md, "approach") == ["electrophysiological approach"]
    assert _extract_named_list(md, "missing_field") == []


# ---------------------------------------------------------------------------
# Selector with a synthetic dandiset
# ---------------------------------------------------------------------------

def _fake_asset(path: str, size: int, variables: list[str], asset_id: str):
    md = SimpleNamespace(
        variableMeasured=[SimpleNamespace(value=v) for v in variables]
    )
    asset = SimpleNamespace(
        path=path,
        size=size,
        identifier=asset_id,
        get_metadata=lambda: md,
    )
    return asset


def _fake_dandiset(assets: list, identifier: str = "000099", version: str = "draft"):
    return SimpleNamespace(
        identifier=identifier,
        version_id=version,
        get_assets=lambda: iter(assets),
    )


def test_selector_prefers_ecephys_over_behavior_only() -> None:
    behavior = _fake_asset("sub-1/sub-1_behavior.nwb", 20_000_000, ["BehavioralEvents"], "a1")
    ecephys = _fake_asset(
        "sub-1/sub-1_behavior+ecephys.nwb",
        50_000_000,
        ["Units", "BehavioralEvents"],
        "a2",
    )
    dandiset = _fake_dandiset([behavior, ecephys])
    chosen = select_representative_asset(dandiset)
    assert chosen is not None
    assert chosen.asset_id == "a2"
    assert "ecephys" in chosen.asset_path


def test_selector_skips_zero_byte_assets() -> None:
    empty = _fake_asset("sub-1/empty.nwb", 0, ["Units"], "a1")
    real = _fake_asset("sub-1/sub-1_ecephys.nwb", 10_000_000, ["Units"], "a2")
    dandiset = _fake_dandiset([empty, real])
    chosen = select_representative_asset(dandiset)
    assert chosen is not None
    assert chosen.asset_id == "a2"


def test_selector_respects_size_cap() -> None:
    huge = _fake_asset("sub-1/huge_ecephys.nwb", 10**10, ["Units"], "a1")
    dandiset = _fake_dandiset([huge])
    assert select_representative_asset(dandiset, max_size_bytes=100_000_000) is None


# ---------------------------------------------------------------------------
# Live network smoke (slow, optional)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.slow
def test_live_fetch_dandiset_000011_uses_cache(tmp_path, monkeypatch) -> None:
    """End-to-end with the real DANDI API. Slow; uses the existing cache."""
    from mousehash.agents.dandi_agent.fetcher import fetch_dandiset

    result = fetch_dandiset("000011", max_size_bytes=300 * 1024 * 1024)
    assert result.dandiset_id == "000011"
    assert result.nwb_path.exists()
    assert "ecephys" in result.asset.asset_path.lower()
    # Re-fetching should be a cache hit.
    again = fetch_dandiset("000011", max_size_bytes=300 * 1024 * 1024)
    assert again.cached is True
