"""On-demand DANDI dataset fetching.

Given a dandiset id like ``"000011"``, this module:

1. Pulls the dandiset's raw metadata from the DANDI API and caches it as JSON.
2. Scans the dandiset's assets and picks one representative NWB
   (smallest .nwb under ``max_size_bytes``, preferring units-only assets).
3. Downloads that NWB into ``DATA_ROOT/dandi_agent/cache/<dandiset>/<version>/``
   unless it's already there.

All three steps are cache-aware: re-running with the same dandiset_id is a no-op.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mousehash.config import DATA_ROOT

CACHE_ROOT = DATA_ROOT / "dandi_agent" / "cache"
METADATA_ROOT = DATA_ROOT / "dandi_agent" / "metadata"

# Hard cap to keep an agent from accidentally pulling a multi-GB asset.
DEFAULT_MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
ABSOLUTE_MAX_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


_DANDISET_ID_RE = re.compile(r"^\d{6}$")


def _normalize_dandiset_id(raw: str) -> str:
    """Accept ``"000011"``, ``"DANDI:000011"``, ``"DANDI:000011/draft"``."""
    s = raw.strip()
    if s.upper().startswith("DANDI:"):
        s = s.split(":", 1)[1]
    s = s.split("/", 1)[0]  # drop "/draft" or "/0.x..." suffix
    s = s.zfill(6) if s.isdigit() else s
    if not _DANDISET_ID_RE.match(s):
        raise ValueError(f"Not a valid 6-digit DANDI id: {raw!r}")
    return s


def _safe_jsonable(x: Any) -> Any:
    """Project Pydantic / DANDI metadata objects into plain JSON types."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _safe_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_safe_jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        try:
            return _safe_jsonable(x.model_dump())
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return _safe_jsonable(x.dict())
        except Exception:
            pass
    return str(x)


def _flatten_asset_path(asset_path: str) -> str:
    return asset_path.replace("/", "__")


# ---------------------------------------------------------------------------
# Asset selection
# ---------------------------------------------------------------------------

@dataclass
class AssetChoice:
    dandiset_id: str
    version_id: str
    asset_id: str
    asset_path: str
    size_bytes: int | None
    variable_measured: list[str]
    units_only: bool
    asset_metadata: dict[str, Any]


def _extract_named_list(md: Any, field: str) -> list[str]:
    """Pull labels from a DANDI metadata field that's a list of typed objects.

    DANDI's ``variableMeasured`` uses ``PropertyValue.value``;
    ``measurementTechnique`` / ``approach`` use the ``.name`` of a typed
    concept. Try ``.value`` first, then ``.name``, then string fallback.
    """
    out: list[str] = []
    if md is None:
        return out
    raw = getattr(md, field, None)
    if raw is None and isinstance(md, dict):
        raw = md.get(field)
    if not raw:
        return out
    for item in raw:
        label: str | None = None
        if isinstance(item, dict):
            label = item.get("value") or item.get("name")
        else:
            for attr in ("value", "name"):
                v = getattr(item, attr, None)
                if v:
                    label = str(v)
                    break
        if label:
            out.append(str(label))
    return out


def _is_units_only(variables: Iterable[str]) -> bool:
    keep = {v for v in variables if v}
    if not keep:
        return False
    return keep <= {"Units", "SpikeEventSeries"}


def _build_asset_choice(dandiset: Any, asset: Any, asset_md: Any) -> AssetChoice:
    variables = _extract_named_list(asset_md, "variableMeasured")
    size = getattr(asset, "size", None) or getattr(asset_md, "contentSize", None)
    try:
        size = int(size) if size is not None else None
    except (TypeError, ValueError):
        size = None
    return AssetChoice(
        dandiset_id=str(getattr(dandiset, "identifier", "")),
        version_id=str(getattr(dandiset, "version_id", "")),
        asset_id=str(getattr(asset, "identifier", "")),
        asset_path=str(getattr(asset, "path", "")),
        size_bytes=size,
        variable_measured=variables,
        units_only=_is_units_only(variables),
        asset_metadata=_safe_jsonable(asset_md) if asset_md is not None else {},
    )


def _ephys_score(path: str) -> int:
    """Score an asset path by how likely it is to contain neural data."""
    p = path.lower()
    score = 0
    if "ecephys" in p or "ephys" in p:
        score += 100
    if "ogen" in p or "opto" in p:
        score += 20
    if "behavior" in p:
        score += 10
    if "image" in p or "movie" in p:
        score -= 30  # large imaging assets are slow and often non-spike
    return score


def select_representative_asset(
    dandiset: Any,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    max_assets_to_scan: int = 100,
    min_size_bytes: int = 1024,  # exclude truly empty assets
) -> AssetChoice | None:
    """Pick the most useful .nwb asset for role inference.

    Ranking (highest first):
      1. Units-only metadata + smallest size.
      2. Path mentions ``ecephys`` / ``ogen`` / ``behavior``.
      3. Smallest size under the cap.
    """
    candidates: list[AssetChoice] = []
    for i, asset in enumerate(dandiset.get_assets(), start=1):
        if i > max_assets_to_scan:
            break
        path = str(getattr(asset, "path", "") or "")
        if not path.endswith(".nwb"):
            continue
        try:
            md = asset.get_metadata()
        except Exception:
            md = None
        choice = _build_asset_choice(dandiset, asset, md)
        if choice.size_bytes is None or choice.size_bytes < min_size_bytes:
            continue
        if choice.size_bytes > max_size_bytes:
            continue
        candidates.append(choice)
    if not candidates:
        return None

    def _key(c: AssetChoice) -> tuple[int, int, int]:
        # Sort by (units_only desc, ephys_score desc, size asc).
        return (
            -int(c.units_only),
            -_ephys_score(c.asset_path),
            c.size_bytes or 0,
        )

    candidates.sort(key=_key)
    return candidates[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    dandiset_id: str
    version_id: str
    metadata_path: Path
    asset: AssetChoice
    nwb_path: Path
    cached: bool


def _metadata_cache_path(dandiset_id: str, version_id: str) -> Path:
    return METADATA_ROOT / dandiset_id / f"{version_id}.metadata.json"


def _asset_cache_path(asset: AssetChoice) -> Path:
    return (
        CACHE_ROOT
        / asset.dandiset_id
        / asset.version_id
        / _flatten_asset_path(asset.asset_path)
    )


def fetch_dandiset(
    dandiset_id: str,
    version: str | None = None,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
) -> FetchResult:
    """Resolve a dandiset id to a local NWB + cached metadata.

    Steps:
      1. Hit the DANDI API to resolve the dandiset version.
      2. Cache raw metadata under ``metadata/<id>/<version>.metadata.json``.
      3. Pick a representative NWB asset (smallest, units-only-preferred).
      4. Download it under ``cache/<id>/<version>/<flattened_path>`` if missing.

    Raises ``RuntimeError`` if no suitable asset is found within
    ``max_size_bytes`` (default 500 MB).
    """
    if max_size_bytes > ABSOLUTE_MAX_SIZE_BYTES:
        raise ValueError(
            f"max_size_bytes={max_size_bytes} exceeds hard cap {ABSOLUTE_MAX_SIZE_BYTES}"
        )

    dandiset_id = _normalize_dandiset_id(dandiset_id)

    # Lazy import — keeps `dandi` off the import path for unit tests that
    # don't need network.
    from dandi.dandiapi import DandiAPIClient

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        # Prefer the most recent published version if one exists; that's
        # typically the cleanest snapshot for analysis. Fall back to draft.
        base = client.get_dandiset(dandiset_id, version or "draft")
        if version is None:
            published = getattr(base, "most_recent_published_version", None)
            if published is not None:
                try:
                    dandiset = base.for_version(published)
                except Exception:
                    dandiset = base
            else:
                dandiset = base
        else:
            dandiset = base
        version_id = str(getattr(dandiset, "version_id", "draft"))

        meta_path = _metadata_cache_path(dandiset_id, version_id)
        if not meta_path.exists():
            raw_metadata = dandiset.get_raw_metadata()
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                json.dumps(_safe_jsonable(raw_metadata), indent=2, ensure_ascii=False)
            )

        chosen = select_representative_asset(dandiset, max_size_bytes=max_size_bytes)
        if chosen is None:
            raise RuntimeError(
                f"No .nwb asset under {max_size_bytes} bytes found in DANDI:{dandiset_id}. "
                f"Raise --max-size-bytes or fetch a specific asset manually."
            )

        dest = _asset_cache_path(chosen)
        cached = dest.exists() and dest.stat().st_size > 0
        if not cached:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # The DandiAPIClient asset handle is what knows how to download.
            asset_handle = next(
                a for a in dandiset.get_assets() if str(getattr(a, "identifier", "")) == chosen.asset_id
            )
            asset_handle.download(dest)

    return FetchResult(
        dandiset_id=dandiset_id,
        version_id=version_id,
        metadata_path=meta_path,
        asset=chosen,
        nwb_path=dest,
        cached=cached,
    )
