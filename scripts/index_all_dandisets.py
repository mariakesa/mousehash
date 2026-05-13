"""Bulk-build MouseHash role manifests for DANDI dandisets.

Edit the two constants below before running:

    SAVE_PATH: where fetched metadata and representative NWB files are cached.
    JSON_INDICES: where inferred manifest JSON files and the run summary are written.

Examples:
    python scripts/index_all_dandisets.py
    python scripts/index_all_dandisets.py --max-dandisets 25
    python scripts/index_all_dandisets.py --dandiset-id 000011
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Update these two paths for your environment.
SAVE_PATH = Path("/media/maria/notsudata/DANDIRepresentativeData")
JSON_INDICES = Path("/media/maria/notsudata/MousehashManifests")

MAX_SIZE_MB = 500
MAX_ASSETS_TO_SCAN = 100

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

from dandi.dandiapi import DandiAPIClient

from mousehash.agents.dandi_agent.catalogs.loaders import tools_catalog_version
from mousehash.agents.dandi_agent.fetcher import (
    _normalize_dandiset_id,
    select_representative_asset,
)
from mousehash.agents.dandi_agent.parser import parse_mousehash_roles


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _safe_jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _safe_jsonable(value.dict())
        except Exception:
            pass
    return str(value)


def _flatten_asset_path(asset_path: str) -> str:
    return asset_path.replace("/", "__")


def _metadata_cache_path(dandiset_id: str, version_id: str) -> Path:
    return SAVE_PATH / "metadata" / dandiset_id / f"{version_id}.metadata.json"


def _nwb_cache_path(dandiset_id: str, version_id: str, asset_path: str) -> Path:
    return SAVE_PATH / "nwb" / dandiset_id / version_id / _flatten_asset_path(asset_path)


def _manifest_path(dandiset_id: str, version_id: str) -> Path:
    return JSON_INDICES / f"{dandiset_id}_{version_id}.manifest.json"


def _summary_path() -> Path:
    return JSON_INDICES / "index_summary.json"


def _resolve_versioned_dandiset(client: DandiAPIClient, dandiset_id: str, version: str | None) -> Any:
    base = client.get_dandiset(dandiset_id, version or "draft")
    if version is not None:
        return base
    published = getattr(base, "most_recent_published_version", None)
    if published is None:
        return base
    try:
        return base.for_version(published)
    except Exception:
        return base


def _iter_dandisets(client: DandiAPIClient, dandiset_id: str | None, version: str | None):
    if dandiset_id:
        yield _resolve_versioned_dandiset(client, _normalize_dandiset_id(dandiset_id), version)
        return

    for dandiset in client.get_dandisets():
        dandiset_identifier = str(getattr(dandiset, "identifier", ""))
        if not dandiset_identifier:
            continue
        try:
            yield _resolve_versioned_dandiset(client, dandiset_identifier, version)
        except Exception as exc:
            logger.warning("Skipping %s: could not resolve version (%s)", dandiset_identifier, exc)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _download_asset_if_needed(asset_handle: Any, destination: Path) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    asset_handle.download(destination)
    return True


def index_one_dandiset(dandiset: Any, max_size_bytes: int) -> dict[str, Any]:
    dandiset_id = str(getattr(dandiset, "identifier", "unknown"))
    version_id = str(getattr(dandiset, "version_id", "draft"))

    logger.info("Indexing DANDI:%s version=%s", dandiset_id, version_id)

    raw_metadata = _safe_jsonable(dandiset.get_raw_metadata())
    metadata_path = _metadata_cache_path(dandiset_id, version_id)
    _write_json(metadata_path, raw_metadata)

    selected_asset = select_representative_asset(
        dandiset,
        max_size_bytes=max_size_bytes,
        max_assets_to_scan=MAX_ASSETS_TO_SCAN,
    )

    nwb_path: Path | None = None
    downloaded = False
    manifest_source = "metadata_only"

    if selected_asset is not None:
        nwb_path = _nwb_cache_path(
            dandiset_id,
            version_id,
            selected_asset.asset_path,
        )
        asset_handle = next(
            asset
            for asset in dandiset.get_assets()
            if str(getattr(asset, "identifier", "")) == selected_asset.asset_id
        )
        downloaded = _download_asset_if_needed(asset_handle, nwb_path)
        manifest_source = "metadata_plus_nwb"

    manifest = parse_mousehash_roles(
        dandiset_id=dandiset_id,
        dandiset_metadata=raw_metadata,
        nwb_path=str(nwb_path) if nwb_path is not None else None,
        asset_id=selected_asset.asset_id if selected_asset is not None else None,
        catalog_version=tools_catalog_version(),
    )

    manifest_path = _manifest_path(dandiset_id, version_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    return {
        "dandiset_id": dandiset_id,
        "version_id": version_id,
        "name": raw_metadata.get("name", ""),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "manifest_source": manifest_source,
        "asset_id": selected_asset.asset_id if selected_asset is not None else None,
        "asset_path": selected_asset.asset_path if selected_asset is not None else None,
        "asset_size_bytes": selected_asset.size_bytes if selected_asset is not None else None,
        "nwb_local_path": str(nwb_path) if nwb_path is not None else None,
        "downloaded": downloaded,
        "n_role_paths": len(manifest.roles),
        "n_present": sum(1 for role in manifest.roles.values() if role.status == "present"),
        "n_likely_present": sum(1 for role in manifest.roles.values() if role.status == "likely_present"),
        "n_derived_possible": sum(
            1 for role in manifest.roles.values() if role.status == "derived_possible"
        ),
        "warnings": manifest.warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dandiset-id", default=None, help="Restrict to one DANDI id, e.g. 000011.")
    parser.add_argument("--version", default=None, help="Optional explicit DANDI version.")
    parser.add_argument("--max-dandisets", type=int, default=None)
    parser.add_argument("--max-size-mb", type=int, default=MAX_SIZE_MB)
    args = parser.parse_args()

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    JSON_INDICES.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        for index, dandiset in enumerate(_iter_dandisets(client, args.dandiset_id, args.version), start=1):
            if args.max_dandisets is not None and index > args.max_dandisets:
                break
            try:
                record = index_one_dandiset(dandiset, max_size_bytes=args.max_size_mb * 1024 * 1024)
                records.append(record)
                logger.info(
                    "Saved manifest for DANDI:%s at %s",
                    record["dandiset_id"],
                    record["manifest_path"],
                )
            except Exception as exc:
                dandiset_id = str(getattr(dandiset, "identifier", "unknown"))
                version_id = str(getattr(dandiset, "version_id", "unknown"))
                logger.exception("Failed to index DANDI:%s version=%s", dandiset_id, version_id)
                errors.append(
                    {
                        "dandiset_id": dandiset_id,
                        "version_id": version_id,
                        "error": str(exc),
                    }
                )
            _write_json(
                _summary_path(),
                {
                    "save_path": str(SAVE_PATH),
                    "json_indices": str(JSON_INDICES),
                    "records": records,
                    "errors": errors,
                },
            )

    logger.info("Indexed %d dandisets with %d errors", len(records), len(errors))
    logger.info("Summary written to %s", _summary_path())
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())