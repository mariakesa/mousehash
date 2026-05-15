#!/usr/bin/env python3
"""
Fetch lightweight-ish spike-timing NWB assets from DANDI.

This script searches DANDI asset metadata and keeps only .nwb assets whose
`variableMeasured` field is non-empty and contains ONLY:

    - Units
    - SpikeEventSeries

That is, these pass:
    {"Units"}
    {"SpikeEventSeries"}
    {"Units", "SpikeEventSeries"}

These do NOT pass:
    {"Units", "ElectricalSeries"}
    {"LFP"}
    {"ElectricalSeries"}
    missing/empty variableMeasured

Default behavior is conservative: download only the first matching asset.
Use --download-all to download every matching asset found within the scan limits.

Install:
    pip install dandi

Examples:
    # Conservative: download one matching Units/SpikeEventSeries-only NWB
    python fetch_dandi_units_only_nwbs.py

    # Search a specific Dandiset
    python fetch_dandi_units_only_nwbs.py --dandiset-id 000006 --version draft

    # Download every matching asset found within limits
    python fetch_dandi_units_only_nwbs.py --download-all --max-dandisets 500 --max-assets-per-dandiset 500

    # List matches without downloading
    python fetch_dandi_units_only_nwbs.py --dry-run --download-all

    # Require the Dandiset/asset to also mention extracellular ephys measurementTechnique
    python fetch_dandi_units_only_nwbs.py --require-ephys-technique
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from dandi.dandiapi import DandiAPIClient

DEFAULT_OUTPUT_DIR = Path("/home/maria/mousehash/src/DANDI-agent/dandi_nwb_cache")
DEFAULT_MANIFEST_NAME = "selected_dandi_units_only_nwbs_manifest.json"

ALLOWED_VARIABLES = {"Units", "SpikeEventSeries"}
TARGET_EPHYS_TECHNIQUE = "multi electrode extracellular electrophysiology recording technique"


@dataclass
class Candidate:
    dandiset_id: str
    version_id: str
    asset_id: Optional[str]
    asset_path: str
    download_url: Optional[str]
    size_bytes: Optional[int]
    variable_measured: list[str]
    measurement_technique: list[str]


def norm_text(x: Any) -> str:
    """Normalize text for robust metadata matching."""
    return str(x).strip().casefold()


def jsonable_metadata(md: Any) -> dict[str, Any]:
    """Convert dandischema/Pydantic metadata objects to plain dictionaries when possible."""
    if md is None:
        return {}
    if isinstance(md, dict):
        return md

    # dandischema/Pydantic versions differ, so try the common APIs.
    for method_name in ("model_dump", "dict", "json_dict"):
        method = getattr(md, method_name, None)
        if callable(method):
            try:
                return method(exclude_none=True)
            except TypeError:
                try:
                    return method()
                except Exception:
                    pass
            except Exception:
                pass

    return {}


def iter_leaf_values(obj: Any) -> Iterable[Any]:
    """Yield leaves from nested dict/list metadata."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_leaf_values(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from iter_leaf_values(value)
    else:
        yield obj


def unique_preserve_order(values: Iterable[Any]) -> list[str]:
    """Return unique, non-empty string values while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        s = str(value).strip()
        if not s:
            continue
        key = norm_text(s)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def extract_named_list(md: Any, field_name: str) -> list[str]:
    """
    Extract names from metadata fields that may appear as:
      metadata.variableMeasured = [object(name="Units"), ...]
      metadata["variableMeasured"] = [{"name": "Units"}, ...]
      metadata["variableMeasured"] = ["Units", ...]
    """
    values: list[Any] = []

    # Object-style metadata.
    field = getattr(md, field_name, None)
    if field:
        if isinstance(field, (str, bytes)):
            values.append(field)
        else:
            try:
                iterator = iter(field)
            except TypeError:
                iterator = iter([field])
            for item in iterator:
                if isinstance(item, str):
                    values.append(item)
                else:
                    values.append(getattr(item, "name", None))

    # Dict-style metadata fallback.
    mdict = jsonable_metadata(md)
    field2 = mdict.get(field_name)
    if field2:
        if isinstance(field2, dict):
            field2 = [field2]
        elif isinstance(field2, (str, bytes)):
            field2 = [field2]

        for item in field2:
            if isinstance(item, dict):
                values.append(item.get("name") or item.get("schemaKey") or item.get("identifier"))
            else:
                values.append(item)

    return unique_preserve_order(values)


def extract_variable_measured(md: Any) -> list[str]:
    return extract_named_list(md, "variableMeasured")


def extract_measurement_technique(md: Any) -> list[str]:
    return extract_named_list(md, "measurementTechnique")


def metadata_mentions_ephys_technique(md: Any) -> bool:
    """Return True if metadata mentions the target extracellular ephys technique."""
    target = norm_text(TARGET_EPHYS_TECHNIQUE)

    names = extract_measurement_technique(md)
    if any(target == norm_text(name) or target in norm_text(name) for name in names):
        return True

    # Fallback for messy metadata objects.
    mdict = jsonable_metadata(md)
    return any(target in norm_text(value) for value in iter_leaf_values(mdict))


def is_units_only_metadata(md: Any) -> tuple[bool, list[str]]:
    """
    Check whether asset metadata is exactly Units/SpikeEventSeries-only.

    Returns:
        (is_match, variable_names)
    """
    variables = extract_variable_measured(md)
    variable_set = set(variables)

    if not variable_set:
        return False, variables

    is_match = variable_set.issubset(ALLOWED_VARIABLES) and bool(variable_set & ALLOWED_VARIABLES)
    return is_match, variables


def parse_size_bytes(value: Any) -> Optional[int]:
    """Best-effort conversion of DANDI contentSize/asset.size to int bytes."""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def get_asset_size_bytes(asset: Any, md: Any = None) -> Optional[int]:
    """Read size from the asset object or metadata if available."""
    for attr in ("size", "blob_size", "content_size"):
        size = parse_size_bytes(getattr(asset, attr, None))
        if size is not None:
            return size

    mdict = jsonable_metadata(md)
    for key in ("contentSize", "size", "blobSize"):
        size = parse_size_bytes(mdict.get(key))
        if size is not None:
            return size

    return None


def safe_flat_filename(dandiset_id: str, version_id: str, asset_path: str) -> str:
    """Flatten a DANDI asset path into a reasonably safe filename."""
    return f"{dandiset_id}__{version_id}__{asset_path}".replace("/", "__")


def local_download_path(candidate: Candidate, output_dir: Path, preserve_tree: bool) -> Path:
    """Choose the destination path for an asset."""
    if preserve_tree:
        return output_dir / candidate.dandiset_id / candidate.version_id / candidate.asset_path
    return output_dir / safe_flat_filename(candidate.dandiset_id, candidate.version_id, candidate.asset_path)


def download_asset(asset: Any, dest: Path) -> Path:
    """Download one asset unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache hit] {dest}")
        return dest

    print(f"[download] {getattr(asset, 'download_url', None)}")
    print(f"[to]       {dest}")
    asset.download(dest)
    return dest


def candidate_dandisets(
    client: DandiAPIClient,
    dandiset_id: Optional[str],
    version: Optional[str],
) -> Iterator[Any]:
    """Yield DANDI RemoteDandiset objects."""
    if dandiset_id:
        yield client.get_dandiset(dandiset_id, version or "draft")
        return

    for dandiset in client.get_dandisets():
        try:
            most_recent = getattr(dandiset, "most_recent_published_version", None)
            if most_recent:
                yield dandiset.for_version(most_recent)
            else:
                yield dandiset.for_version("draft")
        except Exception as exc:
            print(f"[skip dandiset version] {dandiset}: {exc}", file=sys.stderr)


def build_candidate(dandiset: Any, asset: Any, asset_md: Any) -> Candidate:
    variables = extract_variable_measured(asset_md)
    techniques = extract_measurement_technique(asset_md)
    size = get_asset_size_bytes(asset, asset_md)

    return Candidate(
        dandiset_id=str(getattr(dandiset, "identifier", "unknown_dandiset")),
        version_id=str(getattr(dandiset, "version_id", "unknown_version")),
        asset_id=getattr(asset, "identifier", None),
        asset_path=str(getattr(asset, "path", "")),
        download_url=getattr(asset, "download_url", None),
        size_bytes=size,
        variable_measured=variables,
        measurement_technique=techniques,
    )


def scan_dandiset_for_candidates(
    dandiset: Any,
    max_assets_per_dandiset: int,
    require_ephys_technique: bool,
    debug_errors: bool,
) -> list[tuple[Candidate, Any]]:
    """
    Scan one Dandiset and return matching (candidate, asset_object) pairs.

    Matching is asset-level: the NWB asset itself must have variableMeasured
    containing only Units and/or SpikeEventSeries.
    """
    candidates: list[tuple[Candidate, Any]] = []

    dandiset_ephys_match = False
    if require_ephys_technique:
        try:
            dandiset_md = dandiset.get_metadata()
            dandiset_ephys_match = metadata_mentions_ephys_technique(dandiset_md)
        except Exception as exc:
            print(f"[metadata warning] could not read Dandiset metadata: {exc}", file=sys.stderr)
            if debug_errors:
                traceback.print_exc()

    try:
        assets_iter = dandiset.get_assets()
    except Exception as exc:
        print(f"[asset warning] could not list assets for {dandiset}: {exc}", file=sys.stderr)
        return candidates

    for asset_i, asset in enumerate(assets_iter, start=1):
        if asset_i > max_assets_per_dandiset:
            print(f"[asset limit] stopped after {max_assets_per_dandiset} assets in {dandiset}")
            break

        path = str(getattr(asset, "path", "") or "")
        if not path.endswith(".nwb"):
            continue

        try:
            asset_md = asset.get_metadata()
        except Exception as exc:
            print(f"[asset metadata warning] {path}: {exc}", file=sys.stderr)
            if debug_errors:
                traceback.print_exc()
            continue

        is_units_only, variables = is_units_only_metadata(asset_md)
        if not is_units_only:
            print(f"[skip] {path} variableMeasured={variables}")
            continue

        if require_ephys_technique:
            asset_ephys_match = metadata_mentions_ephys_technique(asset_md)
            if not (asset_ephys_match or dandiset_ephys_match):
                print(f"[skip ephys technique] {path} variableMeasured={variables}")
                continue

        candidate = build_candidate(dandiset, asset, asset_md)
        candidates.append((candidate, asset))
        print(
            f"[candidate] dandiset={candidate.dandiset_id} "
            f"version={candidate.version_id} size={candidate.size_bytes} "
            f"variables={candidate.variable_measured} path={candidate.asset_path}"
        )

    # Prefer smaller files first when choosing only one or one per Dandiset.
    candidates.sort(key=lambda pair: pair[0].size_bytes if pair[0].size_bytes is not None else float("inf"))
    return candidates


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selector": {
            "asset_extension": ".nwb",
            "variableMeasured_rule": "non-empty subset of allowed_variables",
            "allowed_variables": sorted(ALLOWED_VARIABLES),
            "excluded_by_rule_examples": ["ElectricalSeries", "LFP", "TwoPhotonSeries"],
        },
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download DANDI NWB assets whose variableMeasured is only Units/SpikeEventSeries."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--dandiset-id", default=None, help="Restrict to one Dandiset, e.g. 000006")
    parser.add_argument("--version", default=None, help="Dandiset version, e.g. draft or 0.220126.1855")
    parser.add_argument("--max-dandisets", type=int, default=200)
    parser.add_argument("--max-assets-per-dandiset", type=int, default=200)
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download every matching asset found within scan limits. Without this, stop after one download.",
    )
    parser.add_argument(
        "--one-per-dandiset",
        action="store_true",
        help="When used with --download-all, download only the smallest matching asset from each Dandiset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching candidates and write manifest, but do not download files.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Flatten output filenames instead of preserving DANDI asset paths under dandiset/version folders.",
    )
    parser.add_argument(
        "--require-ephys-technique",
        action="store_true",
        help=(
            "Also require Dandiset or asset metadata to mention "
            f"measurementTechnique={TARGET_EPHYS_TECHNIQUE!r}."
        ),
    )
    parser.add_argument("--debug-errors", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or (args.output_dir / DEFAULT_MANIFEST_NAME)

    print(f"[output dir] {args.output_dir}")
    print(f"[allowed variableMeasured only] {sorted(ALLOWED_VARIABLES)}")
    print(f"[download mode] {'all matches' if args.download_all else 'first match only'}")
    if args.dry_run:
        print("[dry run] no files will be downloaded")

    checked_dandisets = 0
    checked_assets_or_candidates = 0
    records: list[dict[str, Any]] = []

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        for dandiset in candidate_dandisets(client, args.dandiset_id, args.version):
            checked_dandisets += 1
            if not args.dandiset_id and checked_dandisets > args.max_dandisets:
                break

            dandiset_id = getattr(dandiset, "identifier", "unknown")
            version_id = getattr(dandiset, "version_id", "unknown")
            print(f"\n[dandiset] {dandiset_id} version={version_id}")

            matches = scan_dandiset_for_candidates(
                dandiset=dandiset,
                max_assets_per_dandiset=args.max_assets_per_dandiset,
                require_ephys_technique=args.require_ephys_technique,
                debug_errors=args.debug_errors,
            )

            if args.one_per_dandiset and matches:
                matches = matches[:1]

            for candidate, asset in matches:
                checked_assets_or_candidates += 1
                dest = local_download_path(candidate, args.output_dir, preserve_tree=not args.flat)

                record = asdict(candidate)
                record["local_path"] = str(dest)
                record["downloaded"] = False
                record["dry_run"] = bool(args.dry_run)

                if not args.dry_run:
                    download_asset(asset, dest)
                    record["downloaded"] = True

                records.append(record)
                write_manifest(manifest_path, records)
                print(f"[manifest updated] {manifest_path}")

                if not args.download_all:
                    print("[done] stopped after first matching asset")
                    return 0

    write_manifest(manifest_path, records)

    if not records:
        print(
            f"\n[not found] Checked {checked_dandisets} Dandisets without finding .nwb assets "
            f"whose variableMeasured is only {sorted(ALLOWED_VARIABLES)}.",
            file=sys.stderr,
        )
        print("Try increasing --max-dandisets / --max-assets-per-dandiset or pass --dandiset-id.", file=sys.stderr)
        return 2

    print(f"\n[done] records={len(records)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
