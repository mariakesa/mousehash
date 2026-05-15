#!/usr/bin/env python3
"""
Fetch one NWB asset from DANDI whose metadata contains
measurementTechnique = 'multi electrode extracellular electrophysiology recording technique'.

Default output folder:
    /home/maria/mousehash/src/DANDI-agent/dandi_nwb_cache

Install:
    pip install dandi

Example:
    python fetch_dandi_ephys_nwb.py
    python fetch_dandi_ephys_nwb.py --max-dandisets 50 --max-assets-per-dandiset 25
    python fetch_dandi_ephys_nwb.py --dandiset-id 000006 --version draft
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

from dandi.dandiapi import DandiAPIClient

TARGET_TECHNIQUE = "multi electrode extracellular electrophysiology recording technique"
DEFAULT_OUTPUT_DIR = Path("/home/maria/mousehash/src/DANDI-agent/dandi_nwb_cache")


def norm_text(x: Any) -> str:
    """Normalize strings for robust metadata matching."""
    return str(x).strip().casefold()


def jsonable_metadata(md: Any) -> dict[str, Any]:
    """Convert DANDI/dandischema metadata object to a plain dict when possible."""
    if md is None:
        return {}
    if isinstance(md, dict):
        return md
    for method in ("json_dict", "model_dump", "dict"):
        fn = getattr(md, method, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                try:
                    return fn(exclude_none=True)
                except Exception:
                    pass
    return {}


def iter_values(obj: Any) -> Iterable[Any]:
    """Recursively walk a nested metadata object/dict/list and yield leaf-ish values."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iter_values(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from iter_values(v)
    else:
        yield obj


def extract_measurement_technique_names(md: Any) -> list[str]:
    """
    Extract measurementTechnique names from either dandischema objects or dict metadata.
    Handles common forms:
      metadata.measurementTechnique = [object(name=...), ...]
      metadata['measurementTechnique'] = [{'name': ...}, ...]
    """
    names: list[str] = []

    # Object-style metadata from dandi/dandischema
    mts = getattr(md, "measurementTechnique", None)
    if mts:
        for mt in mts:
            name = getattr(mt, "name", None)
            if name:
                names.append(str(name))

    # Dict-style fallback
    mdict = jsonable_metadata(md)
    mts2 = mdict.get("measurementTechnique") or []
    if isinstance(mts2, dict):
        mts2 = [mts2]
    for mt in mts2:
        if isinstance(mt, dict):
            name = mt.get("name") or mt.get("schemaKey") or mt.get("identifier")
            if name:
                names.append(str(name))
        elif mt:
            names.append(str(mt))

    # Preserve order, remove duplicates
    seen = set()
    unique = []
    for name in names:
        key = norm_text(name)
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def metadata_has_target_technique(md: Any, target: str = TARGET_TECHNIQUE) -> bool:
    """True if measurementTechnique contains target. Falls back to recursive text search."""
    target_n = norm_text(target)

    technique_names = extract_measurement_technique_names(md)
    if any(target_n == norm_text(name) or target_n in norm_text(name) for name in technique_names):
        return True

    # Conservative fallback for messy metadata: only search the plain dict conversion.
    mdict = jsonable_metadata(md)
    return any(target_n in norm_text(v) for v in iter_values(mdict))


def safe_filename(path: str) -> str:
    """Flatten DANDI asset path into a filesystem-safe-ish name while preserving .nwb."""
    return path.replace("/", "__")


def download_asset(asset: Any, output_dir: Path, preserve_tree: bool = True) -> Path:
    """Download asset using the DANDI client asset.download method."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if preserve_tree:
        dest = output_dir / asset.path
    else:
        dest = output_dir / safe_filename(asset.path)

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache hit] {dest}")
        return dest

    print(f"[download] {asset.download_url}")
    print(f"[to]       {dest}")
    asset.download(dest)
    return dest


def candidate_dandisets(client: DandiAPIClient, dandiset_id: Optional[str], version: Optional[str]):
    """Yield RemoteDandiset objects."""
    if dandiset_id:
        yield client.get_dandiset(dandiset_id, version)
        return

    for d in client.get_dandisets():
        # Prefer most recent published version. Fall back to draft if no published version exists.
        try:
            if getattr(d, "most_recent_published_version", None):
                yield d.for_version(d.most_recent_published_version)
            else:
                yield d.for_version("draft")
        except Exception as exc:
            print(f"[skip dandiset version] {d}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-technique", default=TARGET_TECHNIQUE)
    parser.add_argument("--dandiset-id", default=None, help="Optional: restrict to one Dandiset, e.g. 000006")
    parser.add_argument("--version", default=None, help="Optional version, e.g. draft or 0.220126.1855")
    parser.add_argument("--max-dandisets", type=int, default=200, help="Safety limit when scanning all Dandisets")
    parser.add_argument("--max-assets-per-dandiset", type=int, default=100, help="Safety limit per Dandiset")
    parser.add_argument("--flat", action="store_true", help="Do not preserve DANDI directory tree under output-dir")
    parser.add_argument("--debug-errors", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[target measurementTechnique] {args.target_technique}")
    print(f"[output dir] {args.output_dir}")

    checked_dandisets = 0
    checked_assets = 0

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        for dandiset in candidate_dandisets(client, args.dandiset_id, args.version):
            checked_dandisets += 1
            if not args.dandiset_id and checked_dandisets > args.max_dandisets:
                break

            print(f"\n[dandiset] {dandiset}")

            # First test Dandiset-level metadata. This is fast and gives us a coarse filter.
            dandiset_md_matches = False
            try:
                dandiset_md = dandiset.get_metadata()
                dandiset_md_matches = metadata_has_target_technique(dandiset_md, args.target_technique)
                if dandiset_md_matches:
                    names = extract_measurement_technique_names(dandiset_md)
                    print(f"[dandiset metadata match] measurementTechnique={names}")
            except Exception as exc:
                print(f"[metadata warning] could not read Dandiset metadata: {exc}", file=sys.stderr)
                if args.debug_errors:
                    traceback.print_exc()

            asset_i = 0
            try:
                assets_iter = dandiset.get_assets()
            except Exception as exc:
                print(f"[asset warning] could not list assets: {exc}", file=sys.stderr)
                continue

            for asset in assets_iter:
                asset_i += 1
                checked_assets += 1
                if asset_i > args.max_assets_per_dandiset:
                    print(f"[asset limit] stopped after {args.max_assets_per_dandiset} assets in {dandiset}")
                    break

                path = getattr(asset, "path", "") or ""
                if not path.endswith(".nwb"):
                    continue

                asset_matches = dandiset_md_matches
                technique_names: list[str] = []

                # Asset metadata is often more precise than Dandiset metadata.
                try:
                    asset_md = asset.get_metadata()
                    technique_names = extract_measurement_technique_names(asset_md)
                    asset_matches = metadata_has_target_technique(asset_md, args.target_technique) or asset_matches
                except Exception as exc:
                    print(f"[asset metadata warning] {path}: {exc}", file=sys.stderr)
                    if args.debug_errors:
                        traceback.print_exc()

                if not asset_matches:
                    continue

                print(f"[match] dandiset={dandiset.identifier} version={dandiset.version_id}")
                print(f"[asset] {path}")
                if technique_names:
                    print(f"[asset measurementTechnique] {technique_names}")

                local_path = download_asset(asset, args.output_dir, preserve_tree=not args.flat)

                manifest = {
                    "dandiset_id": dandiset.identifier,
                    "version_id": dandiset.version_id,
                    "asset_id": getattr(asset, "identifier", None),
                    "asset_path": path,
                    "download_url": getattr(asset, "download_url", None),
                    "local_path": str(local_path),
                    "target_measurementTechnique": args.target_technique,
                    "asset_measurementTechnique": technique_names,
                }
                manifest_path = args.output_dir / "selected_dandi_nwb_manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"[manifest] {manifest_path}")
                print("[done]")
                return 0

    print(
        f"\n[not found] Checked {checked_dandisets} Dandisets and {checked_assets} assets "
        f"without finding an NWB asset matching measurementTechnique={args.target_technique!r}.",
        file=sys.stderr,
    )
    print("Try increasing --max-dandisets / --max-assets-per-dandiset or pass a known --dandiset-id.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
