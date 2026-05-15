#!/usr/bin/env python3
"""
Fetch spike-timing-oriented NWB assets from DANDI.

Default selector:
    Download NWB assets whose asset-level variableMeasured is a non-empty
    subset of {"Units", "SpikeEventSeries"}.

Why this exists:
    measurementTechnique="multi electrode extracellular electrophysiology recording technique"
    can match huge raw ElectricalSeries assets. For lightweight spike timing,
    we prefer asset metadata that says the file measures Units and/or
    SpikeEventSeries, and nothing else.

Install:
    pip install dandi

Examples:
    # Download the first matching Units/SpikeEventSeries-only NWB found
    python fetch_dandi_units_only_nwbs_v2.py

    # Scan one Dandiset
    python fetch_dandi_units_only_nwbs_v2.py --dandiset-id 000006 --version draft

    # Download all strict matches within scan limits
    python fetch_dandi_units_only_nwbs_v2.py --download-all --max-dandisets 500 --max-assets-per-dandiset 500

    # Do not download; write a diagnostic manifest showing what was seen
    python fetch_dandi_units_only_nwbs_v2.py --dry-run --verbose-skips

    # If strict mode finds nothing, try assets that CONTAIN Units/SpikeEventSeries,
    # even if they also contain raw variables. Useful for debugging metadata.
    python fetch_dandi_units_only_nwbs_v2.py --match-mode contains --dry-run --verbose-skips
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import Counter
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
    match_mode: str


@dataclass
class ScanStats:
    dandisets_seen: int = 0
    assets_seen: int = 0
    nwb_assets_seen: int = 0
    metadata_read_errors: int = 0
    candidates_seen: int = 0


def norm_text(x: Any) -> str:
    return str(x).strip().casefold()


def canonical_variable_name(name: Any) -> Optional[str]:
    """Map messy DANDI/NWB metadata names onto the names we care about."""
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None

    key = norm_text(s)
    if key in {"units", "unit"}:
        return "Units"
    if key in {"spikeeventseries", "spike event series", "spikeevents", "spike events"}:
        return "SpikeEventSeries"
    if key in {"electricalseries", "electrical series"}:
        return "ElectricalSeries"
    if key == "lfp":
        return "LFP"
    return s


def unique_preserve_order(values: Iterable[Any], canonicalize: bool = False) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = canonical_variable_name(value) if canonicalize else value
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


def jsonable_metadata(md: Any) -> dict[str, Any]:
    """Convert dandischema/Pydantic metadata objects to plain dictionaries when possible."""
    if md is None:
        return {}
    if isinstance(md, dict):
        return md

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


def collect_names_from_nested(obj: Any) -> list[str]:
    """
    Robustly collect possible names from a nested metadata object.

    DANDI metadata may represent variableMeasured as strings, dicts with name,
    schemaKey, identifier, nested value/name fields, or Pydantic objects.
    """
    values: list[Any] = []

    if obj is None:
        return []

    if isinstance(obj, str):
        return [obj]

    if isinstance(obj, dict):
        # Prefer explicit human-readable fields.
        for key in ("name", "label", "value", "schemaKey", "identifier"):
            if key in obj and isinstance(obj[key], str):
                values.append(obj[key])
        # Recurse too, because some entries look like {"value": {"name": "Units"}}.
        for value in obj.values():
            if isinstance(value, (dict, list, tuple, set)):
                values.extend(collect_names_from_nested(value))
        return values

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            values.extend(collect_names_from_nested(item))
        return values

    # Pydantic-ish/object fallback.
    for attr in ("name", "label", "value", "schemaKey", "identifier"):
        value = getattr(obj, attr, None)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (dict, list, tuple, set)):
            values.extend(collect_names_from_nested(value))

    obj_dict = jsonable_metadata(obj)
    if obj_dict:
        values.extend(collect_names_from_nested(obj_dict))

    return values


def extract_named_list(md: Any, field_name: str, canonicalize: bool = False) -> list[str]:
    values: list[Any] = []

    field = getattr(md, field_name, None)
    values.extend(collect_names_from_nested(field))

    mdict = jsonable_metadata(md)
    if field_name in mdict:
        values.extend(collect_names_from_nested(mdict.get(field_name)))

    return unique_preserve_order(values, canonicalize=canonicalize)


def extract_variable_measured(md: Any) -> list[str]:
    return extract_named_list(md, "variableMeasured", canonicalize=True)


def extract_measurement_technique(md: Any) -> list[str]:
    return extract_named_list(md, "measurementTechnique", canonicalize=False)


def metadata_mentions_ephys_technique(md: Any) -> bool:
    target = norm_text(TARGET_EPHYS_TECHNIQUE)

    names = extract_measurement_technique(md)
    if any(target == norm_text(name) or target in norm_text(name) for name in names):
        return True

    mdict = jsonable_metadata(md)
    return any(target in norm_text(value) for value in iter_leaf_values(mdict))


def variable_match(md: Any, match_mode: str) -> tuple[bool, list[str]]:
    variables = extract_variable_measured(md)
    variable_set = set(variables)

    if not variable_set:
        return False, variables

    if match_mode == "exact":
        is_match = variable_set.issubset(ALLOWED_VARIABLES) and bool(variable_set & ALLOWED_VARIABLES)
    elif match_mode == "contains":
        is_match = bool(variable_set & ALLOWED_VARIABLES)
    else:
        raise ValueError(f"Unknown match_mode={match_mode!r}")

    return is_match, variables


def parse_size_bytes(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        pass

    # Sometimes contentSize can be a string like "12345 bytes".
    m = re.search(r"\d+", str(value))
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def get_asset_size_bytes(asset: Any, md: Any = None) -> Optional[int]:
    for attr in ("size", "blob_size", "content_size", "contentSize"):
        size = parse_size_bytes(getattr(asset, attr, None))
        if size is not None:
            return size

    mdict = jsonable_metadata(md)
    for key in ("contentSize", "size", "blobSize", "content_size"):
        size = parse_size_bytes(mdict.get(key))
        if size is not None:
            return size
    return None


def safe_flat_filename(dandiset_id: str, version_id: str, asset_path: str) -> str:
    return f"{dandiset_id}__{version_id}__{asset_path}".replace("/", "__")


def local_download_path(candidate: Candidate, output_dir: Path, preserve_tree: bool) -> Path:
    if preserve_tree:
        return output_dir / candidate.dandiset_id / candidate.version_id / candidate.asset_path
    return output_dir / safe_flat_filename(candidate.dandiset_id, candidate.version_id, candidate.asset_path)


def download_asset(asset: Any, dest: Path) -> Path:
    """Download one asset unless it already exists. Return the expected local path."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache hit] {dest}")
        return dest

    print(f"[download] {getattr(asset, 'download_url', None)}")
    print(f"[to]       {dest}")

    # DANDI's asset.download has varied slightly across versions. For file-like
    # targets, use the parent directory and preserve the asset path only when the
    # client supports a filepath argument cleanly. The direct dest call works in
    # many versions; verify afterward and print a helpful diagnostic if not.
    asset.download(dest)

    if not dest.exists():
        # Some client versions interpret the argument as a directory and place
        # the asset below it. Surface this rather than silently succeeding.
        possible = list(dest.parent.rglob(dest.name))
        if possible:
            print(f"[notice] DANDI client saved file at {possible[0]} instead of requested path")
            return possible[0]
        raise FileNotFoundError(f"Download finished but expected file was not found: {dest}")

    print(f"[saved]    {dest} ({dest.stat().st_size} bytes)")
    return dest


def candidate_dandisets(
    client: DandiAPIClient,
    dandiset_id: Optional[str],
    version: Optional[str],
) -> Iterator[Any]:
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


def build_candidate(dandiset: Any, asset: Any, asset_md: Any, match_mode: str) -> Candidate:
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
        match_mode=match_mode,
    )


def scan_dandiset_for_candidates(
    dandiset: Any,
    max_assets_per_dandiset: int,
    require_ephys_technique: bool,
    match_mode: str,
    verbose_skips: bool,
    debug_errors: bool,
    stats: ScanStats,
    variable_counter: Counter,
    sample_skips: list[dict[str, Any]],
    max_skip_samples: int,
) -> list[tuple[Candidate, Any]]:
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

        stats.assets_seen += 1
        path = str(getattr(asset, "path", "") or "")
        if not path.endswith(".nwb"):
            continue
        stats.nwb_assets_seen += 1

        try:
            asset_md = asset.get_metadata()
        except Exception as exc:
            stats.metadata_read_errors += 1
            print(f"[asset metadata warning] {path}: {exc}", file=sys.stderr)
            if debug_errors:
                traceback.print_exc()
            continue

        is_match, variables = variable_match(asset_md, match_mode=match_mode)
        variable_key = tuple(sorted(variables)) if variables else ("<missing variableMeasured>",)
        variable_counter[variable_key] += 1

        if not is_match:
            if verbose_skips:
                print(f"[skip] {path} variableMeasured={variables}")
            if len(sample_skips) < max_skip_samples:
                sample_skips.append({"path": path, "variableMeasured": variables})
            continue

        if require_ephys_technique:
            asset_ephys_match = metadata_mentions_ephys_technique(asset_md)
            if not (asset_ephys_match or dandiset_ephys_match):
                if verbose_skips:
                    print(f"[skip ephys technique] {path} variableMeasured={variables}")
                continue

        candidate = build_candidate(dandiset, asset, asset_md, match_mode=match_mode)
        candidates.append((candidate, asset))
        stats.candidates_seen += 1
        print(
            f"[candidate] dandiset={candidate.dandiset_id} "
            f"version={candidate.version_id} size={candidate.size_bytes} "
            f"variables={candidate.variable_measured} path={candidate.asset_path}"
        )

    candidates.sort(key=lambda pair: pair[0].size_bytes if pair[0].size_bytes is not None else float("inf"))
    return candidates


def write_manifest(
    path: Path,
    records: list[dict[str, Any]],
    stats: ScanStats,
    variable_counter: Counter,
    sample_skips: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selector": {
            "asset_extension": ".nwb",
            "match_mode": args.match_mode,
            "variableMeasured_rule_exact": "non-empty subset of allowed_variables",
            "variableMeasured_rule_contains": "contains at least one allowed variable",
            "allowed_variables": sorted(ALLOWED_VARIABLES),
            "require_ephys_technique": bool(args.require_ephys_technique),
            "excluded_by_exact_rule_examples": ["ElectricalSeries", "LFP", "TwoPhotonSeries"],
        },
        "scan_stats": asdict(stats),
        "variableMeasured_distribution_top_50": [
            {"variableMeasured": list(key), "count": count}
            for key, count in variable_counter.most_common(50)
        ],
        "sample_nonmatching_nwb_assets": sample_skips,
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
        "--match-mode",
        choices=["exact", "contains"],
        default="exact",
        help=(
            "exact = variableMeasured must be a non-empty subset of Units/SpikeEventSeries. "
            "contains = asset can contain Units/SpikeEventSeries plus other variables; useful for debugging."
        ),
    )
    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--one-per-dandiset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--require-ephys-technique", action="store_true")
    parser.add_argument("--verbose-skips", action="store_true")
    parser.add_argument("--max-skip-samples", type=int, default=200)
    parser.add_argument("--debug-errors", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or (args.output_dir / DEFAULT_MANIFEST_NAME)

    print(f"[output dir] {args.output_dir}")
    print(f"[manifest]   {manifest_path}")
    print(f"[allowed variableMeasured] {sorted(ALLOWED_VARIABLES)}")
    print(f"[match mode] {args.match_mode}")
    print(f"[download mode] {'all matches' if args.download_all else 'first match only'}")
    if args.dry_run:
        print("[dry run] no files will be downloaded")

    stats = ScanStats()
    variable_counter: Counter = Counter()
    sample_skips: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    try:
        with DandiAPIClient.for_dandi_instance("dandi") as client:
            for dandiset in candidate_dandisets(client, args.dandiset_id, args.version):
                stats.dandisets_seen += 1
                if not args.dandiset_id and stats.dandisets_seen > args.max_dandisets:
                    break

                dandiset_id = getattr(dandiset, "identifier", "unknown")
                version_id = getattr(dandiset, "version_id", "unknown")
                print(f"\n[dandiset] {dandiset_id} version={version_id}")

                matches = scan_dandiset_for_candidates(
                    dandiset=dandiset,
                    max_assets_per_dandiset=args.max_assets_per_dandiset,
                    require_ephys_technique=args.require_ephys_technique,
                    match_mode=args.match_mode,
                    verbose_skips=args.verbose_skips,
                    debug_errors=args.debug_errors,
                    stats=stats,
                    variable_counter=variable_counter,
                    sample_skips=sample_skips,
                    max_skip_samples=args.max_skip_samples,
                )

                if args.one_per_dandiset and matches:
                    matches = matches[:1]

                for candidate, asset in matches:
                    dest = local_download_path(candidate, args.output_dir, preserve_tree=not args.flat)

                    record = asdict(candidate)
                    record["local_path"] = str(dest)
                    record["downloaded"] = False
                    record["dry_run"] = bool(args.dry_run)

                    if not args.dry_run:
                        actual_path = download_asset(asset, dest)
                        record["local_path"] = str(actual_path)
                        record["downloaded"] = True
                        record["local_size_bytes"] = actual_path.stat().st_size if actual_path.exists() else None

                    records.append(record)
                    write_manifest(manifest_path, records, stats, variable_counter, sample_skips, args)
                    print(f"[manifest updated] {manifest_path}")

                    if not args.download_all:
                        print("[done] stopped after first matching asset")
                        return 0

    finally:
        write_manifest(manifest_path, records, stats, variable_counter, sample_skips, args)
        print(f"\n[manifest written] {manifest_path}")

    if not records:
        print(
            f"\n[not found] No matching NWB assets were downloaded/found. "
            f"Seen: dandisets={stats.dandisets_seen}, assets={stats.assets_seen}, "
            f"nwb_assets={stats.nwb_assets_seen}, metadata_errors={stats.metadata_read_errors}.",
            file=sys.stderr,
        )
        print(
            "The manifest still contains diagnostics. Next useful checks:\n"
            "  1) Open the manifest and inspect variableMeasured_distribution_top_50.\n"
            "  2) Try --match-mode contains --dry-run to see whether Units assets also include raw variables.\n"
            "  3) Increase --max-dandisets and --max-assets-per-dandiset.\n",
            file=sys.stderr,
        )
        return 2

    print(f"\n[done] records={len(records)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
