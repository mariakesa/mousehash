#!/usr/bin/env python3
"""
Fetch lightweight-ish spike/behavior NWB assets from DANDI.

Default target signature:
    variableMeasured exactly equals:
        BehavioralEvents
        BehavioralTimeSeries
        ElectrodeGroup
        PropertyValue
        Units

This is designed for the DANDI metadata pattern:
    {
      "variableMeasured": [
        "BehavioralEvents", "BehavioralTimeSeries", "ElectrodeGroup",
        "PropertyValue", "Units"
      ],
      "count": 190
    }

Why not only measurementTechnique?
    measurementTechnique='multi electrode extracellular electrophysiology recording technique'
    can match huge raw voltage assets. This script targets asset-level
    variableMeasured instead, so it prefers sorted units plus behavior and
    electrode metadata while rejecting raw-heavy files.

Install:
    pip install dandi

Examples:
    # Dry-run: find matching assets and write diagnostics, download nothing
    python fetch_dandi_light_spike_behavior_nwbs.py --dry-run

    # Download all assets whose variableMeasured exactly matches the target signature
    python fetch_dandi_light_spike_behavior_nwbs.py --download-all

    # Download only the first match per Dandiset
    python fetch_dandi_light_spike_behavior_nwbs.py --download-all --one-per-dandiset

    # Scan one Dandiset
    python fetch_dandi_light_spike_behavior_nwbs.py --dandiset-id 000XXX --version draft --download-all

    # More permissive: accept any Units asset whose variables are all lightweight
    python fetch_dandi_light_spike_behavior_nwbs.py --selector lightweight --download-all
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

DEFAULT_OUTPUT_DIR = Path("/home/maria/mousehash/src/DANDI-agent/dandi_nwb_cache/light_data")
DEFAULT_MANIFEST_NAME = "selected_dandi_light_spike_behavior_nwbs_manifest.json"

TARGET_EXACT_SIGNATURE = {
    "BehavioralEvents",
    "BehavioralTimeSeries",
    "ElectrodeGroup",
    "PropertyValue",
    "Units",
}

REQUIRED_SPIKE_VARIABLES = {"Units", "SpikeEventSeries"}

ALLOWED_LIGHTWEIGHT_VARIABLES = {
    "Units",
    "SpikeEventSeries",
    "BehavioralEvents",
    "BehavioralTimeSeries",
    "BehavioralEpochs",
    "ElectrodeGroup",
    "PropertyValue",
    "ProcessingModule",
}

DISALLOWED_HEAVY_VARIABLES = {
    "ElectricalSeries",
    "LFP",
    "VoltageClampSeries",
    "VoltageClampStimulusSeries",
    "CurrentClampSeries",
    "CurrentClampStimulusSeries",
    "PatchClampSeries",
    "TwoPhotonSeries",
    "OnePhotonSeries",
    "ImageSeries",
    "OptogeneticSeries",
    "DecompositionSeries",
    "Spectrum",
}

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
    selector: str


@dataclass
class ScanStats:
    dandisets_seen: int = 0
    assets_seen: int = 0
    nwb_assets_seen: int = 0
    metadata_read_errors: int = 0
    candidates_seen: int = 0
    downloaded: int = 0


def norm_text(x: Any) -> str:
    return str(x).strip().casefold()


def canonical_variable_name(name: Any) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None

    key = norm_text(s)
    aliases = {
        "unit": "Units",
        "units": "Units",
        "spikeeventseries": "SpikeEventSeries",
        "spike event series": "SpikeEventSeries",
        "spikeevents": "SpikeEventSeries",
        "spike events": "SpikeEventSeries",
        "behavioralevents": "BehavioralEvents",
        "behavioral events": "BehavioralEvents",
        "behavioraltimeseries": "BehavioralTimeSeries",
        "behavioral time series": "BehavioralTimeSeries",
        "behavioralepochs": "BehavioralEpochs",
        "behavioral epochs": "BehavioralEpochs",
        "electrodegroup": "ElectrodeGroup",
        "electrode group": "ElectrodeGroup",
        "propertyvalue": "PropertyValue",
        "property value": "PropertyValue",
        "processingmodule": "ProcessingModule",
        "processing module": "ProcessingModule",
        "electricalseries": "ElectricalSeries",
        "electrical series": "ElectricalSeries",
        "lfp": "LFP",
        "voltageclampseries": "VoltageClampSeries",
        "voltage clamp series": "VoltageClampSeries",
        "voltageclampstimulusseries": "VoltageClampStimulusSeries",
        "voltage clamp stimulus series": "VoltageClampStimulusSeries",
        "currentclampseries": "CurrentClampSeries",
        "current clamp series": "CurrentClampSeries",
        "currentclampstimulusseries": "CurrentClampStimulusSeries",
        "current clamp stimulus series": "CurrentClampStimulusSeries",
        "patchclampseries": "PatchClampSeries",
        "patch clamp series": "PatchClampSeries",
        "twophotonseries": "TwoPhotonSeries",
        "two photon series": "TwoPhotonSeries",
        "onephotonseries": "OnePhotonSeries",
        "one photon series": "OnePhotonSeries",
        "imageseries": "ImageSeries",
        "image series": "ImageSeries",
        "optogeneticseries": "OptogeneticSeries",
        "optogenetic series": "OptogeneticSeries",
        "decompositionseries": "DecompositionSeries",
        "decomposition series": "DecompositionSeries",
        "spectrum": "Spectrum",
    }
    return aliases.get(key, s)


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
    values: list[Any] = []
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        for key in ("name", "label", "value", "schemaKey", "identifier"):
            if key in obj and isinstance(obj[key], str):
                values.append(obj[key])
        for value in obj.values():
            if isinstance(value, (dict, list, tuple, set)):
                values.extend(collect_names_from_nested(value))
        return values
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            values.extend(collect_names_from_nested(item))
        return values
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


def variable_match(md: Any, selector: str) -> tuple[bool, list[str], str]:
    variables = extract_variable_measured(md)
    variable_set = set(variables)
    if not variable_set:
        return False, variables, "missing variableMeasured"

    if selector == "exact-target":
        ok = variable_set == TARGET_EXACT_SIGNATURE
        return ok, variables, "exact target signature" if ok else "not exact target signature"

    if selector == "lightweight":
        has_spikes = bool(variable_set & REQUIRED_SPIKE_VARIABLES)
        has_heavy = bool(variable_set & DISALLOWED_HEAVY_VARIABLES)
        known_lightweight = variable_set.issubset(ALLOWED_LIGHTWEIGHT_VARIABLES)
        ok = has_spikes and known_lightweight and not has_heavy
        if ok:
            return True, variables, "contains spike variables and only lightweight variables"
        return False, variables, f"has_spikes={has_spikes}, known_lightweight={known_lightweight}, has_heavy={has_heavy}"

    if selector == "contains-units-no-heavy":
        has_spikes = bool(variable_set & REQUIRED_SPIKE_VARIABLES)
        has_heavy = bool(variable_set & DISALLOWED_HEAVY_VARIABLES)
        ok = has_spikes and not has_heavy
        return ok, variables, "contains spike variables and no known heavy variables" if ok else f"has_spikes={has_spikes}, has_heavy={has_heavy}"

    raise ValueError(f"Unknown selector={selector!r}")


def parse_size_bytes(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        pass
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache hit] {dest}")
        return dest
    print(f"[download] {getattr(asset, 'download_url', None)}")
    print(f"[to]       {dest}")
    asset.download(dest)
    if not dest.exists():
        possible = list(dest.parent.rglob(dest.name))
        if possible:
            print(f"[notice] DANDI client saved file at {possible[0]} instead of requested path")
            return possible[0]
        raise FileNotFoundError(f"Download finished but expected file was not found: {dest}")
    print(f"[saved]    {dest} ({dest.stat().st_size} bytes)")
    return dest


def candidate_dandisets(client: DandiAPIClient, dandiset_id: Optional[str], version: Optional[str]) -> Iterator[Any]:
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


def build_candidate(dandiset: Any, asset: Any, asset_md: Any, selector: str) -> Candidate:
    return Candidate(
        dandiset_id=str(getattr(dandiset, "identifier", "unknown_dandiset")),
        version_id=str(getattr(dandiset, "version_id", "unknown_version")),
        asset_id=getattr(asset, "identifier", None),
        asset_path=str(getattr(asset, "path", "")),
        download_url=getattr(asset, "download_url", None),
        size_bytes=get_asset_size_bytes(asset, asset_md),
        variable_measured=extract_variable_measured(asset_md),
        measurement_technique=extract_measurement_technique(asset_md),
        selector=selector,
    )


def scan_dandiset_for_candidates(
    dandiset: Any,
    max_assets_per_dandiset: int,
    require_ephys_technique: bool,
    selector: str,
    verbose_skips: bool,
    debug_errors: bool,
    stats: ScanStats,
    variable_counter: Counter,
    skip_reason_counter: Counter,
    sample_skips: list[dict[str, Any]],
    max_skip_samples: int,
) -> list[tuple[Candidate, Any]]:
    candidates: list[tuple[Candidate, Any]] = []
    dandiset_ephys_match = False
    if require_ephys_technique:
        try:
            dandiset_ephys_match = metadata_mentions_ephys_technique(dandiset.get_metadata())
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

        is_match, variables, reason = variable_match(asset_md, selector=selector)
        variable_key = tuple(sorted(variables)) if variables else ("<missing variableMeasured>",)
        variable_counter[variable_key] += 1
        if not is_match:
            skip_reason_counter[reason] += 1
            if verbose_skips:
                print(f"[skip] {path} variableMeasured={variables} reason={reason}")
            if len(sample_skips) < max_skip_samples:
                sample_skips.append({"path": path, "variableMeasured": variables, "reason": reason})
            continue

        if require_ephys_technique:
            asset_ephys_match = metadata_mentions_ephys_technique(asset_md)
            if not (asset_ephys_match or dandiset_ephys_match):
                skip_reason_counter["missing target ephys measurementTechnique"] += 1
                if verbose_skips:
                    print(f"[skip ephys technique] {path} variableMeasured={variables}")
                continue

        candidate = build_candidate(dandiset, asset, asset_md, selector=selector)
        candidates.append((candidate, asset))
        stats.candidates_seen += 1
        print(
            f"[candidate] dandiset={candidate.dandiset_id} "
            f"version={candidate.version_id} size={candidate.size_bytes} "
            f"variables={candidate.variable_measured} path={candidate.asset_path}"
        )

    candidates.sort(key=lambda pair: pair[0].size_bytes if pair[0].size_bytes is not None else float("inf"))
    return candidates


def write_manifest(path: Path, records: list[dict[str, Any]], stats: ScanStats, variable_counter: Counter, skip_reason_counter: Counter, sample_skips: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selector": {
            "asset_extension": ".nwb",
            "selector_mode": args.selector,
            "target_exact_signature": sorted(TARGET_EXACT_SIGNATURE),
            "required_spike_variables_any_of": sorted(REQUIRED_SPIKE_VARIABLES),
            "allowed_lightweight_variables": sorted(ALLOWED_LIGHTWEIGHT_VARIABLES),
            "disallowed_heavy_variables": sorted(DISALLOWED_HEAVY_VARIABLES),
            "require_ephys_technique": bool(args.require_ephys_technique),
            "one_per_dandiset": bool(args.one_per_dandiset),
        },
        "scan_stats": asdict(stats),
        "variableMeasured_distribution_top_100": [
            {"variableMeasured": list(key), "count": count}
            for key, count in variable_counter.most_common(100)
        ],
        "skip_reason_distribution": [
            {"reason": reason, "count": count}
            for reason, count in skip_reason_counter.most_common(50)
        ],
        "sample_nonmatching_nwb_assets": sample_skips,
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DANDI lightweight spike/behavior NWB assets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--dandiset-id", default=None, help="Restrict to one Dandiset, e.g. 000006")
    parser.add_argument("--version", default=None, help="Dandiset version, e.g. draft or 0.220126.1855")
    parser.add_argument("--max-dandisets", type=int, default=500)
    parser.add_argument("--max-assets-per-dandiset", type=int, default=1000)
    parser.add_argument(
        "--selector",
        choices=["exact-target", "lightweight", "contains-units-no-heavy"],
        default="exact-target",
        help=(
            "exact-target = variableMeasured exactly matches BehavioralEvents/BehavioralTimeSeries/"
            "ElectrodeGroup/PropertyValue/Units. lightweight = any Units/SpikeEventSeries asset with only "
            "known lightweight variables. contains-units-no-heavy = any Units/SpikeEventSeries asset without known heavy variables."
        ),
    )
    parser.add_argument("--download-all", action="store_true", help="Download all matches. Without this, stop after the first match.")
    parser.add_argument("--one-per-dandiset", action="store_true", help="After sorting by size, download at most one matching NWB from each Dandiset.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--flat", action="store_true", help="Save files with flattened names instead of Dandiset/version/path tree.")
    parser.add_argument("--require-ephys-technique", action="store_true")
    parser.add_argument("--verbose-skips", action="store_true")
    parser.add_argument("--max-skip-samples", type=int, default=300)
    parser.add_argument("--debug-errors", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or (args.output_dir / DEFAULT_MANIFEST_NAME)

    print(f"[output dir] {args.output_dir}")
    print(f"[manifest]   {manifest_path}")
    print(f"[selector]   {args.selector}")
    print(f"[target exact signature] {sorted(TARGET_EXACT_SIGNATURE)}")
    print(f"[download mode] {'all matches' if args.download_all else 'first match only'}")
    if args.one_per_dandiset:
        print("[one per dandiset] enabled")
    if args.dry_run:
        print("[dry run] no files will be downloaded")

    stats = ScanStats()
    variable_counter: Counter = Counter()
    skip_reason_counter: Counter = Counter()
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
                    selector=args.selector,
                    verbose_skips=args.verbose_skips,
                    debug_errors=args.debug_errors,
                    stats=stats,
                    variable_counter=variable_counter,
                    skip_reason_counter=skip_reason_counter,
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
                        stats.downloaded += 1

                    records.append(record)
                    write_manifest(manifest_path, records, stats, variable_counter, skip_reason_counter, sample_skips, args)
                    print(f"[manifest updated] {manifest_path}")

                    if not args.download_all:
                        print("[done] stopped after first matching asset")
                        return 0
    finally:
        write_manifest(manifest_path, records, stats, variable_counter, skip_reason_counter, sample_skips, args)
        print(f"\n[manifest written] {manifest_path}")

    if not records:
        print(
            f"\n[not found] No matching NWB assets were downloaded/found. "
            f"Seen: dandisets={stats.dandisets_seen}, assets={stats.assets_seen}, "
            f"nwb_assets={stats.nwb_assets_seen}, metadata_errors={stats.metadata_read_errors}.",
            file=sys.stderr,
        )
        print(
            "The manifest still contains diagnostics. Try:\n"
            "  python fetch_dandi_light_spike_behavior_nwbs.py --selector lightweight --dry-run\n"
            "  python fetch_dandi_light_spike_behavior_nwbs.py --selector contains-units-no-heavy --dry-run\n",
            file=sys.stderr,
        )
        return 2

    print(f"\n[done] records={len(records)} downloaded={stats.downloaded} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
