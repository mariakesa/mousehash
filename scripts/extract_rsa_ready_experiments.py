"""Extract DANDI manifests that are analyzable with RSA.

Edit the constants below before running:

    JSON_INDICES: directory containing ``*.manifest.json`` files.
    OUTPUT_PATH: JSON file that will store RSA-ready experiment summaries.

Examples:
    python scripts/extract_rsa_ready_experiments.py
    python scripts/extract_rsa_ready_experiments.py --max-manifests 25
    python scripts/extract_rsa_ready_experiments.py --output-path /tmp/rsa_ready.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Update these paths for your environment.
JSON_INDICES = Path("/media/maria/notsudata/MousehashManifests")
OUTPUT_PATH = JSON_INDICES / "rsa_ready_experiments.json"

RSA_TOOL_ID = "run_rsa"
DEFAULT_STATUSES = ("ready", "ready_after_transformation", "needs_confirmation")

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

from mousehash.agents.dandi_agent.catalogs.loaders import load_tools
from mousehash.agents.dandi_agent.models import EvidenceBackedRoleManifest, ReadinessStatus
from mousehash.agents.dandi_agent.readiness import build_analysis_move, compute_tool_readiness


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_index_records(json_indices: Path) -> dict[str, dict[str, Any]]:
    summary_path = json_indices / "index_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = payload.get("records", []) or []
    return {
        str(record.get("manifest_path")): record
        for record in records
        if record.get("manifest_path")
    }


def _iter_manifest_paths(json_indices: Path) -> list[Path]:
    return sorted(
        path for path in json_indices.glob("*.manifest.json")
        if path.is_file()
    )


def _evaluate_manifest(
    manifest_path: Path,
    index_record: dict[str, Any] | None,
    allowed_statuses: set[ReadinessStatus],
) -> dict[str, Any] | None:
    manifest = EvidenceBackedRoleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    rsa_tool = load_tools()[RSA_TOOL_ID]
    report = compute_tool_readiness(manifest, rsa_tool)
    if report.status not in allowed_statuses:
        return None

    move = build_analysis_move(manifest, rsa_tool)
    return {
        "dandiset_id": manifest.dandiset_id,
        "asset_id": manifest.asset_id,
        "manifest_path": str(manifest_path),
        "nwb_path": manifest.nwb_path,
        "status": report.status,
        "score": report.score,
        "tool_id": report.tool_id,
        "tool_name": report.tool_name,
        "required_view": report.required_view,
        "suggested_transformations": report.suggested_transformations,
        "satisfied_roles": report.satisfied_roles,
        "uncertain_roles": report.uncertain_roles,
        "derivable_roles": report.derivable_roles,
        "missing_roles": report.missing_roles,
        "optional_roles_present": report.optional_roles_present,
        "rationale": report.rationale,
        "validation_plan": move.validation_plan,
        "artifacts": move.artifacts,
        "assumptions": move.assumptions,
        "warnings": manifest.warnings,
        "index_record": index_record or {},
    }


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-indices", type=Path, default=JSON_INDICES)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--statuses",
        nargs="+",
        choices=["ready", "ready_after_transformation", "needs_confirmation", "blocked", "not_recommended"],
        default=list(DEFAULT_STATUSES),
        help="Readiness statuses to keep in the output.",
    )
    parser.add_argument("--max-manifests", type=int, default=None)
    args = parser.parse_args()

    manifest_paths = _iter_manifest_paths(args.json_indices)
    if args.max_manifests is not None:
        manifest_paths = manifest_paths[: args.max_manifests]

    index_records = _load_index_records(args.json_indices)
    allowed_statuses = set(args.statuses)
    matches: list[dict[str, Any]] = []
    skipped = 0
    errors: list[dict[str, Any]] = []

    for manifest_path in manifest_paths:
        try:
            result = _evaluate_manifest(
                manifest_path,
                index_records.get(str(manifest_path)),
                allowed_statuses,
            )
            if result is None:
                skipped += 1
                continue
            matches.append(result)
        except Exception as exc:
            logger.exception("Failed to evaluate %s", manifest_path)
            errors.append({"manifest_path": str(manifest_path), "error": str(exc)})

    matches.sort(key=lambda item: item["score"], reverse=True)
    status_counts = Counter(match["status"] for match in matches)

    payload = {
        "json_indices": str(args.json_indices),
        "tool_id": RSA_TOOL_ID,
        "statuses_included": sorted(allowed_statuses),
        "n_manifests_scanned": len(manifest_paths),
        "n_matches": len(matches),
        "n_skipped": skipped,
        "n_errors": len(errors),
        "status_counts": dict(status_counts),
        "matches": matches,
        "errors": errors,
    }
    _write_output(args.output_path, payload)

    logger.info("Scanned %d manifests", len(manifest_paths))
    logger.info("Found %d RSA-eligible experiments", len(matches))
    logger.info("Wrote results to %s", args.output_path)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())