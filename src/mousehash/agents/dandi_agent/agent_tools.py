"""Tool implementations for the DANDI chat agent.

These are plain Python functions wrapped with ``smolagents.tool`` in
``tools.py``. Splitting the logic here keeps the wrappers thin and the
testable surface plain.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mousehash.agents.dandi_agent.catalogs.loaders import (
    load_tools,
    load_transformations,
    tools_catalog_version,
)
from mousehash.agents.dandi_agent.models import EvidenceBackedRoleManifest
from mousehash.agents.dandi_agent.parser import parse_mousehash_roles
from mousehash.agents.dandi_agent.readiness import (
    build_analysis_move,
    rank_analysis_suggestions,
    suggest_analysis_moves,
)
from mousehash.config import DATA_ROOT

MANIFEST_DIR = DATA_ROOT / "dandi_agent" / "manifests"


def _ensure_manifest_dir() -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return MANIFEST_DIR


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_metadata(metadata_path: str | None) -> dict[str, Any] | None:
    if not metadata_path:
        return None
    return json.loads(Path(metadata_path).expanduser().read_text())


def _persist_manifest(manifest: EvidenceBackedRoleManifest) -> tuple[str, Path]:
    """Write the manifest JSON to disk and return ``(manifest_id, path)``."""
    out_dir = _ensure_manifest_dir()
    payload = manifest.model_dump_json(indent=2)
    manifest_id = _hash_text(payload)
    stem = f"{manifest.dandiset_id or 'unknown'}_{manifest_id[:12]}"
    out_path = out_dir / f"{stem}.manifest.json"
    out_path.write_text(payload)
    return manifest_id, out_path


def _register_manifest_row(
    manifest: EvidenceBackedRoleManifest,
    manifest_id: str,
    manifest_path: Path,
) -> None:
    """Best-effort DataJoint insert. Falls back silently if DB unavailable."""
    try:
        from mousehash.schema import dandi_agent as schema_module
    except Exception:
        return
    if not manifest.dandiset_id:
        return
    try:
        schema_module.Dandiset.insert1(
            dict(
                dandiset_id=manifest.dandiset_id,
                metadata_path=str(manifest_path),
                name="",
            ),
            skip_duplicates=True,
        )
        n_present = sum(1 for r in manifest.roles.values() if r.status == "present")
        n_likely = sum(1 for r in manifest.roles.values() if r.status == "likely_present")
        n_derived = sum(1 for r in manifest.roles.values() if r.status == "derived_possible")
        schema_module.RoleManifest.insert1(
            dict(
                manifest_id=manifest_id,
                dandiset_id=manifest.dandiset_id,
                asset_id=manifest.asset_id or "",
                parser_version=manifest.parser_version,
                catalog_version=manifest.catalog_version or "",
                manifest_path=str(manifest_path),
                n_present_roles=n_present,
                n_likely_present_roles=n_likely,
                n_derived_possible_roles=n_derived,
            ),
            skip_duplicates=True,
        )
    except Exception:
        # Persistence is auxiliary — never break the agent over it.
        pass


# ---------------------------------------------------------------------------
# Agent-callable tools
# ---------------------------------------------------------------------------

def inspect_dandiset(dandiset_id: str, metadata_path: str = "") -> str:
    """Summarize what is known about a DANDI dandiset before parsing NWB files.

    Args:
        dandiset_id: DANDI identifier, e.g. ``"000011"``.
        metadata_path: Optional path to a pre-fetched dandiset metadata JSON.
            If empty, this returns a placeholder pointing the user at
            ``parse_nwb_manifest`` instead.
    """
    metadata = _load_metadata(metadata_path or None)
    if metadata is None:
        return (
            f"No metadata path supplied for dandiset {dandiset_id}. "
            f"Use parse_nwb_manifest(dandiset_id, nwb_path=...) to build the "
            f"role manifest directly from a local NWB file."
        )
    keys = sorted(metadata.keys())
    summary = metadata.get("assetsSummary", {}) or {}
    return json.dumps(
        {
            "dandiset_id": dandiset_id,
            "metadata_path": metadata_path,
            "name": metadata.get("name", ""),
            "n_subjects": summary.get("numberOfSubjects"),
            "n_files": summary.get("numberOfFiles"),
            "variable_measured": summary.get("variableMeasured", []),
            "measurement_technique": summary.get("measurementTechnique", []),
            "approach": summary.get("approach", []),
            "top_level_keys": keys,
        },
        indent=2,
        default=str,
    )


def parse_nwb_manifest(
    dandiset_id: str,
    nwb_path: str,
    asset_id: str = "",
    metadata_path: str = "",
) -> str:
    """Build the EvidenceBackedRoleManifest for a dandiset/NWB pair.

    Persists the manifest JSON under ``DATA_ROOT/dandi_agent/manifests/`` and
    inserts a ``RoleManifest`` DataJoint row when the DB is reachable.

    Args:
        dandiset_id: DANDI identifier.
        nwb_path: Absolute path to a local NWB file.
        asset_id: Optional DANDI asset identifier for the NWB.
        metadata_path: Optional path to dandiset metadata JSON.

    Returns:
        A JSON summary with manifest_id, role counts, and the on-disk path.
    """
    metadata = _load_metadata(metadata_path or None)
    manifest = parse_mousehash_roles(
        dandiset_id=dandiset_id,
        dandiset_metadata=metadata,
        nwb_path=nwb_path,
        asset_id=asset_id or None,
        catalog_version=tools_catalog_version(),
    )
    manifest_id, manifest_path = _persist_manifest(manifest)
    _register_manifest_row(manifest, manifest_id, manifest_path)

    return json.dumps(
        {
            "dandiset_id": dandiset_id,
            "asset_id": asset_id or None,
            "manifest_id": manifest_id,
            "manifest_path": str(manifest_path),
            "n_role_paths": len(manifest.roles),
            "n_present": sum(1 for r in manifest.roles.values() if r.status == "present"),
            "n_likely_present": sum(1 for r in manifest.roles.values() if r.status == "likely_present"),
            "n_derived_possible": sum(1 for r in manifest.roles.values() if r.status == "derived_possible"),
            "warnings": manifest.warnings,
        },
        indent=2,
    )


def show_role_manifest(manifest_path: str) -> str:
    """Pretty-print a persisted role manifest.

    Args:
        manifest_path: Path returned by ``parse_nwb_manifest``.
    """
    manifest = EvidenceBackedRoleManifest.model_validate_json(
        Path(manifest_path).read_text()
    )
    lines = [
        f"# MouseHash role manifest",
        f"dandiset: {manifest.dandiset_id}  asset: {manifest.asset_id}",
        f"parser: {manifest.parser_version}  catalog: {manifest.catalog_version}",
        f"created_at: {manifest.created_at.isoformat()}",
        "",
        "## Roles",
    ]
    for path in sorted(manifest.roles.keys()):
        entry = manifest.roles[path]
        lines.append(
            f"  {path:50s}  {entry.status:18s}  conf={entry.confidence:.2f}  "
            f"n_evidence={len(entry.evidence)}"
        )
    if manifest.warnings:
        lines.append("")
        lines.append("## Warnings")
        for w in manifest.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def suggest_analyses(manifest_path: str, top_k: int = 10) -> str:
    """Return ranked AnalysisMoves for a persisted manifest.

    Args:
        manifest_path: Path returned by ``parse_nwb_manifest``.
        top_k: Max number of suggestions to return.
    """
    manifest = EvidenceBackedRoleManifest.model_validate_json(
        Path(manifest_path).read_text()
    )
    pairs = suggest_analysis_moves(manifest, top_k=top_k)
    out: list[dict[str, Any]] = []
    for report, move in pairs:
        out.append(
            {
                "tool_id": move.tool_id,
                "tool_name": move.tool_name,
                "status": report.status,
                "score": report.score,
                "question": move.question,
                "required_view": move.required_view,
                "transformation_plan": move.transformation_plan,
                "validation_plan": move.validation_plan,
                "artifacts": move.artifacts,
                "satisfied_roles": report.satisfied_roles,
                "uncertain_roles": report.uncertain_roles,
                "derivable_roles": report.derivable_roles,
                "missing_roles": report.missing_roles,
                "rationale": report.rationale,
            }
        )
    return json.dumps({"manifest_path": manifest_path, "suggestions": out}, indent=2)


def propose_transformation_plan(manifest_path: str, tool_id: str) -> str:
    """Return the full transformation chain + assumptions + failure modes for one tool.

    Args:
        manifest_path: Path returned by ``parse_nwb_manifest``.
        tool_id: A tool_id from the catalog (use ``suggest_analyses`` to discover them).
    """
    manifest = EvidenceBackedRoleManifest.model_validate_json(
        Path(manifest_path).read_text()
    )
    tools = load_tools()
    if tool_id not in tools:
        return json.dumps({"error": f"unknown tool_id {tool_id!r}", "available": sorted(tools)})

    tool = tools[tool_id]
    move = build_analysis_move(manifest, tool, load_transformations())
    tx_catalog = load_transformations()
    enriched_plan = []
    for name in move.transformation_plan:
        spec = tx_catalog.get(name)
        enriched_plan.append(
            {
                "name": name,
                "family": spec.family if spec else "unknown",
                "purpose": spec.purpose if spec else "(not catalogued)",
                "leakage_risk": spec.leakage_risk if spec else None,
                "parameters": dict(spec.parameters) if spec else {},
            }
        )
    return json.dumps(
        {
            "move_id": move.move_id,
            "tool_name": move.tool_name,
            "required_view": move.required_view,
            "role_signature": move.role_signature.model_dump(),
            "transformation_plan": enriched_plan,
            "assumptions": move.assumptions,
            "failure_modes": move.failure_modes,
            "validation_plan": move.validation_plan,
            "artifacts": move.artifacts,
            "question": move.question,
        },
        indent=2,
    )


def explain_blocked_tools(manifest_path: str) -> str:
    """List blocked tools with the specific roles they need and how to unblock them.

    Args:
        manifest_path: Path returned by ``parse_nwb_manifest``.
    """
    manifest = EvidenceBackedRoleManifest.model_validate_json(
        Path(manifest_path).read_text()
    )
    reports = rank_analysis_suggestions(manifest)
    blocked = [r for r in reports if r.status == "blocked"]
    out: list[dict[str, Any]] = []
    for r in blocked:
        out.append(
            {
                "tool_id": r.tool_id,
                "tool_name": r.tool_name,
                "missing_roles": r.missing_roles,
                "uncertain_roles": r.uncertain_roles,
                "rationale": r.rationale,
            }
        )
    return json.dumps({"manifest_path": manifest_path, "blocked_tools": out}, indent=2)


def list_paper_evidence(manifest_path: str) -> str:
    """Stub for MVP 1. Paper/DOI enrichment lands in MVP 3.

    Args:
        manifest_path: Path returned by ``parse_nwb_manifest``.
    """
    return json.dumps(
        {
            "manifest_path": manifest_path,
            "paper_evidence": [],
            "note": "Paper / DOI / abstract enrichment is scheduled for MVP 3.",
        }
    )
