"""Parser orchestrator: DANDI metadata + NWB structure -> EvidenceBackedRoleManifest.

Implements the pseudocode from mousehash_parser_design.md §5. Reuses the existing
extractors in ``src/DANDI-agent/`` rather than duplicating their pattern logic:

- ``dandi_crawl.apply_rules`` for dandiset-metadata rules (§7).
- ``parse_mousehash_nwb_manifest.ManifestParser`` for NWB structural evidence (§8-14).

Both legacy extractors emit untyped dict / dataclass shapes; this module projects
them into the typed Pydantic model and performs merging, consistency checks, and
derived-role inference (§18-§20).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mousehash.agents.dandi_agent.models import (
    CONFIDENCE_CAPS,
    DerivationRecipe,
    EvidenceBackedRoleManifest,
    EvidenceItem,
    EvidenceSource,
    RoleEntry,
    RoleStatus,
    TOP_LEVEL_ROLES,
)

PARSER_VERSION = "0.1.0"

# Make the legacy extractors importable. They live outside the mousehash
# package so we extend sys.path lazily on first use.
_LEGACY_ROOT = Path(__file__).resolve().parents[3] / "DANDI-agent"
_LEGACY_ANALYSIS = _LEGACY_ROOT / "analysis"
for _p in (_LEGACY_ROOT, _LEGACY_ANALYSIS):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# Status precedence for merging (mousehash_parser_design.md §19).
_STATUS_RANK: dict[RoleStatus, int] = {
    "unknown": 0,
    "absent": 1,
    "derived_possible": 2,
    "ambiguous": 3,
    "likely_present": 4,
    "present": 5,
}

# Coarse "high/medium/low" -> float mapping used by dandi_crawl.apply_rules.
_LEGACY_CONFIDENCE: dict[str, float] = {
    "high": CONFIDENCE_CAPS["dandi_assets_summary"],
    "medium": CONFIDENCE_CAPS["dandi_description"],
    "low": 0.55,
}

# dandi_crawl uses a smaller status vocabulary ("likely", "present", "unknown").
# Map it into the typed ``RoleStatus`` Literal.
_LEGACY_STATUS: dict[str, RoleStatus] = {
    "present": "present",
    "likely": "likely_present",
    "likely_present": "likely_present",
    "absent": "absent",
    "unknown": "unknown",
}

# Legacy (role, label) pairs from parse_mousehash_nwb_manifest into the
# canonical taxonomy paths. Labels not listed pass through unchanged.
_LABEL_NORMALIZATION: dict[str, str] = {
    "neural_data.lfp_or_raw_ephys_present": "neural_data.lfp",
    "neural_data.spikes": "neural_data.spikes",
    "stimuli.tactile": "stimuli.sensory.tactile",
    "stimuli.optogenetic_intervention": "stimuli.interventions.optogenetic",
    "behavior.reaction_times_or_movement_timing": "behavior.reaction_times",
    "behavior.behavioral_events_or_timeseries": "behavior.behavioral_states",
    "behavior.choices": "behavior.choices",
    "conditions.trial_labels": "conditions.trial_labels",
    "conditions.session_phases": "conditions.session_phases",
    "conditions.perturbation_labels": "conditions.perturbation_labels",
    "time_organization.timestamps": "time_organization.continuous_time",
    "time_organization.trials": "time_organization.trials",
    "time_organization.epochs": "time_organization.epochs",
    "time_organization.epochs_or_intervals": "time_organization.epochs",
    "time_organization.events": "time_organization.events",
    "metadata.subject": "metadata.subject",
    "metadata.brain_area_ALM": "metadata.brain_area",
    "metadata.hemisphere": "metadata.brain_area",
    "metadata.electrode_group": "metadata.probe_electrode_imaging_plane",
    "metadata.recording_device_or_electrodes": "metadata.acquisition_device",
    "metadata.photoinhibition_or_optogenetic_protocol": "stimuli.interventions.optogenetic",
}

# Roles to drop from the legacy parser (book-keeping, not canonical roles).
_LEGACY_DROP_ROLES: set[str] = {"nwb_structure"}


def _normalize_role_path(legacy_role: str, legacy_label: str) -> str | None:
    if legacy_role in _LEGACY_DROP_ROLES:
        return None
    raw = f"{legacy_role}.{legacy_label}"
    return _LABEL_NORMALIZATION.get(raw, raw)


def _classify_nwb_source(source_path: str) -> EvidenceSource:
    """Treat HDF5/NWB paths as ``nwb``; anything else as ``derived``."""
    if isinstance(source_path, str) and source_path.startswith("/"):
        return "nwb"
    return "derived"


def _nwb_confidence(source_path: str) -> float:
    """Heuristic confidence cap from mousehash_parser_design.md §6."""
    if "/units" in source_path or "/spike_times" in source_path:
        return CONFIDENCE_CAPS["nwb_direct_path"]
    if "/intervals/trials/" in source_path:
        return CONFIDENCE_CAPS["nwb_table_column"]
    return CONFIDENCE_CAPS["nwb_neurodata_type"]


# ---------------------------------------------------------------------------
# NWB structural parser (wraps the legacy ManifestParser)
# ---------------------------------------------------------------------------

def parse_nwb_file(nwb_path: str) -> tuple[list[EvidenceItem], dict[str, Any]]:
    """Run the legacy ``ManifestParser`` and convert its evidence into typed form.

    Returns a list of :class:`EvidenceItem` plus the legacy raw summary dict
    (preserved verbatim under ``EvidenceBackedRoleManifest.raw_summary``).
    """
    from parse_mousehash_nwb_manifest import ManifestParser  # legacy module

    parser = ManifestParser(nwb_path)
    raw_manifest = parser.parse(use_pynwb=True)

    items: list[EvidenceItem] = []
    seen: set[tuple[str, str, str]] = set()
    for legacy in parser.evidence:
        role_path = _normalize_role_path(legacy.role, legacy.label)
        if role_path is None:
            continue
        source = _classify_nwb_source(legacy.source)
        key = (role_path, legacy.source, legacy.reason)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            EvidenceItem(
                role_path=role_path,
                status="present" if source == "nwb" else "likely_present",
                confidence=_nwb_confidence(legacy.source),
                source=source,
                field=legacy.source,
                value=legacy.label,
                reason=legacy.reason,
            )
        )
    return items, raw_manifest


# ---------------------------------------------------------------------------
# DANDI metadata parser (wraps dandi_crawl.apply_rules)
# ---------------------------------------------------------------------------

def parse_dandiset_metadata(metadata: dict[str, Any]) -> list[EvidenceItem]:
    """Convert ``dandi_crawl.apply_rules`` output into typed evidence items."""
    from dandi_crawl import apply_rules  # legacy module

    manifest_dict, evidence_df = apply_rules(metadata)
    items: list[EvidenceItem] = []
    for _, row in evidence_df.iterrows():
        role_path = _LABEL_NORMALIZATION.get(row["role"], row["role"])
        conf = _LEGACY_CONFIDENCE.get(row["confidence"], 0.55)
        status = _LEGACY_STATUS.get(row["status"], "likely_present")
        items.append(
            EvidenceItem(
                role_path=role_path,
                status=status,
                confidence=conf,
                source="dandiset_metadata",
                field=row["source"],
                value=str(row.get("value", ""))[:240],
                reason=f"matched pattern {row['pattern']!r}",
            )
        )
    return items


# ---------------------------------------------------------------------------
# Derived-role inference (§18)
# ---------------------------------------------------------------------------

def infer_derived_roles_from_combinations(
    role_paths_present: Iterable[str],
    trial_columns: Iterable[str],
) -> list[EvidenceItem]:
    """Mint ``derived_possible`` evidence from role co-occurrence patterns."""
    present = set(role_paths_present)
    cols = {c.lower() for c in trial_columns}
    derived: list[EvidenceItem] = []

    has_onset = any("stimulus_onset" in c or "start_time" in c for c in cols)
    has_resp = any("response_time" in c or "reaction_time" in c for c in cols)
    if has_onset and has_resp and "behavior.reaction_times" not in present:
        derived.append(
            EvidenceItem(
                role_path="behavior.reaction_times",
                status="derived_possible",
                confidence=0.85,
                source="derived",
                reason="trials table has stimulus_onset and response_time columns",
            )
        )

    if (
        "stimuli.interventions.optogenetic" in present
        and "time_organization.trials" in present
        and "conditions.perturbation_labels" not in present
    ):
        derived.append(
            EvidenceItem(
                role_path="conditions.perturbation_labels",
                status="derived_possible",
                confidence=0.8,
                source="derived",
                reason="optogenetic stimulus + trials present; labels derivable by intersection",
            )
        )

    if (
        "time_organization.events" in present
        and "time_organization.continuous_time" in present
        and "time_organization.alignment_rules" not in present
    ):
        derived.append(
            EvidenceItem(
                role_path="time_organization.alignment_rules",
                status="derived_possible",
                confidence=0.8,
                source="derived",
                reason="events + continuous time present; alignment is derivable",
            )
        )
    return derived


# ---------------------------------------------------------------------------
# Evidence merger (§19) and consistency checks (§20)
# ---------------------------------------------------------------------------

def _pick_status(evidence: list[EvidenceItem]) -> tuple[RoleStatus, float]:
    if not evidence:
        return "unknown", 0.0

    by_status: dict[RoleStatus, list[EvidenceItem]] = {}
    for e in evidence:
        by_status.setdefault(e.status, []).append(e)

    # Prefer the highest-ranked status that has at least one piece of evidence.
    for status in ("present", "likely_present", "ambiguous", "derived_possible", "absent"):
        items = by_status.get(status)  # type: ignore[arg-type]
        if items:
            conf = max(e.confidence for e in items)
            return status, conf  # type: ignore[return-value]
    return "unknown", 0.0


def merge_evidence_into_manifest(
    evidence: list[EvidenceItem],
    parser_version: str = PARSER_VERSION,
    catalog_version: str | None = None,
    dandiset_id: str | None = None,
    asset_id: str | None = None,
    nwb_path: str | None = None,
    raw_summary: dict[str, Any] | None = None,
) -> EvidenceBackedRoleManifest:
    """Aggregate per role_path into a typed manifest."""
    by_path: dict[str, list[EvidenceItem]] = {}
    for item in evidence:
        by_path.setdefault(item.role_path, []).append(item)

    roles: dict[str, RoleEntry] = {}
    for path, items in by_path.items():
        status, conf = _pick_status(items)
        coverage: dict[str, int] = {}
        for it in items:
            coverage[it.source] = coverage.get(it.source, 0) + 1
        roles[path] = RoleEntry(
            status=status,
            confidence=conf,
            evidence=items[:25],
            source_coverage=coverage,
            needs_human_review=(status in ("likely_present", "ambiguous", "derived_possible"))
            and not any(it.source == "nwb" for it in items),
        )

    manifest = EvidenceBackedRoleManifest(
        dandiset_id=dandiset_id,
        asset_id=asset_id,
        nwb_path=nwb_path,
        parser_version=parser_version,
        catalog_version=catalog_version,
        created_at=datetime.now(timezone.utc),
        roles=roles,
        raw_summary=raw_summary or {},
    )
    run_consistency_checks(manifest)
    propagate_top_level_roles(manifest)
    return manifest


def propagate_top_level_roles(manifest: EvidenceBackedRoleManifest) -> None:
    """If any sub-role is present, the top-level role is at least likely_present.

    This lets the readiness engine answer simple checks like
    ``manifest.status("neural_data")`` without enumerating leaves.
    """
    for top in TOP_LEVEL_ROLES:
        leaf_statuses = [
            entry.status
            for path, entry in manifest.roles.items()
            if path == top or path.startswith(top + ".")
        ]
        if not leaf_statuses:
            continue
        best = max(leaf_statuses, key=lambda s: _STATUS_RANK[s])
        if best in ("present", "likely_present", "derived_possible"):
            existing = manifest.roles.get(top)
            if existing is None or _STATUS_RANK[existing.status] < _STATUS_RANK[best]:
                manifest.roles[top] = RoleEntry(
                    status=best,
                    confidence=max(
                        (
                            e.confidence
                            for p, e in manifest.roles.items()
                            if p == top or p.startswith(top + ".")
                        ),
                        default=0.0,
                    ),
                )


def run_consistency_checks(manifest: EvidenceBackedRoleManifest) -> None:
    """Apply the consistency rules from mousehash_parser_design.md §20."""
    s = manifest.status

    if s("neural_data.spikes") == "present" and s("time_organization.continuous_time") == "unknown":
        manifest.warnings.append(
            "Spike times imply continuous time; check timestamp extraction."
        )
    if s("neural_data.calcium") == "present" and s("time_organization.frames") == "unknown":
        manifest.warnings.append(
            "Calcium imaging usually implies frames; inspect imaging objects."
        )
    if s("behavior.reaction_times") in ("present", "derived_possible") and s("time_organization.trials") == "unknown":
        manifest.warnings.append(
            "Reaction times without trial structure may not be interpretable."
        )
    if (
        s("stimuli.interventions.optogenetic") == "present"
        and s("conditions.perturbation_labels") == "unknown"
    ):
        manifest.warnings.append(
            "Perturbation labels may be derivable by intersecting stimulation times with trials."
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def parse_mousehash_roles(
    *,
    dandiset_id: str | None = None,
    dandiset_metadata: dict[str, Any] | None = None,
    nwb_path: str | None = None,
    asset_id: str | None = None,
    catalog_version: str | None = None,
) -> EvidenceBackedRoleManifest:
    """Top-level entry point mirroring mousehash_parser_design.md §5.

    At least one of ``dandiset_metadata`` or ``nwb_path`` should be supplied.
    Fetching dandiset metadata from the DANDI API is intentionally out of
    scope here — pass it in pre-fetched so the parser stays offline-testable.
    """
    evidence: list[EvidenceItem] = []
    raw_summary: dict[str, Any] = {}

    if dandiset_metadata is not None:
        evidence.extend(parse_dandiset_metadata(dandiset_metadata))
        raw_summary["dandiset_metadata_keys"] = sorted(dandiset_metadata.keys())

    trial_columns: list[str] = []
    if nwb_path is not None:
        nwb_items, raw_nwb = parse_nwb_file(nwb_path)
        evidence.extend(nwb_items)
        raw_summary["nwb"] = {
            "session_metadata": raw_nwb.get("nwb_summary", {}).get("session_metadata", {}),
            "trial_columns": raw_nwb.get("nwb_summary", {}).get("trial_columns", []),
            "units": raw_nwb.get("nwb_summary", {}).get("units", {}),
            "intervals": raw_nwb.get("nwb_summary", {}).get("intervals", []),
        }
        trial_columns = raw_nwb.get("nwb_summary", {}).get("trial_columns", []) or []

    present_paths = {e.role_path for e in evidence if e.status in ("present", "likely_present")}
    evidence.extend(infer_derived_roles_from_combinations(present_paths, trial_columns))

    return merge_evidence_into_manifest(
        evidence,
        parser_version=PARSER_VERSION,
        catalog_version=catalog_version,
        dandiset_id=dandiset_id,
        asset_id=asset_id,
        nwb_path=nwb_path,
        raw_summary=raw_summary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build an EvidenceBackedRoleManifest from a DANDI metadata JSON and/or local NWB.",
    )
    ap.add_argument("--dandiset-metadata", type=Path, help="Path to a dandiset metadata JSON file.")
    ap.add_argument("--nwb", type=Path, help="Path to a local NWB file.")
    ap.add_argument("--dandiset-id", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path (defaults to stdout).")
    args = ap.parse_args()

    metadata = None
    if args.dandiset_metadata:
        metadata = json.loads(args.dandiset_metadata.read_text())

    manifest = parse_mousehash_roles(
        dandiset_id=args.dandiset_id,
        dandiset_metadata=metadata,
        nwb_path=str(args.nwb) if args.nwb else None,
    )
    payload = manifest.model_dump_json(indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(f"[saved] {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
