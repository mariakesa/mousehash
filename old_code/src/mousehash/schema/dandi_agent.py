"""DataJoint schema for the MouseHash DANDI chat agent.

Tables (per the approved plan in ``can-you-plan-out-robust-quill.md``):

    Dandiset (M) ── NWBAsset (M) ── RoleManifest (M, computed-in-Python)
                                       │
                                       ▼
                                   AnalysisView (M)
                                       │
              TransformationSpec (L)── │   ToolSpec (L)
                       │               │      │
                       ▼               ▼      ▼
              TransformationRun (M)   ToolRun (M) ── AnalysisArtifact (M)

All tables use ``dj.Manual`` rather than ``dj.Computed`` so the agent can
write rows directly without owning a DataJoint pipeline. Lookup tables hold
the catalog rows (one row per catalog_version + name).
"""

from __future__ import annotations

import datajoint as dj

from mousehash.config import DB_PREFIX

schema = dj.Schema(f"{DB_PREFIX}_dandi_agent")


@schema
class Dandiset(dj.Manual):
    """A DANDI dandiset the agent has been asked about."""

    definition = """
    dandiset_id: varchar(16)
    ---
    metadata_path: varchar(1024)
    name="": varchar(512)
    created_at=CURRENT_TIMESTAMP: timestamp
    """


@schema
class NWBAsset(dj.Manual):
    """A specific NWB file within a dandiset."""

    definition = """
    -> Dandiset
    asset_id: varchar(64)
    ---
    local_path: varchar(1024)
    asset_size_bytes=0: bigint
    created_at=CURRENT_TIMESTAMP: timestamp
    """


@schema
class RoleManifest(dj.Manual):
    """An EvidenceBackedRoleManifest persisted by the parser."""

    definition = """
    manifest_id: char(64)         # sha256 of the manifest JSON
    ---
    -> Dandiset
    asset_id="": varchar(64)
    parser_version: varchar(32)
    catalog_version="": varchar(32)
    manifest_path: varchar(1024)  # on-disk JSON
    n_present_roles=0: int
    n_likely_present_roles=0: int
    n_derived_possible_roles=0: int
    created_at=CURRENT_TIMESTAMP: timestamp
    """


@schema
class TransformationSpec(dj.Lookup):
    """Catalog row for a transformation (one per name + catalog_version)."""

    definition = """
    catalog_version: varchar(32)
    transformation_name: varchar(128)
    ---
    family: varchar(64)
    purpose: varchar(512)
    spec_json: longblob
    leakage_risk: varchar(16)
    """


@schema
class ToolSpec(dj.Lookup):
    """Catalog row for a scientific tool (one per tool_id + catalog_version)."""

    definition = """
    catalog_version: varchar(32)
    tool_id: varchar(64)
    ---
    name: varchar(128)
    workflow_family: varchar(64)
    requires_view: varchar(256)
    spec_json: longblob
    mvp_priority: varchar(16)
    """


@schema
class AnalysisView(dj.Manual):
    """An analysis-ready object materialized from transformations."""

    definition = """
    view_id: char(64)             # sha256 of (manifest_id + view_type + transformation_lineage)
    ---
    -> RoleManifest
    view_type: varchar(64)
    artifact_path: varchar(1024)
    schema_json: longblob
    created_at=CURRENT_TIMESTAMP: timestamp
    """


@schema
class TransformationRun(dj.Manual):
    """One execution of a transformation, with parameter + input hashes."""

    definition = """
    run_id: char(64)
    ---
    -> TransformationSpec
    -> RoleManifest
    params_hash: char(64)
    input_hash="": char(64)
    output_path: varchar(1024)
    status: varchar(32)
    code_version="": varchar(64)
    created_at=CURRENT_TIMESTAMP: timestamp
    """


@schema
class ToolRun(dj.Manual):
    """One execution of a scientific tool against an AnalysisView."""

    definition = """
    tool_run_id: char(64)
    ---
    -> ToolSpec
    -> AnalysisView
    tool_run_spec_id="": varchar(64)   # FK reference into mousehash_blahml.ToolRunSpec
    params_hash: char(64)
    artifact_path: varchar(1024)
    status: varchar(32)
    code_version="": varchar(64)
    created_at=CURRENT_TIMESTAMP: timestamp
    """


@schema
class AnalysisArtifact(dj.Manual):
    """Typed artifact bundle produced by a ToolRun: model, metrics, plots, or report."""

    definition = """
    artifact_id: char(64)
    ---
    -> ToolRun
    artifact_type: varchar(64)
    artifact_path: varchar(1024)
    summary_json=NULL: longblob
    created_at=CURRENT_TIMESTAMP: timestamp
    """


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------

def register_catalog() -> None:
    """Upsert the on-disk YAML catalogs into the Lookup tables.

    Idempotent: re-running with the same ``catalog_version`` is a no-op for
    the rows it has already inserted.
    """
    from mousehash.agents.dandi_agent.catalogs.loaders import (
        load_tools,
        load_transformations,
        tools_catalog_version,
        transformations_catalog_version,
    )

    tx_version = transformations_catalog_version()
    for name, spec in load_transformations().items():
        TransformationSpec.insert1(
            dict(
                catalog_version=tx_version,
                transformation_name=name,
                family=spec.family,
                purpose=spec.purpose[:512],
                spec_json=spec.model_dump_json().encode("utf-8"),
                leakage_risk=spec.leakage_risk,
            ),
            skip_duplicates=True,
        )

    tools_version = tools_catalog_version()
    for tool_id, spec in load_tools().items():
        ToolSpec.insert1(
            dict(
                catalog_version=tools_version,
                tool_id=tool_id,
                name=spec.name,
                workflow_family=spec.workflow_family,
                requires_view=spec.requires_view[:256],
                spec_json=spec.model_dump_json().encode("utf-8"),
                mvp_priority=spec.mvp_priority,
            ),
            skip_duplicates=True,
        )
