"""Pydantic models for the DANDI chat agent.

These types replace the loose dataclasses in
``src/DANDI-agent/analysis/parse_mousehash_nwb_manifest.py`` and
``src/DANDI-agent/analysis/map_mousehash_manifest_to_analyses.py`` with
JSON-serializable models suitable for DataJoint payloads and LLM round-trips.

References:
- mousehash_parser_design.md (§3 status vocabulary, §4 role taxonomy, §6 confidence caps)
- mousehash_tools_spec.md (§3 ToolSpec schema, §4 readiness statuses)
- mousehash_transformations_spec.md (§4 TransformationSpec, §26 AnalysisView schema)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RoleStatus = Literal[
    "present",
    "likely_present",
    "absent",
    "unknown",
    "ambiguous",
    "derived_possible",
]

ReadinessStatus = Literal[
    "ready",
    "ready_after_transformation",
    "needs_confirmation",
    "blocked",
    "not_recommended",
]

TransformationReadiness = Literal[
    "available",
    "available_after_derivation",
    "needs_confirmation",
    "blocked",
    "unsafe_without_human_review",
]

EvidenceSource = Literal[
    "nwb",
    "dandiset_metadata",
    "asset_metadata",
    "paper_abstract",
    "llm_inference",
    "derived",
    "filename_hint",
]

ToolPriority = Literal["mvp", "soon", "experimental", "later"]


# Per mousehash_parser_design.md §6.
CONFIDENCE_CAPS: dict[str, float] = {
    "nwb_direct_path": 0.99,
    "nwb_neurodata_type": 0.97,
    "nwb_table_column": 0.95,
    "dandi_assets_summary": 0.85,
    "dandi_description": 0.75,
    "paper_abstract_llm": 0.75,
    "filename_hint": 0.45,
}


# Canonical role taxonomy from k-tuple.md / mousehash_parser_design.md §4.
ROLE_TAXONOMY: dict[str, Any] = {
    "neural_data": [
        "spikes",
        "lfp",
        "eeg",
        "calcium",
        "photometry",
        "images",
        "current_clamp",
        "voltage_clamp",
        "patch_clamp",
    ],
    "stimuli": {
        "sensory": ["visual", "auditory", "tactile", "odor"],
        "interventions": [
            "optogenetic",
            "electrical",
            "pharmacological",
            "anesthesia",
        ],
    },
    "behavior": [
        "choices",
        "reaction_times",
        "pose",
        "locomotion",
        "pupil",
        "kinematics",
        "behavioral_states",
        "running",
    ],
    "conditions": [
        "task_labels",
        "trial_labels",
        "experimental_groups",
        "brain_states",
        "session_phases",
        "perturbation_labels",
    ],
    "time_organization": [
        "continuous_time",
        "trials",
        "epochs",
        "events",
        "frames",
        "alignment_rules",
    ],
    "metadata": [
        "subject",
        "species",
        "genotype",
        "session",
        "brain_area",
        "probe_electrode_imaging_plane",
        "acquisition_device",
        "preprocessing_info",
    ],
}

TOP_LEVEL_ROLES: tuple[str, ...] = tuple(ROLE_TAXONOMY.keys())


class EvidenceItem(BaseModel):
    """A single piece of evidence supporting (or contradicting) a role claim.

    Promoted from the dataclass in
    ``DANDI-agent/analysis/parse_mousehash_nwb_manifest.py:80`` so it can be
    serialized into DataJoint manifest blobs and compared across runs.
    """

    model_config = ConfigDict(extra="forbid")

    role_path: str
    status: RoleStatus
    confidence: float = Field(ge=0.0, le=1.0)
    source: EvidenceSource
    field: str | None = None
    value: Any | None = None
    reason: str
    contradicts: list[str] = Field(default_factory=list)


class DerivationRecipe(BaseModel):
    """How a derived role can be computed from other roles.

    Examples include reaction_time = response_time - stimulus_onset_time
    (mousehash_parser_design.md §3.1).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    formula: str | None = None
    required_fields: list[str] = Field(default_factory=list)
    notes: str | None = None


class RoleEntry(BaseModel):
    """Status + evidence + derivation hint for one role path."""

    model_config = ConfigDict(extra="forbid")

    status: RoleStatus = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    derivation_recipe: DerivationRecipe | None = None
    needs_human_review: bool = False
    source_coverage: dict[str, int] = Field(default_factory=dict)


class EvidenceBackedRoleManifest(BaseModel):
    """The parser's typed output.

    Implements mousehash_parser_design.md §3 — a nested dict whose leaves are
    ``RoleEntry`` objects. The shape mirrors ``ROLE_TAXONOMY``; nested children
    (e.g. ``stimuli.sensory.visual``) are addressed by dotted role paths in
    each ``EvidenceItem.role_path``.
    """

    model_config = ConfigDict(extra="forbid")

    dandiset_id: str | None = None
    asset_id: str | None = None
    nwb_path: str | None = None
    parser_version: str
    catalog_version: str | None = None
    created_at: datetime
    roles: dict[str, RoleEntry] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    raw_summary: dict[str, Any] = Field(default_factory=dict)

    def get(self, role_path: str) -> RoleEntry:
        return self.roles.get(role_path, RoleEntry())

    def status(self, role_path: str) -> RoleStatus:
        return self.get(role_path).status


class TransformationSpec(BaseModel):
    """Catalog entry for a transformation.

    Schema follows mousehash_transformations_spec.md §4.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    family: str
    purpose: str
    requires_roles: list[str] = Field(default_factory=list)
    requires_inputs: list[str] = Field(default_factory=list)
    produces_view_or_role: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)
    provenance_required: list[str] = Field(default_factory=list)
    leakage_risk: Literal["none", "low", "medium", "high"] = "low"
    default_for_exploration: bool = False
    default_for_evaluation: bool = False


class AnalysisView(BaseModel):
    """Schema entry for an analysis-ready object.

    Follows mousehash_transformations_spec.md §26.
    """

    model_config = ConfigDict(extra="forbid")

    view_id: str
    view_type: str
    source_roles: list[str] = Field(default_factory=list)
    transformation_lineage: list[str] = Field(default_factory=list)
    axis_schema: dict[str, str] = Field(default_factory=dict)
    shape: list[int] | None = None
    units: dict[str, str] = Field(default_factory=dict)
    coordinate_system: str | None = None
    observation_identity: list[str] = Field(default_factory=list)
    condition_identity: list[str] = Field(default_factory=list)
    split_identity: str | None = None
    artifact_path: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class RoleRequirement(BaseModel):
    """Required-roles spec for a tool.

    ``all_of`` roles must all be present. ``any_of`` is a list of groups; each
    group is satisfied if at least one role inside it is present.
    """

    model_config = ConfigDict(extra="forbid")

    all_of: list[str] = Field(default_factory=list)
    any_of: list[list[str]] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """Catalog entry for a scientific tool.

    Schema follows mousehash_tools_spec.md §3.
    """

    model_config = ConfigDict(extra="forbid")

    tool_id: str
    name: str
    workflow_family: str
    requires_roles: RoleRequirement
    optional_roles: list[str] = Field(default_factory=list)
    target_subcategories: list[str] = Field(default_factory=list)
    requires_view: str
    default_transformation_path: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    parser_readiness_rules: list[str] = Field(default_factory=list)
    mvp_priority: ToolPriority = "later"
    leakage_risk: Literal["none", "low", "medium", "high"] = "low"
    has_visualization: bool = False
    question: str | None = None


class AnalysisMove(BaseModel):
    """RoleSignature + TransformationPlan + AnalysisView + Tool + ValidationPlan + Artifact.

    Formula from mousehash_transformations_spec.md §37.
    """

    model_config = ConfigDict(extra="forbid")

    move_id: str
    tool_id: str
    tool_name: str
    family: str
    role_signature: RoleRequirement
    optional_roles: list[str] = Field(default_factory=list)
    transformation_plan: list[str] = Field(default_factory=list)
    required_view: str
    validation_plan: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    question: str | None = None


class ToolReadinessReport(BaseModel):
    """Per-tool readiness verdict for a given manifest.

    Computed from ``compute_tool_readiness`` (mousehash_tools_spec.md §4).
    """

    model_config = ConfigDict(extra="forbid")

    tool_id: str
    tool_name: str
    status: ReadinessStatus
    score: float
    satisfied_roles: list[str] = Field(default_factory=list)
    uncertain_roles: list[str] = Field(default_factory=list)
    derivable_roles: list[str] = Field(default_factory=list)
    missing_roles: list[str] = Field(default_factory=list)
    optional_roles_present: list[str] = Field(default_factory=list)
    required_view: str
    suggested_transformations: list[str] = Field(default_factory=list)
    rationale: str | None = None
