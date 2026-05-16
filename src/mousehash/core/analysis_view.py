"""AnalysisView: explicit, lineage-tracked input to a tool.

Tools consume `AnalysisView` objects, not raw target data. A view carries:

  - its kind (e.g. observation_by_neuron, RDM, trial_time_neuron),
  - its shape and named axes,
  - the source roles it was derived from,
  - the ordered list of transformations that produced it,
  - a stable lineage hash for content addressing.

This makes statements like "PCA was run on a stimulus-averaged response matrix,
not on the raw trial tensor" recoverable from provenance alone.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from mousehash.core.ids import ManifestId, ViewId, stable_hash
from mousehash.core.role_bundle import RoleName


class AnalysisViewKind(str, Enum):
    OBSERVATION_BY_NEURON = "observation_by_neuron"
    OBSERVATION_BY_FEATURE = "observation_by_feature"
    TRIAL_TIME_NEURON = "trial_time_neuron"
    CONDITION_TIME_NEURON = "condition_time_neuron"
    DESIGN_MATRIX = "design_matrix"
    LAGGED_DESIGN_MATRIX = "lagged_design_matrix"
    RDM = "rdm"
    LATENT_TRAJECTORY = "latent_trajectory"
    FUNCTIONAL_GRAPH = "functional_graph"
    METRIC_TABLE = "metric_table"
    REPORT_BUNDLE = "report_bundle"
    PRESENTATION_TABLE = "presentation_table"


class AnalysisView(BaseModel):
    """A typed, lineage-tracked input for downstream tools."""

    view_id: ViewId
    kind: AnalysisViewKind
    manifest_id: ManifestId
    shape: list[int] = Field(description="Materialized array shape, e.g. [118, 240].")
    axes: dict[str, str] = Field(
        description="Axis name -> semantic label, e.g. {'observations': 'stimulus_presentations', 'features': 'neurons'}."
    )
    source_roles: list[RoleName] = Field(default_factory=list)
    transformation_lineage: list[str] = Field(
        default_factory=list,
        description="Ordered transformation names applied to produce this view.",
    )
    artifact_path: str | None = Field(
        default=None,
        description="Filesystem path or URI where the materialized array lives. None for in-memory views.",
    )
    lineage_hash: str = Field(default="")
    summary: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def new(
        cls,
        kind: AnalysisViewKind,
        manifest_id: ManifestId,
        shape: list[int],
        axes: dict[str, str],
        source_roles: list[RoleName],
        transformation_lineage: list[str],
        artifact_path: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> AnalysisView:
        """Construct a view with a deterministic view_id derived from its lineage."""
        lineage_payload = {
            "kind": kind.value,
            "manifest_id": manifest_id,
            "shape": shape,
            "axes": axes,
            "source_roles": source_roles,
            "transformation_lineage": transformation_lineage,
        }
        lineage_hash = stable_hash(lineage_payload)
        return cls(
            view_id=ViewId(f"view_{lineage_hash}"),
            kind=kind,
            manifest_id=manifest_id,
            shape=shape,
            axes=axes,
            source_roles=source_roles,
            transformation_lineage=transformation_lineage,
            artifact_path=artifact_path,
            lineage_hash=lineage_hash,
            summary=summary or {},
        )
