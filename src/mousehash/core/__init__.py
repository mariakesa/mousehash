"""Shared scientific core: roles, manifests, views, contracts, artifacts."""

from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.artifact import Artifact, ArtifactKind
from mousehash.core.contracts import (
    ToolContract,
    ToolReadiness,
    check_manifest_satisfies,
)
from mousehash.core.errors import (
    ContractViolationError,
    MouseHashError,
    RoleMissingError,
    ViewKindMismatchError,
)
from mousehash.core.ids import (
    ArtifactId,
    DatasetId,
    ManifestId,
    TargetName,
    ToolRunId,
    ViewId,
    stable_hash,
)
from mousehash.core.manifests import DatasetRef, RoleManifest
from mousehash.core.role_bundle import (
    BehaviorRole,
    ConditionsRole,
    MetadataRole,
    NeuralDataRole,
    RoleBundle,
    RoleConfidence,
    RoleEvidence,
    RoleStatus,
    StimuliRole,
    TimeOrganizationRole,
)

__all__ = [
    "AnalysisView",
    "AnalysisViewKind",
    "Artifact",
    "ArtifactKind",
    "ArtifactId",
    "BehaviorRole",
    "ConditionsRole",
    "ContractViolationError",
    "DatasetId",
    "DatasetRef",
    "ManifestId",
    "MetadataRole",
    "MouseHashError",
    "NeuralDataRole",
    "RoleBundle",
    "RoleConfidence",
    "RoleEvidence",
    "RoleManifest",
    "RoleMissingError",
    "RoleStatus",
    "StimuliRole",
    "TargetName",
    "TimeOrganizationRole",
    "ToolContract",
    "ToolReadiness",
    "ToolRunId",
    "ViewId",
    "ViewKindMismatchError",
    "check_manifest_satisfies",
    "stable_hash",
]
