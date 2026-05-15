"""TargetAdapter: the contract every data-source adapter must satisfy.

Allen, IBL, and DANDI each speak their own dialect. The adapter layer
translates that dialect into MouseHash concepts — datasets, resources,
role manifests, role bundles — so that everything downstream is target-
agnostic.

The Protocol is intentionally narrow: an adapter is an ingestion + access
layer, not an analysis system.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from mousehash.core.ids import DatasetId, TargetName
from mousehash.core.manifests import DatasetRef, RoleManifest
from mousehash.core.role_bundle import RoleBundle

__all__ = [
    "DatasetMetadata",
    "DatasetQuery",
    "DatasetRef",
    "LocalResource",
    "ResourceRef",
    "TargetAdapter",
]


class DatasetQuery(BaseModel):
    """Adapter-agnostic search parameters. Adapters interpret fields they understand and ignore the rest."""

    target: TargetName | None = None
    text: str | None = None
    species: str | None = None
    modality: str | None = None
    brain_region: str | None = None
    limit: int = 50
    extra: dict[str, Any] = Field(default_factory=dict)


class DatasetMetadata(BaseModel):
    """Lightweight metadata for one dataset, returned by an adapter's metadata lookup."""

    dataset_ref: DatasetRef
    title: str | None = None
    description: str | None = None
    species: str | None = None
    n_subjects: int | None = None
    n_sessions: int | None = None
    modalities: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class ResourceRef(BaseModel):
    """Pointer to a single materializable resource inside a dataset (an NWB asset, a session file, etc.)."""

    dataset_ref: DatasetRef
    resource_id: str
    resource_kind: str = Field(description="Adapter-defined, e.g. 'nwb_asset', 'allen_session', 'ibl_eid'.")
    extra: dict[str, Any] = Field(default_factory=dict)


class LocalResource(BaseModel):
    """A resource that has been fetched (or made readable) on the local machine."""

    resource_ref: ResourceRef
    local_path: str
    is_remote_backed: bool = Field(
        default=False,
        description="True when local_path is a streaming/remote-backed handle rather than a downloaded file.",
    )
    byte_size: int | None = None


@runtime_checkable
class TargetAdapter(Protocol):
    """Every target adapter implements this surface.

    Adapters MUST NOT analyze data. They list, fetch, and translate. Once a
    `RoleBundle` exists, shared MouseHash tools take over.
    """

    target_name: TargetName

    def list_datasets(self, query: DatasetQuery) -> list[DatasetRef]: ...

    def get_dataset_metadata(self, dataset_ref: DatasetRef) -> DatasetMetadata: ...

    def build_manifest(self, dataset_ref: DatasetRef) -> RoleManifest: ...

    def materialize_resource(self, resource_ref: ResourceRef) -> LocalResource: ...

    def load_role_bundle(self, manifest: RoleManifest) -> RoleBundle: ...


# Re-export for convenience: callers can import DatasetId from either core.ids or targets.base.
__all__ += ["DatasetId"]
