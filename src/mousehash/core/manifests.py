"""RoleManifest: an evidence-backed mapping from a dataset to its roles.

A manifest is the bridge between target-specific data (Allen sessions, IBL
eids, DANDI assets) and generic MouseHash tools. It says, for one dataset:
which roles are present, with what evidence, and at what confidence — the
readiness engine reads only this object.

Manifests round-trip to JSON and YAML so they can be cached, diffed, and
served as MCP resources without re-running expensive adapters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yaml
from pydantic import BaseModel, Field

from mousehash.core.ids import DatasetId, ManifestId, TargetName, stable_hash
from mousehash.core.role_bundle import RoleBundle


class DatasetRef(BaseModel):
    """Stable reference to a dataset within a target ecosystem."""

    target: TargetName
    dataset_id: DatasetId
    dataset_version: str | None = None
    label: str | None = Field(default=None, description="Human-friendly label, e.g. dandiset title or Allen session name.")

    def fingerprint(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


class RoleManifest(BaseModel):
    """Evidence-backed role manifest for one dataset."""

    manifest_id: ManifestId
    dataset: DatasetRef
    roles: RoleBundle = Field(default_factory=RoleBundle)
    parser_version: str = "0.1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str | None = None

    @classmethod
    def new(
        cls,
        dataset: DatasetRef,
        roles: RoleBundle | None = None,
        parser_version: str = "0.1.0",
        notes: str | None = None,
    ) -> RoleManifest:
        """Construct a manifest, deriving a deterministic manifest_id from the dataset fingerprint."""
        roles = roles or RoleBundle()
        manifest_id = ManifestId(f"mf_{dataset.fingerprint()}")
        return cls(
            manifest_id=manifest_id,
            dataset=dataset,
            roles=roles,
            parser_version=parser_version,
            notes=notes,
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> RoleManifest:
        data: dict[str, Any] = yaml.safe_load(text)
        return cls.model_validate(data)
