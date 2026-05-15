"""AllenAdapter: concrete TargetAdapter for the Allen Brain Observatory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mousehash.artifacts.paths import stimuli_root
from mousehash.core.ids import DatasetId, TargetName
from mousehash.core.manifests import DatasetRef, RoleManifest
from mousehash.core.role_bundle import RoleBundle
from mousehash.targets.allen.client import get_brain_observatory_cache, resolve_manifest_path
from mousehash.targets.allen.loaders import load_allen_role_bundle
from mousehash.targets.allen.manifest import (
    ALLEN_DATASET_NAME,
    ALLEN_TARGET,
    build_natural_scenes_manifest,
)
from mousehash.targets.base import DatasetMetadata, DatasetQuery, LocalResource, ResourceRef

logger = logging.getLogger(__name__)


class AllenAdapter:
    """Allen Brain Observatory adapter.

    Currently scoped to natural-scene stimuli + their per-session presentation
    timing + cell metadata. Other Allen modalities (calcium traces, Neuropixels
    spikes, running, pupil) plug in by adding more roles to the manifest.
    """

    target_name: TargetName = ALLEN_TARGET

    def __init__(self, allen_manifest_path: Path | str | None = None) -> None:
        self.allen_manifest_path = resolve_manifest_path(allen_manifest_path)

    # ---- TargetAdapter Protocol ----

    def list_datasets(self, query: DatasetQuery) -> list[DatasetRef]:
        """For v0 we expose one dataset: the natural-scenes stimulus set."""
        return [self._natural_scenes_ref(scene_set_id="allen_natural_scenes_v1")]

    def get_dataset_metadata(self, dataset_ref: DatasetRef) -> DatasetMetadata:
        boc = get_brain_observatory_cache(str(self.allen_manifest_path))
        exps = boc.get_ophys_experiments(stimuli=["natural_scenes"])
        return DatasetMetadata(
            dataset_ref=dataset_ref,
            title=f"Allen Brain Observatory natural scenes ({dataset_ref.dataset_id})",
            description="118 grayscale natural scene templates and per-session presentation tables.",
            species="Mus musculus",
            n_sessions=len(exps),
            modalities=["natural_scenes", "ophys_calcium"],
            extra={"allen_manifest_path": str(self.allen_manifest_path)},
        )

    def build_manifest(self, dataset_ref: DatasetRef) -> RoleManifest:
        return build_natural_scenes_manifest(
            scene_set_id=dataset_ref.dataset_id,
            manifest_path=self.allen_manifest_path,
        )

    def materialize_resource(self, resource_ref: ResourceRef) -> LocalResource:
        """Return a LocalResource pointing at the on-disk thumbnail directory."""
        scene_set_id = resource_ref.dataset_ref.dataset_id
        image_dir = stimuli_root() / scene_set_id / "images"
        if not image_dir.exists():
            raise FileNotFoundError(
                f"No thumbnails materialized for {scene_set_id}; call build_manifest first."
            )
        return LocalResource(
            resource_ref=resource_ref,
            local_path=str(image_dir),
            is_remote_backed=False,
        )

    def load_role_bundle(self, manifest: RoleManifest) -> RoleBundle:
        """Return the evidence-only RoleBundle from the manifest.

        For the array-bearing dict (images + catalog), call
        `mousehash.targets.allen.loaders.load_allen_role_bundle` directly —
        the Protocol's return type is just evidence.
        """
        return manifest.roles

    # ---- Allen-specific extras ----

    def load_full_bundle(self, manifest: RoleManifest) -> dict[str, Any]:
        """Materialize images + catalog + evidence. Allen-specific shortcut."""
        return load_allen_role_bundle(manifest, manifest_path=self.allen_manifest_path)

    def _natural_scenes_ref(self, scene_set_id: str) -> DatasetRef:
        return DatasetRef(
            target=ALLEN_TARGET,
            dataset_id=DatasetId(scene_set_id),
            dataset_version=ALLEN_DATASET_NAME,
            label=f"Allen natural scenes — {scene_set_id}",
        )
