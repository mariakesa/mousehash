"""MCP wrappers for target-adapter operations.

Currently exposes the Allen adapter only. DANDI / IBL slot in here when their
adapters land (each gets its own `<target>_list_datasets` / `<target>_build_manifest`
pair).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mousehash.artifacts.paths import manifests_root
from mousehash.mcp.errors import mcp_safe
from mousehash.targets.allen.adapter import AllenAdapter
from mousehash.targets.allen.manifest import build_natural_scenes_manifest
from mousehash.targets.base import DatasetQuery


@mcp_safe
def allen_list_datasets() -> dict[str, Any]:
    """Enumerate Allen Brain Observatory datasets exposed by MouseHash.

    Returns:
        {"target": "allen", "datasets": [{dataset_id, target, dataset_version, label}, ...]}.
        v0 exposes a single dataset for the natural-scenes stimulus set.
    """
    adapter = AllenAdapter()
    refs = adapter.list_datasets(DatasetQuery())
    return {
        "target": "allen",
        "datasets": [r.model_dump(mode="json") for r in refs],
    }


@mcp_safe
def allen_build_manifest(
    scene_set_id: str,
    allen_manifest_path: str = "",
) -> dict[str, Any]:
    """Build (or reuse) the Allen natural-scenes role manifest for a scene set.

    Idempotent: existing thumbnails + image catalog are reused, no re-fetch.

    Args:
        scene_set_id: Label for this scene set (e.g. "allen_natural_scenes_v1").
        allen_manifest_path: AllenSDK BrainObservatoryCache manifest JSON. Empty
            string -> resolved from ALLEN_MANIFEST_PATH or ALLEN_DATA env vars.

    Returns:
        {manifest_id, dataset, satisfied_roles, manifest_yaml_path}.
    """
    manifest = build_natural_scenes_manifest(
        scene_set_id=scene_set_id,
        manifest_path=Path(allen_manifest_path).expanduser() if allen_manifest_path else None,
    )
    return {
        "manifest_id": manifest.manifest_id,
        "dataset": manifest.dataset.model_dump(mode="json"),
        "satisfied_roles": manifest.roles.satisfied_roles(),
        "manifest_yaml_path": str(manifests_root() / f"{manifest.manifest_id}.yaml"),
    }
