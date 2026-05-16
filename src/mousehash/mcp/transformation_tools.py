"""MCP wrappers for transformations: ViT features + JPEG sizes.

Both transformations take an image stack. The wrappers load that stack from
the target adapter using the manifest's `dataset.target`. For v0 only the
Allen target is wired up; other targets surface a `MouseHashError` until
their adapter lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mousehash.artifacts.paths import manifests_root
from mousehash.core.errors import MouseHashError
from mousehash.core.manifests import RoleManifest
from mousehash.mcp.errors import mcp_safe
from mousehash.transformations.feature_extraction import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_VIT_MODEL,
    extract_vit_features_view,
)
from mousehash.transformations.image_compression import (
    DEFAULT_JPEG_QUALITIES,
    extract_jpeg_size_view,
)


class UnsupportedTargetError(MouseHashError):
    """The manifest's target has no adapter wired up for image materialization yet."""


def _load_manifest(manifest_id: str) -> RoleManifest:
    path = manifests_root() / f"{manifest_id}.yaml"
    if not path.exists():
        from mousehash.mcp.manifest_tools import ManifestNotFoundError
        raise ManifestNotFoundError(
            f"No manifest with id {manifest_id!r} under {manifests_root()}. "
            "Build one first via allen_build_manifest."
        )
    return RoleManifest.from_yaml(path.read_text(encoding="utf-8"))


def _materialize_image_stack(
    manifest: RoleManifest, allen_manifest_path: str = "",
) -> dict[str, Any]:
    """Dispatch to the right adapter based on manifest.dataset.target."""
    target = manifest.dataset.target
    if target == "allen":
        from mousehash.targets.allen.loaders import load_allen_role_bundle
        return load_allen_role_bundle(
            manifest,
            manifest_path=Path(allen_manifest_path).expanduser() if allen_manifest_path else None,
        )
    raise UnsupportedTargetError(
        f"Target {target!r} has no image-stack adapter wired for MCP yet. "
        "Currently supported: allen."
    )


@mcp_safe
def extract_vit_features(
    manifest_id: str,
    representation_spec_id: str = "vit_base_imagenet_v0",
    model_name: str = DEFAULT_VIT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    animate_threshold: int = 397,
    allen_manifest_path: str = "",
) -> dict[str, Any]:
    """Run ViT-ImageNet feature extraction on a manifest's image stack.

    Idempotent: identical args + identical pixel content -> cache hit.

    Args:
        manifest_id: role manifest id returned by `allen_build_manifest`.
        representation_spec_id: informational label (does NOT affect cache key).
        model_name: HuggingFace model id; affects cache key.
        batch_size: forwarded to ViT loop (output is independent of batch_size).
        device: "cpu" or "cuda".
        animate_threshold: ImageNet class index <= this is treated as animate (default 397).
        allen_manifest_path: AllenSDK manifest JSON; empty -> env-resolved.

    Returns:
        {view_id, artifact_path, summary, from_cache}.
    """
    manifest = _load_manifest(manifest_id)
    bundle = _materialize_image_stack(manifest, allen_manifest_path=allen_manifest_path)
    view, vit_bundle = extract_vit_features_view(
        frames=bundle["images"],
        manifest_id=manifest.manifest_id,
        scene_set_id=manifest.dataset.dataset_id,
        representation_spec_id=representation_spec_id,
        model_name=model_name,
        batch_size=int(batch_size),
        device=device,
        animate_threshold=int(animate_threshold),
    )
    return {
        "view_id": view.view_id,
        "artifact_path": view.artifact_path,
        "summary": vit_bundle["summary"],
        "from_cache": vit_bundle.get("from_cache", False),
    }


@mcp_safe
def extract_jpeg_sizes(
    manifest_id: str,
    qualities: list[int] | None = None,
    label: str = "",
    allen_manifest_path: str = "",
) -> dict[str, Any]:
    """Compute JPEG byte sizes at multiple quality levels for a manifest's image stack.

    Idempotent: identical args + identical pixel content -> cache hit.

    Args:
        manifest_id: role manifest id returned by `allen_build_manifest`.
        qualities: PIL JPEG quality levels in [1, 95]. None / empty list -> v0 defaults.
        label: free-form tag (does NOT affect cache key).
        allen_manifest_path: AllenSDK manifest JSON; empty -> env-resolved.

    Returns:
        {view_id, artifact_path, summary, from_cache}.
    """
    manifest = _load_manifest(manifest_id)
    bundle = _materialize_image_stack(manifest, allen_manifest_path=allen_manifest_path)
    q = tuple(qualities) if qualities else DEFAULT_JPEG_QUALITIES
    view, summary = extract_jpeg_size_view(
        frames=bundle["images"],
        manifest_id=manifest.manifest_id,
        scene_set_id=manifest.dataset.dataset_id,
        qualities=q,
        label=label or None,
    )
    return {
        "view_id": view.view_id,
        "artifact_path": view.artifact_path,
        "summary": summary,
        "from_cache": summary.get("from_cache", False),
    }
