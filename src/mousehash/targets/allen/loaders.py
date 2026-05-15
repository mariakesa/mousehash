"""Materialize an Allen RoleManifest into a working RoleBundle for downstream tools.

The manifest carries identity + evidence; the bundle carries the actual
numpy arrays / DataFrames. This separation lets manifests be cheap to serve
over MCP while bundle materialization is opt-in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mousehash.core.manifests import RoleManifest
from mousehash.core.role_bundle import RoleBundle
from mousehash.targets.allen.manifest import load_image_catalog
from mousehash.targets.allen.stimuli import fetch_natural_scene_template

logger = logging.getLogger(__name__)


def load_natural_scene_image_stack(scene_set_id: str, manifest_path: Path | str | None = None) -> np.ndarray:
    """Return the full (n_images, H, W) raw template for downstream feature extraction.

    `manifest_path` here is the AllenSDK manifest, not the MouseHash manifest.
    The image stack is fetched fresh from AllenSDK so ViT runs on the canonical
    pixel values (the disk thumbnails are downscaled for display only).
    """
    return fetch_natural_scene_template(manifest_path)


def load_allen_role_bundle(manifest: RoleManifest, manifest_path: Path | str | None = None) -> dict[str, Any]:
    """Materialize the bundle dict for an Allen natural-scenes manifest.

    Returns a plain dict (not a `RoleBundle`) because the bundle's role-class
    fields hold evidence, not arrays. The dict shape is:

        {
            "manifest_id": str,
            "scene_set_id": str,
            "images": np.ndarray (n_images, H, W) uint-ish,
            "image_catalog": list[dict],    # per-image SHA1/path catalog
        }

    Roles themselves are read off `manifest.roles`.
    """
    scene_set_id = manifest.dataset.dataset_id
    catalog = load_image_catalog(scene_set_id)
    images = load_natural_scene_image_stack(scene_set_id, manifest_path=manifest_path)
    if len(images) != catalog["n_images"]:
        raise ValueError(
            f"Image stack / catalog mismatch for {scene_set_id}: "
            f"stack={len(images)} catalog={catalog['n_images']}"
        )
    return {
        "manifest_id": manifest.manifest_id,
        "scene_set_id": scene_set_id,
        "images": images,
        "image_catalog": catalog["images"],
        "roles": manifest.roles,
    }


def get_role_bundle_evidence(manifest: RoleManifest) -> RoleBundle:
    """Return the manifest's role evidence bundle (no array materialization)."""
    return manifest.roles
