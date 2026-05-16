"""MCP wrapper for the structure-discovery HTML report.

Takes IDs (manifest, PCA view, NMF view, ViT view) and assembles every
input the underlying `generate_structure_discovery_report` needs:
- the manifest YAML,
- the on-disk image catalog (for thumbnails),
- the PCA + NMF summary JSONs (for artifact paths),
- the `animate_inanimate.npy` array from the ViT view's artifact directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mousehash.artifacts.io import load_json, load_npy
from mousehash.artifacts.paths import manifests_root
from mousehash.core.errors import MouseHashError
from mousehash.core.manifests import RoleManifest
from mousehash.mcp.errors import mcp_safe
from mousehash.mcp.views import find_view_by_id
from mousehash.targets.allen.manifest import load_image_catalog
from mousehash.tools.reports.structure_discovery import generate_structure_discovery_report


class ReportInputError(MouseHashError):
    """A referenced cache file (animate_inanimate.npy, summary.json) is missing."""


@mcp_safe
def generate_structure_report(
    manifest_id: str,
    pca_view_id: str,
    nmf_view_id: str,
    vit_view_id: str,
    output_dir: str = "",
) -> dict[str, Any]:
    """Assemble the structure-discovery HTML report bundle.

    Args:
        manifest_id: role manifest id (typically from `allen_build_manifest`).
        pca_view_id: view returned by `run_pca`.
        nmf_view_id: view returned by `run_nmf`.
        vit_view_id: source ViT view, used to load the `animate_inanimate.npy`
            label vector for plot coloring.
        output_dir: explicit output directory; empty -> defaults to
            `<reports_root>/<manifest_id>/`.

    Returns:
        {output_dir, reports: {index, pca, nmf}, pca_summary, nmf_summary}.
    """
    # Manifest
    mf_path = manifests_root() / f"{manifest_id}.yaml"
    if not mf_path.exists():
        from mousehash.mcp.manifest_tools import ManifestNotFoundError
        raise ManifestNotFoundError(f"No manifest with id {manifest_id!r}.")
    manifest = RoleManifest.from_yaml(mf_path.read_text(encoding="utf-8"))

    # Views
    pca_view = find_view_by_id(pca_view_id)
    nmf_view = find_view_by_id(nmf_view_id)
    vit_view = find_view_by_id(vit_view_id)

    # Per-view summaries (PCA/NMF caches write summary.json into their dirs)
    pca_summary_path = Path(pca_view.artifact_path) / "summary.json"
    nmf_summary_path = Path(nmf_view.artifact_path) / "summary.json"
    if not pca_summary_path.exists():
        raise ReportInputError(f"PCA summary missing at {pca_summary_path}.")
    if not nmf_summary_path.exists():
        raise ReportInputError(f"NMF summary missing at {nmf_summary_path}.")
    pca_summary = load_json(pca_summary_path)
    nmf_summary = load_json(nmf_summary_path)

    # animate/inanimate labels come from the ViT view's artifact dir
    ai_path = Path(vit_view.artifact_path) / "animate_inanimate.npy"
    if not ai_path.exists():
        raise ReportInputError(
            f"animate_inanimate.npy missing at {ai_path}. "
            "vit_view_id must point at the ViT feature view that produced the labels."
        )
    animate_inanimate = load_npy(ai_path)

    # Image catalog (scoped by scene_set_id = dataset_id for Allen)
    catalog = load_image_catalog(manifest.dataset.dataset_id)

    bundle = generate_structure_discovery_report(
        manifest=manifest,
        pca_summary=pca_summary,
        nmf_summary=nmf_summary,
        image_catalog=catalog["images"],
        animate_inanimate=animate_inanimate,
        output_dir=Path(output_dir).expanduser() if output_dir else None,
        title_prefix=manifest.dataset.label,
    )
    return {
        "manifest_id": manifest_id,
        "output_dir": bundle["output_dir"],
        "reports": bundle["reports"],
        "pca_summary": pca_summary,
        "nmf_summary": nmf_summary,
    }
