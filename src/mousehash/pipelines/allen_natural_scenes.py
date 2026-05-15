"""v0 natural-scenes structure-discovery pipeline.

This is the canonical end-to-end recipe described in architecture doc §13.2:

    Allen adapter -> RoleManifest
    AnalysisView (ViT features)
    -> run_pca
    -> run_nmf
    -> structure-discovery report

The pipeline is a thin orchestrator; every step is a separately auditable
call into adapters / transformations / tools.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mousehash.artifacts.paths import reports_root
from mousehash.targets.allen.adapter import AllenAdapter
from mousehash.targets.allen.loaders import load_allen_role_bundle
from mousehash.targets.allen.manifest import build_natural_scenes_manifest
from mousehash.tools.factor_models.nmf import run_nmf
from mousehash.tools.factor_models.pca import run_pca
from mousehash.tools.reports.structure_discovery import generate_structure_discovery_report
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

logger = logging.getLogger(__name__)


def run_allen_natural_scenes_v0(
    scene_set_id: str = "allen_natural_scenes_v1",
    allen_manifest_path: Path | str | None = None,
    representation_spec_id: str = "vit_base_imagenet_v0",
    pca_n_components: int = 10,
    nmf_n_components: int = 10,
    nmf_temperature: float = 1.0,
    vit_model_name: str = DEFAULT_VIT_MODEL,
    vit_batch_size: int = DEFAULT_BATCH_SIZE,
    vit_device: str = DEFAULT_DEVICE,
    animate_threshold: int = 397,
    jpeg_qualities: tuple[int, ...] | list[int] = DEFAULT_JPEG_QUALITIES,
    report_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full Allen natural-scenes structure-discovery pipeline.

    Steps:
      1. Build / reuse the Allen natural-scenes RoleManifest.
      2. Materialize the image stack via the adapter.
      3. ViT-ImageNet feature extraction -> AnalysisView.
      4. JPEG-compressibility feature extraction -> AnalysisView.
      5. PCA on logits and NMF on probabilities.
      6. Structure-discovery HTML report.

    Returns a dict with every artifact path the pipeline produced.
    """
    adapter = AllenAdapter(allen_manifest_path=allen_manifest_path)
    logger.info("Building Allen manifest for scene_set_id=%s", scene_set_id)
    manifest = build_natural_scenes_manifest(
        scene_set_id=scene_set_id,
        manifest_path=adapter.allen_manifest_path,
    )

    bundle = load_allen_role_bundle(manifest, manifest_path=adapter.allen_manifest_path)
    images = bundle["images"]
    image_catalog = bundle["image_catalog"]
    logger.info("Materialized %d images for ViT extraction", len(images))

    view, vit_bundle = extract_vit_features_view(
        frames=images,
        manifest_id=manifest.manifest_id,
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        model_name=vit_model_name,
        batch_size=vit_batch_size,
        device=vit_device,
        animate_threshold=animate_threshold,
    )

    jpeg_view, jpeg_summary = extract_jpeg_size_view(
        frames=images,
        manifest_id=manifest.manifest_id,
        scene_set_id=scene_set_id,
        qualities=tuple(jpeg_qualities),
        label=f"jpeg_v0_{scene_set_id}",
    )
    logger.info(
        "JPEG view: %s (shape=%s, qualities=%s)",
        jpeg_view.view_id, jpeg_view.shape, jpeg_summary["qualities"],
    )

    pca_summary = run_pca(
        view=view,
        n_components=pca_n_components,
        normalize_input=True,
        input_array_name="logits",
        decomposition_spec_id=f"pca_logits_{pca_n_components}",
    )
    nmf_summary = run_nmf(
        view=view,
        n_components=nmf_n_components,
        temperature=nmf_temperature,
        input_array_name="probabilities",
        decomposition_spec_id=f"nmf_probs_{nmf_n_components}_T{nmf_temperature}",
    )

    report_dir = report_output_dir or (reports_root() / manifest.manifest_id)
    report_summary = generate_structure_discovery_report(
        manifest=manifest,
        pca_summary=pca_summary,
        nmf_summary=nmf_summary,
        image_catalog=image_catalog,
        animate_inanimate=vit_bundle["animate_inanimate"],
        output_dir=report_dir,
        title_prefix=manifest.dataset.label,
    )

    return {
        "manifest": manifest.model_dump(mode="json"),
        "manifest_id": manifest.manifest_id,
        "view_id": view.view_id,
        "vit_summary": vit_bundle["summary"],
        "jpeg_view_id": jpeg_view.view_id,
        "jpeg_summary": jpeg_summary,
        "pca_summary": pca_summary,
        "nmf_summary": nmf_summary,
        "report": report_summary,
    }
