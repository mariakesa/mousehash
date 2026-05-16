"""MCP wrapper for the all-in-one v0 pipeline.

Convenience tool. Instead of chaining
`allen_build_manifest` -> `extract_vit_features` -> `extract_jpeg_sizes`
-> `run_pca` -> `run_nmf` -> `generate_structure_report` step-by-step, an
agent can call this one tool with a scene_set_id and the full bundle
materializes (with cache hits everywhere along the way).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mousehash.mcp.errors import mcp_safe
from mousehash.pipelines.allen_natural_scenes import run_allen_natural_scenes_v0 as _run_v0
from mousehash.transformations.feature_extraction import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_VIT_MODEL,
)
from mousehash.transformations.image_compression import DEFAULT_JPEG_QUALITIES


@mcp_safe
def run_allen_natural_scenes_v0(
    scene_set_id: str = "allen_natural_scenes_v1",
    pca_n_components: int = 10,
    nmf_n_components: int = 10,
    nmf_temperature: float = 1.0,
    vit_model_name: str = DEFAULT_VIT_MODEL,
    vit_batch_size: int = DEFAULT_BATCH_SIZE,
    vit_device: str = DEFAULT_DEVICE,
    animate_threshold: int = 397,
    jpeg_qualities: list[int] | None = None,
    allen_manifest_path: str = "",
    report_output_dir: str = "",
) -> dict[str, Any]:
    """Run the full Allen natural-scenes pipeline: manifest -> ViT + JPEG -> PCA + NMF -> report.

    Idempotent end-to-end: every step uses the cached_computation pattern, so a
    second call with identical args reuses everything.

    Args:
        scene_set_id: scene-set scope (default: allen_natural_scenes_v1).
        pca_n_components / nmf_n_components: number of factors for each decomposition.
        nmf_temperature: softmax temperature applied to probabilities before NMF.
        vit_model_name: HuggingFace ViT model id.
        vit_batch_size / vit_device: forwarded to the ViT runner.
        animate_threshold: ImageNet class index <= this is "animate".
        jpeg_qualities: list of JPEG quality levels. None / empty -> v0 defaults.
        allen_manifest_path: AllenSDK BrainObservatoryCache manifest JSON; empty
            -> env-resolved (ALLEN_MANIFEST_PATH / ALLEN_DATA).
        report_output_dir: where to write the HTML bundle; empty -> default under
            reports_root.

    Returns:
        Pipeline result dict: manifest_id, view_id (ViT), jpeg_view_id, pca_view_id,
        nmf_view_id, summaries for each step, report bundle paths.
    """
    result = _run_v0(
        scene_set_id=scene_set_id,
        allen_manifest_path=Path(allen_manifest_path).expanduser() if allen_manifest_path else None,
        pca_n_components=int(pca_n_components),
        nmf_n_components=int(nmf_n_components),
        nmf_temperature=float(nmf_temperature),
        vit_model_name=vit_model_name,
        vit_batch_size=int(vit_batch_size),
        vit_device=vit_device,
        animate_threshold=int(animate_threshold),
        jpeg_qualities=tuple(jpeg_qualities) if jpeg_qualities else DEFAULT_JPEG_QUALITIES,
        report_output_dir=Path(report_output_dir).expanduser() if report_output_dir else None,
    )
    # Pipeline result is already JSON-friendly (it model_dumps the manifest), pass through.
    return result
