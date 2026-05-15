"""JPEG-compressibility feature extraction.

For each image and each quality level in `qualities`, encode the image as JPEG
and record the resulting byte size. The output is a (n_images, n_qualities)
matrix exposed as an `OBSERVATION_BY_FEATURE` AnalysisView, alongside its
kilobyte and megabyte projections for convenience.

Scientific motivation: JPEG size at a fixed quality is a cheap, model-free
proxy for image complexity / spatial-frequency content. Comparing JPEG-size
features against ViT-feature decompositions (or against neural responses)
isolates how much "structure" downstream tools capture beyond raw
compressibility.

Like the ViT extractor, this transformation is content-addressed via
`cached_computation`: same `(scene_set_id, qualities, frames)` -> same
on-disk location, second call is a cache hit.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np

from mousehash.artifacts.cache import ComputationSpec, cached_computation, fingerprint_array
from mousehash.artifacts.io import save_npy
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.ids import ManifestId

logger = logging.getLogger(__name__)

DEFAULT_JPEG_QUALITIES: tuple[int, ...] = (10, 25, 50, 75, 90)


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "JPEG compression requires Pillow. Install with: pip install -e '.[allen]' (Pillow is in the allen extra)."
        ) from exc
    return Image


def _to_uint8_global(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalize an arbitrary-dtype frame to uint8 using global min/max.

    Centralized so the encoder is invariant to the input dtype/range — the
    same JPEG bytes come out whether the caller passes uint8 thumbnails,
    float32 templates, or raw Allen uint16-ish arrays.
    """
    frame = np.asarray(frame).astype(np.float32)
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    if hi == lo:
        return np.zeros_like(frame, dtype=np.uint8)
    frame = (frame - lo) / (hi - lo)
    frame = np.clip(frame, 0.0, 1.0)
    return np.round(frame * 255.0).astype(np.uint8)


def jpeg_size_bytes(image_u8: np.ndarray, quality: int) -> int:
    """Encode a single 2D uint8 image as JPEG and return its byte count."""
    Image = _require_pillow()
    buf = io.BytesIO()
    Image.fromarray(image_u8).convert("L").save(buf, format="JPEG", quality=int(quality))
    return buf.tell()


def extract_jpeg_size_view(
    frames: np.ndarray,
    manifest_id: ManifestId,
    scene_set_id: str,
    qualities: tuple[int, ...] | list[int] = DEFAULT_JPEG_QUALITIES,
    label: str | None = None,
) -> tuple[AnalysisView, dict[str, Any]]:
    """Compute JPEG byte sizes for `frames` at each quality level.

    Args:
        frames: (n, H, W) image stack of any numeric dtype; normalized to
                uint8 via global min/max before encoding.
        manifest_id: the dataset's RoleManifest id (for view lineage).
        scene_set_id: scope name (Allen scene set id, dandiset id, etc).
        qualities: PIL JPEG quality levels in [1, 95].
        label: informational tag; does not affect the cache key.

    Returns:
        view:    OBSERVATION_BY_FEATURE AnalysisView with shape (n, len(qualities)).
                 `view.summary["units"] = "bytes"`.
        summary: per-quality means + min/max + artifact paths.

    Re-running with identical inputs is a cache hit (no re-encoding).
    """
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"Expected (n, H, W) image stack, got shape {frames.shape}")
    qualities_sorted = sorted(int(q) for q in qualities)
    for q in qualities_sorted:
        if not (1 <= q <= 95):
            raise ValueError(f"JPEG quality must be in [1, 95], got {q}")

    input_fp = fingerprint_array(frames)
    spec = ComputationSpec(
        family="compression",
        scope=scene_set_id,
        name="jpeg_size",
        label=label,
        parameters={"qualities": qualities_sorted},
        input_fingerprints=[input_fp],
    )

    def _compute(out_dir: Path):
        n = frames.shape[0]
        Q = len(qualities_sorted)
        lo = float(np.nanmin(frames))
        hi = float(np.nanmax(frames))

        sizes_bytes = np.zeros((n, Q), dtype=np.int64)
        for i in range(n):
            img_u8 = _to_uint8_global(frames[i], lo, hi)
            for j, q in enumerate(qualities_sorted):
                sizes_bytes[i, j] = jpeg_size_bytes(img_u8, q)
            if (i + 1) % 25 == 0 or i + 1 == n:
                logger.info("  JPEG encoded %d / %d images across %d qualities", i + 1, n, Q)

        sizes_kb = (sizes_bytes / 1024.0).astype(np.float32)
        sizes_mb = (sizes_bytes / (1024.0 * 1024.0)).astype(np.float32)

        save_npy(out_dir / "sizes_bytes.npy", sizes_bytes)
        save_npy(out_dir / "sizes_kb.npy", sizes_kb)
        save_npy(out_dir / "sizes_mb.npy", sizes_mb)

        per_quality_kb = sizes_kb.mean(axis=0).tolist()
        summary = {
            "scene_set_id": scene_set_id,
            "qualities": qualities_sorted,
            "n_images": int(n),
            "n_qualities": Q,
            "units": "bytes",
            "global_min_pixel": lo,
            "global_max_pixel": hi,
            "mean_size_bytes_by_quality": sizes_bytes.mean(axis=0).tolist(),
            "mean_size_kb_by_quality": per_quality_kb,
            "median_size_kb_by_quality": np.median(sizes_kb, axis=0).tolist(),
            "min_size_kb": float(sizes_kb.min()),
            "max_size_kb": float(sizes_kb.max()),
            "max_size_mb": float(sizes_mb.max()),
            "artifacts": {
                "sizes_bytes": str(out_dir / "sizes_bytes.npy"),
                "sizes_kb": str(out_dir / "sizes_kb.npy"),
                "sizes_mb": str(out_dir / "sizes_mb.npy"),
            },
        }

        view = AnalysisView.new(
            kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
            manifest_id=manifest_id,
            shape=[int(n), Q],
            axes={"observations": "stimulus_presentations", "features": "jpeg_quality_levels"},
            source_roles=["stimuli"],
            transformation_lineage=[
                f"input:{input_fp[:12]}",
                "normalize:global_uint8",
                f"jpeg_encode:qualities={','.join(str(q) for q in qualities_sorted)}",
                "size_bytes",
            ],
            artifact_path=str(out_dir),
            summary={
                "qualities": qualities_sorted,
                "n_images": int(n),
                "units": "bytes",
            },
        )
        return view, summary

    view, summary, from_cache = cached_computation(spec, _compute)
    if from_cache:
        logger.info("JPEG cache hit (%s) -> %s", spec.hash(), view.artifact_path)
    summary["from_cache"] = from_cache
    return view, summary
