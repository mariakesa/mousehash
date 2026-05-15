"""ViT-ImageNet feature extraction.

This is target-agnostic: it takes either a stacked (n, H, W) frame array or
a list of image paths and returns logits + softmax probabilities. Allen,
DANDI image stimuli, or anything else that produces grayscale frames goes
through the same path.

`extract_vit_features_view` wraps the raw extraction in an `AnalysisView`
of kind `OBSERVATION_BY_FEATURE` so downstream tools can consume it via
contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mousehash.artifacts.cache import ComputationSpec, cached_computation, fingerprint_array
from mousehash.artifacts.io import save_npy
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.ids import ManifestId
from mousehash.transformations.labeling import derive_animate_inanimate, derive_top1

logger = logging.getLogger(__name__)

DEFAULT_VIT_MODEL = "google/vit-base-patch16-224"
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "cpu"


def run_vit_on_frames(
    frames_array: np.ndarray,
    model_name: str = DEFAULT_VIT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ViT classifier on a (n, H, W) grayscale frame stack.

    Each grayscale frame is replicated across 3 channels (matching the original
    ProjectionSort workflow) and processed one-at-a-time. Returns
    `(logits, probabilities)` as float32 (n, n_classes) arrays.
    """
    try:
        import torch
        from scipy.special import softmax
        from transformers import AutoModelForImageClassification, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "ViT extraction requires the ml + science extras. "
            "Install with: pip install -e '.[ml,science]'"
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    frames_3ch = np.repeat(frames_array[:, None, :, :], 3, axis=1)
    all_logits = np.empty((len(frames_array), model.config.num_labels), dtype=np.float32)
    for index, frame in enumerate(frames_3ch):
        inputs = processor(images=frame, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0).detach().cpu().numpy()
        all_logits[index] = logits.astype(np.float32)
        if (index + 1) % 10 == 0 or index + 1 == len(frames_array):
            logger.info("  ViT processed %d / %d frames", index + 1, len(frames_array))

    probabilities = softmax(all_logits, axis=1).astype(np.float32)
    return all_logits, probabilities


def run_vit(
    image_paths: list[Path],
    model_name: str = DEFAULT_VIT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ViT classifier on a list of image files.

    Lets the HuggingFace processor handle resizing, so this path supports
    mixed-size image datasets without pre-stacking. Returns `(logits, probs)`.
    """
    try:
        import torch
        from PIL import Image
        from scipy.special import softmax
        from transformers import AutoModelForImageClassification, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "ViT extraction requires the ml + science extras. "
            "Install with: pip install -e '.[ml,science]'"
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    all_logits: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu().numpy()
        all_logits.append(logits.astype(np.float32))
        logger.info("  ViT processed %d / %d images", min(start + batch_size, len(image_paths)), len(image_paths))

    logits = np.concatenate(all_logits, axis=0)
    probabilities = softmax(logits, axis=1).astype(np.float32)
    return logits, probabilities


def extract_vit_features_view(
    frames: np.ndarray,
    manifest_id: ManifestId,
    scene_set_id: str,
    representation_spec_id: str = "vit_base_imagenet_v0",
    model_name: str = DEFAULT_VIT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    animate_threshold: int = 397,
) -> tuple[AnalysisView, dict[str, Any]]:
    """Run ViT and package the result as an observation_by_feature AnalysisView.

    The output directory is content-addressed via `cached_computation`: same
    `(scene_set_id, model_name, animate_threshold, frames)` -> same on-disk
    location, and a second call is a cache hit (no ViT re-run).

    `representation_spec_id` is informational only — it does not affect the
    cache key, so re-tagging the same computation doesn't fragment the cache.

    Returns:
        view:   AnalysisView pointing at the feature artifact directory.
        bundle: dict carrying the raw arrays + paths for downstream tools.
                Has a `from_cache` flag for callers that care.
    """
    input_fp = fingerprint_array(frames)
    spec = ComputationSpec(
        family="representations",
        scope=scene_set_id,
        name="vit_imagenet",
        label=representation_spec_id,
        parameters={
            "model_name": model_name,
            "animate_threshold": int(animate_threshold),
        },
        input_fingerprints=[input_fp],
    )

    def _compute(out_dir):
        logger.info(
            "Running ViT (%s, device=%s, batch=%d) on %d frames -> %s",
            model_name, device, batch_size, len(frames), out_dir,
        )
        logits, probabilities = run_vit_on_frames(
            frames, model_name=model_name, batch_size=batch_size, device=device
        )
        top1 = derive_top1(probabilities)
        animate_inanimate = derive_animate_inanimate(top1, animate_threshold)

        save_npy(out_dir / "logits.npy", logits)
        save_npy(out_dir / "probabilities.npy", probabilities)
        save_npy(out_dir / "top1.npy", top1)
        save_npy(out_dir / "animate_inanimate.npy", animate_inanimate)

        n_animate = int(animate_inanimate.sum())
        summary = {
            "scene_set_id": scene_set_id,
            "representation_spec_id": representation_spec_id,
            "model_name": model_name,
            "device": device,
            "batch_size": int(batch_size),
            "animate_threshold": int(animate_threshold),
            "n_images": int(len(frames)),
            "n_classes": int(logits.shape[1]),
            "n_animate": n_animate,
            "n_inanimate": int(len(frames) - n_animate),
            "artifacts": {
                "logits": str(out_dir / "logits.npy"),
                "probabilities": str(out_dir / "probabilities.npy"),
                "top1": str(out_dir / "top1.npy"),
                "animate_inanimate": str(out_dir / "animate_inanimate.npy"),
            },
        }
        view = AnalysisView.new(
            kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
            manifest_id=manifest_id,
            shape=list(logits.shape),
            axes={"observations": "stimulus_presentations", "features": "imagenet_classifier_output"},
            source_roles=["stimuli"],
            transformation_lineage=[
                f"input:{input_fp[:12]}",
                f"vit:{model_name}",
                "softmax",
                f"animate_inanimate@{animate_threshold}",
            ],
            artifact_path=str(out_dir),
            summary={
                "model_name": model_name,
                "n_images": summary["n_images"],
                "n_classes": summary["n_classes"],
                "n_animate": n_animate,
            },
        )
        return view, summary

    view, summary, from_cache = cached_computation(spec, _compute)
    if from_cache:
        logger.info("ViT cache hit (%s) -> %s", spec.hash(), view.artifact_path)

    # Lazy-load the arrays from disk so the caller can use them either way.
    out_dir = Path(view.artifact_path)
    bundle = {
        "view": view,
        "logits": np.load(out_dir / "logits.npy"),
        "probabilities": np.load(out_dir / "probabilities.npy"),
        "top1": np.load(out_dir / "top1.npy"),
        "animate_inanimate": np.load(out_dir / "animate_inanimate.npy"),
        "out_dir": str(out_dir),
        "summary": summary,
        "from_cache": from_cache,
    }
    return view, bundle
