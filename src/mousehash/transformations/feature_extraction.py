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

from mousehash.artifacts.io import save_json, save_npy
from mousehash.artifacts.paths import representations_root
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

    Side effect: saves logits.npy, probabilities.npy, top1.npy,
    animate_inanimate.npy, and summary.json under
    `<artifact_root>/representations/<scene_set_id>/<representation_spec_id>/`.

    Returns:
        view:   AnalysisView pointing at the feature artifact directory.
        bundle: dict carrying the raw arrays + paths for downstream tools.
    """
    logger.info(
        "Running ViT (%s, device=%s, batch=%d) on %d frames",
        model_name, device, batch_size, len(frames),
    )
    logits, probabilities = run_vit_on_frames(
        frames, model_name=model_name, batch_size=batch_size, device=device
    )
    top1 = derive_top1(probabilities)
    animate_inanimate = derive_animate_inanimate(top1, animate_threshold)

    out_dir = representations_root() / scene_set_id / representation_spec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logits_path = save_npy(out_dir / "logits.npy", logits)
    probs_path = save_npy(out_dir / "probabilities.npy", probabilities)
    top1_path = save_npy(out_dir / "top1.npy", top1)
    ai_path = save_npy(out_dir / "animate_inanimate.npy", animate_inanimate)

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
            "logits": str(logits_path),
            "probabilities": str(probs_path),
            "top1": str(top1_path),
            "animate_inanimate": str(ai_path),
        },
    }
    summary_path = save_json(out_dir / "summary.json", summary)

    view = AnalysisView.new(
        kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
        manifest_id=manifest_id,
        shape=list(logits.shape),
        axes={"observations": "stimulus_presentations", "features": "imagenet_classifier_output"},
        source_roles=["stimuli"],
        transformation_lineage=[
            f"vit:{model_name}",
            f"softmax",
            f"animate_inanimate@{animate_threshold}",
        ],
        artifact_path=str(out_dir),
        summary={
            "model_name": model_name,
            "n_images": summary["n_images"],
            "n_classes": summary["n_classes"],
            "n_animate": n_animate,
            "summary_path": str(summary_path),
        },
    )

    bundle = {
        "view": view,
        "logits": logits,
        "probabilities": probabilities,
        "top1": top1,
        "animate_inanimate": animate_inanimate,
        "out_dir": str(out_dir),
        "summary": summary,
    }
    return view, bundle
