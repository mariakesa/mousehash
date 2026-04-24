from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)


def run_vit(
    image_paths: list[Path],
    model_name: str = "google/vit-base-patch16-224",
    batch_size: int = 16,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run ViT ImageNet classifier on a list of image files.

    Converts grayscale images to RGB before processing (AllenSDK natural
    scenes are single-channel; ViT expects three-channel input).

    Args:
        image_paths: Ordered list of image files to classify.
        model_name:  HuggingFace model ID.
        batch_size:  Number of images per forward pass.
        device:      ``"cpu"`` or ``"cuda"``.

    Returns:
        Tuple of ``(logits, probabilities)``, each ``(n_images, n_classes)``
        float32 arrays.
    """
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification, ViTImageProcessor

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    all_logits: list[np.ndarray] = []
    n = len(image_paths)

    for start in range(0, n, batch_size):
        batch = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        all_logits.append(outputs.logits.cpu().float().numpy())
        logger.info("  Processed %d / %d images", min(start + batch_size, n), n)

    logits = np.concatenate(all_logits, axis=0)          # (n, 1000)
    probabilities = softmax(logits, axis=1).astype(np.float32)
    return logits.astype(np.float32), probabilities


def compute_stimulus_representation(
    scene_set_id: str,
    representation_spec_id: str,
    rule_id: str,
) -> dict:
    """Full pipeline: load images → ViT → save arrays → register in DataJoint.

    Idempotent: if the DataJoint record already exists the function returns
    immediately without recomputing.

    Args:
        scene_set_id:           Primary key of an ingested ``AllenNaturalSceneSet``.
        representation_spec_id: Primary key of a ``RepresentationSpec`` row.
        rule_id:                Primary key of an ``AnimateInanimateRule`` row.

    Returns:
        Summary dict (mirrors what is stored in ``summary.json``).
    """
    from mousehash.artifacts.io import save_json, save_npy
    from mousehash.artifacts.paths import representations_root
    from mousehash.schema.representations import (
        AnimateInanimateRule,
        RepresentationSpec,
        StimulusRepresentation,
    )
    from mousehash.schema.stimuli import AllenNaturalSceneImage
    from mousehash.tools.representations.animate_inanimate import (
        derive_animate_inanimate,
        derive_top1,
    )

    # --- guard: already computed? ---
    key = dict(
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        rule_id=rule_id,
    )
    if StimulusRepresentation & key:
        logger.info("StimulusRepresentation already exists for %s, skipping.", key)
        return (StimulusRepresentation & key).fetch1()

    # --- load spec and rule ---
    spec = (RepresentationSpec & {"representation_spec_id": representation_spec_id}).fetch1()
    rule = (AnimateInanimateRule & {"rule_id": rule_id}).fetch1()

    # --- load ordered image paths from DataJoint ---
    rows = (AllenNaturalSceneImage & {"scene_set_id": scene_set_id}).fetch(
        "image_idx", "image_path", as_dict=True, order_by="image_idx"
    )
    image_paths = [Path(r["image_path"]) for r in rows]
    logger.info(
        "Running ViT on %d images (model=%s, device=%s, batch=%d)",
        len(image_paths),
        spec["model_name"],
        spec["device"],
        spec["batch_size"],
    )

    # --- ViT inference ---
    logits, probabilities = run_vit(
        image_paths,
        model_name=spec["model_name"],
        batch_size=spec["batch_size"],
        device=spec["device"],
    )

    # --- semantic labels ---
    top1 = derive_top1(probabilities)
    animate_inanimate = derive_animate_inanimate(top1, rule["threshold_max_class_idx"])

    # --- save arrays ---
    out_dir = representations_root() / scene_set_id / representation_spec_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logits_path = out_dir / "logits.npy"
    probs_path = out_dir / "probabilities.npy"
    top1_path = out_dir / "top1.npy"
    ai_path = out_dir / "animate_inanimate.npy"
    summary_path = out_dir / "summary.json"

    save_npy(logits_path, logits)
    save_npy(probs_path, probabilities)
    save_npy(top1_path, top1)
    save_npy(ai_path, animate_inanimate)

    # --- summary ---
    unique_cls, counts = np.unique(top1, return_counts=True)
    n_animate = int(animate_inanimate.sum())
    summary = dict(
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        rule_id=rule_id,
        n_images=len(image_paths),
        n_classes=int(logits.shape[1]),
        n_animate=n_animate,
        n_inanimate=len(image_paths) - n_animate,
        top1_distribution={str(int(c)): int(k) for c, k in zip(unique_cls, counts)},
    )
    save_json(summary_path, summary)

    # --- register in DataJoint ---
    StimulusRepresentation.insert1(
        dict(
            **key,
            n_images=summary["n_images"],
            n_classes=summary["n_classes"],
            logits_path=str(logits_path),
            probabilities_path=str(probs_path),
            top1_path=str(top1_path),
            animate_inanimate_path=str(ai_path),
            summary_path=str(summary_path),
        ),
        skip_duplicates=True,
    )
    logger.info(
        "Saved representation: %d animate, %d inanimate.",
        n_animate,
        summary["n_inanimate"],
    )
    return summary
