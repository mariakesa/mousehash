from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)


def run_vit_on_frames(
    frames_array: np.ndarray,
    model_name: str = "google/vit-base-patch16-224",
    batch_size: int = 16,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the ViT classifier directly on raw grayscale stimulus frames.

    This matches the original ProjectionSort workflow: repeat each grayscale
    frame across three channels and run the HuggingFace image-classification
    model one frame at a time.

    Args:
        frames_array: ``(n_images, height, width)`` uint8 stimulus array.
        model_name:   HuggingFace model ID.
        batch_size:   Retained for API compatibility; inference runs per-frame.
        device:       ``"cpu"`` or ``"cuda"``.

    Returns:
        Tuple of ``(logits, probabilities)``, each ``(n_images, n_classes)``
        float32 arrays.
    """
    import torch
    from transformers import AutoModelForImageClassification, AutoProcessor

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
        logger.info("  Processed %d / %d images", index + 1, len(frames_array))

    logits = all_logits
    probabilities = softmax(logits, axis=1).astype(np.float32)
    return logits.astype(np.float32), probabilities


def run_vit(
    image_paths: list[Path],
    model_name: str = "google/vit-base-patch16-224",
    batch_size: int = 16,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run ViT ImageNet classifier on a list of image files.

    Loads grayscale image files, stacks them into a raw frame tensor, then
    delegates to :func:`run_vit_on_frames`.

    Args:
        image_paths: Ordered list of image files to classify.
        model_name:  HuggingFace model ID.
        batch_size:  Retained for API compatibility; inference runs per-frame.
        device:      ``"cpu"`` or ``"cuda"``.

    Returns:
        Tuple of ``(logits, probabilities)``, each ``(n_images, n_classes)``
        float32 arrays.
    """
    from PIL import Image

    grayscale_frames = np.stack(
        [np.array(Image.open(path).convert("L"), dtype=np.uint8) for path in image_paths],
        axis=0,
    )
    return run_vit_on_frames(
        grayscale_frames,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )


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
    from mousehash.artifacts.io import load_json, save_json, save_npy
    from mousehash.artifacts.paths import representations_root
    from mousehash.config import ALLEN_MANIFEST_PATH
    from mousehash.schema.representations import (
        AnimateInanimateRule,
        RepresentationSpec,
        StimulusRepresentation,
    )
    from mousehash.schema.stimuli import AllenNaturalSceneImage
    from mousehash.tools.allen.stimulus_fetch import fetch_natural_scene_template
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
        summary_path = (StimulusRepresentation & key).fetch1("summary_path")
        return load_json(Path(summary_path))

    # --- load spec and rule ---
    spec = (RepresentationSpec & {"representation_spec_id": representation_spec_id}).fetch1()
    rule = (AnimateInanimateRule & {"rule_id": rule_id}).fetch1()

    # --- load ordered image paths from DataJoint for downstream artifacts ---
    rows = (AllenNaturalSceneImage & {"scene_set_id": scene_set_id}).fetch(
        "image_idx", "image_path", as_dict=True, order_by="image_idx"
    )
    image_paths = [Path(r["image_path"]) for r in rows]
    frames_array = fetch_natural_scene_template(ALLEN_MANIFEST_PATH)
    if len(frames_array) != len(image_paths):
        raise ValueError(
            "Allen stimulus template length does not match ingested scene set: "
            f"template={len(frames_array)} images, ingested={len(image_paths)} images"
        )
    logger.info(
        "Running ViT on %d images (model=%s, device=%s, batch=%d)",
        len(frames_array),
        spec["model_name"],
        spec["device"],
        spec["batch_size"],
    )

    # --- ViT inference ---
    logits, probabilities = run_vit_on_frames(
        frames_array,
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
