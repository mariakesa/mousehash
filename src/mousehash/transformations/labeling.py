"""Label derivation: argmax classes, animate/inanimate, ImageNet vocabulary.

The animate/inanimate threshold sits at ImageNet class 397 — the last animal
class in the standard 1000-class ordering. Classes 0..397 are animate
(animals, insects, fish, birds); 398..999 are inanimate objects.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ANIMATE_MAX_CLASS_IDX = 397

_LABELS_PATH = Path(__file__).parent / "imagenet1000_clsidx_to_labels.txt"


@lru_cache(maxsize=1)
def load_imagenet_labels() -> list[str]:
    """Return the 1000 ImageNet class labels, indexed by class id 0..999."""
    return _LABELS_PATH.read_text(encoding="utf-8").splitlines()


def derive_top1(probabilities: np.ndarray) -> np.ndarray:
    """Return (n,) int32 array of argmax class indices."""
    return probabilities.argmax(axis=1).astype(np.int32)


def derive_animate_inanimate(top1: np.ndarray, threshold_max_class_idx: int = ANIMATE_MAX_CLASS_IDX) -> np.ndarray:
    """Return (n,) int8 array: 1 = animate, 0 = inanimate."""
    return (top1 <= threshold_max_class_idx).astype(np.int8)
