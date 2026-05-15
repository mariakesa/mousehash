from __future__ import annotations

import numpy as np


def derive_top1(probabilities: np.ndarray) -> np.ndarray:
    """Return (n,) int32 array of argmax class indices."""
    return probabilities.argmax(axis=1).astype(np.int32)


def derive_animate_inanimate(top1: np.ndarray, threshold_max_class_idx: int) -> np.ndarray:
    """Return (n,) int8 array: 1 = animate, 0 = inanimate."""
    return (top1 <= threshold_max_class_idx).astype(np.int8)
