from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# ImageNet classes 0-397 are animate (animals, insects, fish, birds, etc.).
# Classes 398-999 are inanimate objects.  This boundary is used by the
# animate/inanimate binarization rule seeded in AnimateInanimateRule.
ANIMATE_MAX_CLASS_IDX = 397

_LABELS_PATH = Path(__file__).parent / "imagenet1000_clsidx_to_labels.txt"


@lru_cache(maxsize=1)
def load_imagenet_labels() -> list[str]:
    """Return the 1000 ImageNet class labels, indexed by class id (0..999)."""
    return _LABELS_PATH.read_text(encoding="utf-8").splitlines()
