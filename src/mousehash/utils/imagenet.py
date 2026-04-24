from __future__ import annotations

# ImageNet classes 0-397 are animate (animals, insects, fish, birds, etc.).
# Classes 398-999 are inanimate objects.  This boundary is used by the
# animate/inanimate binarization rule seeded in AnimateInanimateRule.
ANIMATE_MAX_CLASS_IDX = 397
