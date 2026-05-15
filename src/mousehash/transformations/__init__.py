"""Target-agnostic transformations: feature extraction, labeling, etc.

Transformations consume a `RoleBundle` (or pieces of one) and produce an
`AnalysisView`. They are the bridge from "raw target data" to "tool input"
and never call AllenSDK / ONE / pynwb directly.
"""

from mousehash.transformations.feature_extraction import (
    extract_vit_features_view,
    run_vit,
    run_vit_on_frames,
)
from mousehash.transformations.labeling import (
    ANIMATE_MAX_CLASS_IDX,
    derive_animate_inanimate,
    derive_top1,
    load_imagenet_labels,
)

__all__ = [
    "ANIMATE_MAX_CLASS_IDX",
    "derive_animate_inanimate",
    "derive_top1",
    "extract_vit_features_view",
    "load_imagenet_labels",
    "run_vit",
    "run_vit_on_frames",
]
