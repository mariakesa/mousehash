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
from mousehash.transformations.image_compression import (
    DEFAULT_JPEG_QUALITIES,
    extract_jpeg_size_view,
    jpeg_size_bytes,
)
from mousehash.transformations.labeling import (
    ANIMATE_MAX_CLASS_IDX,
    derive_animate_inanimate,
    derive_top1,
    load_imagenet_labels,
)

__all__ = [
    "ANIMATE_MAX_CLASS_IDX",
    "DEFAULT_JPEG_QUALITIES",
    "derive_animate_inanimate",
    "derive_top1",
    "extract_jpeg_size_view",
    "extract_vit_features_view",
    "jpeg_size_bytes",
    "load_imagenet_labels",
    "run_vit",
    "run_vit_on_frames",
]
