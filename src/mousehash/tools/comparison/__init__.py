"""Two-group statistical comparison of feature views against a label vector."""

from mousehash.tools.comparison.group_comparison import (
    GROUP_COMPARISON_CONTRACT,
    compare_groups_by_label,
    compare_jpeg_animate_inanimate_views,
    interpret_comparison,
    make_comparison_plot,
)

__all__ = [
    "GROUP_COMPARISON_CONTRACT",
    "compare_groups_by_label",
    "compare_jpeg_animate_inanimate_views",
    "interpret_comparison",
    "make_comparison_plot",
]
