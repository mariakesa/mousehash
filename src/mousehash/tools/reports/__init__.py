"""HTML report generators that consume tool outputs and emit auditable bundles."""

from mousehash.tools.reports.structure_discovery import (
    build_nmf_report,
    build_pca_report,
    generate_structure_discovery_report,
)

__all__ = [
    "build_nmf_report",
    "build_pca_report",
    "generate_structure_discovery_report",
]
