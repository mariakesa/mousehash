"""MCP wrappers for view enumeration + introspection.

These are thin shims over `mcp/views.py` (which does the actual filesystem
walk). They exist so the agent can ask "what views are on disk?" and "tell
me about this specific view" without invoking compute.
"""

from __future__ import annotations

from typing import Any

from mousehash.mcp.errors import mcp_safe
from mousehash.mcp.views import find_view_by_id, list_view_records


@mcp_safe
def list_views() -> dict[str, Any]:
    """Enumerate every AnalysisView currently in the artifact cache.

    Returns:
        {n_views, views: [{view_id, kind, manifest_id, lineage_hash, shape, artifact_path}, ...]}
    """
    records = list_view_records()
    return {"n_views": len(records), "views": records}


@mcp_safe
def inspect_view(view_id: str) -> dict[str, Any]:
    """Return the full AnalysisView payload for the given view_id.

    Args:
        view_id: identifier returned by any view-producing tool
            (e.g. extract_vit_features, extract_jpeg_sizes, run_pca, run_nmf).

    Returns:
        Full AnalysisView dict: view_id, kind, manifest_id, shape, axes,
        source_roles, transformation_lineage, artifact_path, lineage_hash, summary.
    """
    view = find_view_by_id(view_id)
    return view.model_dump(mode="json")
