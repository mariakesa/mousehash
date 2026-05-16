"""Look up `AnalysisView` objects by view_id from the artifact cache.

Every transformation / tool that uses `cached_computation` writes a
`view.json` next to its output. To find a view by id we walk
`artifact_root().rglob("view.json")` and match `view.view_id`. This is O(n)
over the cache but n is small (single-digit hundreds in v0).

When cardinality grows past ~1000, replace with an index file written
inside `cached_computation`. The public surface (`find_view_by_id` /
`list_view_records`) stays stable across that migration.
"""

from __future__ import annotations

from typing import Any

from mousehash.artifacts.paths import artifact_root
from mousehash.core.analysis_view import AnalysisView
from mousehash.mcp.errors import ViewNotFoundError


def _iter_view_paths():
    root = artifact_root()
    if not root.exists():
        return
    yield from root.rglob("view.json")


def find_view_by_id(view_id: str) -> AnalysisView:
    """Return the AnalysisView whose `view_id` matches; raise `ViewNotFoundError` on miss."""
    for path in _iter_view_paths():
        try:
            view = AnalysisView.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if view.view_id == view_id:
            return view
    raise ViewNotFoundError(f"No AnalysisView with view_id={view_id!r} in {artifact_root()}.")


def list_view_records() -> list[dict[str, Any]]:
    """Enumerate every cached view as compact dict records.

    Returns one entry per discovered `view.json`:
        {view_id, kind, manifest_id, lineage_hash, shape, artifact_path}.

    Skips malformed files silently — partial / in-progress writes shouldn't
    crash the enumeration.
    """
    records: list[dict[str, Any]] = []
    for path in _iter_view_paths():
        try:
            view = AnalysisView.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append({
            "view_id": view.view_id,
            "kind": view.kind.value,
            "manifest_id": view.manifest_id,
            "lineage_hash": view.lineage_hash,
            "shape": view.shape,
            "artifact_path": view.artifact_path,
        })
    return records
