"""Content-addressed cache for transformation/tool outputs.

A `ComputationSpec` describes a computation by its family, scope, name, and
parameters. Its `hash()` is the cache key. The on-disk layout is:

    <artifact_root>/<family>/<scope>/<spec_hash>/
        spec.json            # the input ComputationSpec
        view.json            # the produced AnalysisView
        summary.json         # tool-defined summary dict
        <data files...>      # tool-defined .npy etc.

`cached_computation(spec, compute)` is the high-level entry point: it returns
an existing (view, summary) if the cache already contains them; otherwise it
runs `compute(out_dir)` and persists everything.

This is the single canonical pattern transformations / tools use to persist
their intermediate computations. Adding a new transformation = define a
`ComputationSpec` + a `compute` callback.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from mousehash.artifacts.io import load_json, save_json
from mousehash.artifacts.paths import artifact_root
from mousehash.core.analysis_view import AnalysisView
from mousehash.core.ids import stable_hash


class ComputationSpec(BaseModel):
    """Declarative description of a computation, suitable for content-addressing.

    `label` is informational metadata — it does NOT participate in the hash, so
    a user-friendly tag like "vit_base_imagenet_v0" doesn't fragment the cache
    across otherwise-identical runs.

    `input_fingerprints` should be content hashes of the inputs (e.g. sha1 of a
    numpy array's bytes) so that changing the input invalidates the cache.
    """

    family: str = Field(description="Top-level grouping, e.g. 'representations', 'compression', 'decompositions'.")
    scope: str = Field(description="Human-friendly scope name, usually the dataset / scene_set id.")
    name: str = Field(description="Tool / transformation name, e.g. 'vit_base_imagenet', 'jpeg_size'.")
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_fingerprints: list[str] = Field(default_factory=list)
    label: str | None = Field(default=None, description="Free-form tag, not hashed.")

    def hash(self) -> str:
        """Deterministic short hash of the hashable fields (excludes `label`)."""
        return stable_hash(
            {
                "family": self.family,
                "scope": self.scope,
                "name": self.name,
                "parameters": self.parameters,
                "input_fingerprints": sorted(self.input_fingerprints),
            }
        )


def fingerprint_array(array: np.ndarray) -> str:
    """Stable sha1 of a numpy array's bytes + dtype + shape.

    Cheap (one MD/SHA pass over the buffer) but content-sensitive: changing a
    pixel changes the fingerprint, so the cache invalidates correctly.
    """
    h = hashlib.sha1()
    h.update(str(array.dtype).encode("utf-8"))
    h.update(str(array.shape).encode("utf-8"))
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def cache_dir_for(spec: ComputationSpec) -> Path:
    return artifact_root() / spec.family / spec.scope / spec.hash()


def find_cached_view(spec: ComputationSpec) -> tuple[AnalysisView, dict[str, Any]] | None:
    """If the cache holds a view for this spec, return (view, summary); else None."""
    d = cache_dir_for(spec)
    view_path = d / "view.json"
    summary_path = d / "summary.json"
    if not (view_path.exists() and summary_path.exists()):
        return None
    view = AnalysisView.model_validate_json(view_path.read_text(encoding="utf-8"))
    summary = load_json(summary_path)
    return view, summary


def save_cached_view(
    spec: ComputationSpec,
    view: AnalysisView,
    summary: dict[str, Any],
) -> Path:
    """Write spec.json + view.json + summary.json under the spec's cache dir.

    Returns the cache directory path (the caller can have already saved
    additional data files there).
    """
    d = cache_dir_for(spec)
    d.mkdir(parents=True, exist_ok=True)
    save_json(d / "spec.json", spec.model_dump(mode="json"))
    save_json(d / "view.json", view.model_dump(mode="json"))
    save_json(d / "summary.json", summary)
    return d


ComputeFn = Callable[[Path], tuple[AnalysisView, dict[str, Any]]]


def cached_computation(
    spec: ComputationSpec,
    compute: ComputeFn,
) -> tuple[AnalysisView, dict[str, Any], bool]:
    """Look up the cache; on miss, run `compute(out_dir)` and persist.

    `compute` receives the cache directory and must:
      - write any data files (e.g. .npy) into it,
      - return a (view, summary) pair to be saved.

    Returns `(view, summary, from_cache)`. The boolean is useful for tests and
    for letting tools log "cache hit" / "computed fresh".
    """
    existing = find_cached_view(spec)
    if existing is not None:
        view, summary = existing
        return view, summary, True

    out_dir = cache_dir_for(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "spec.json", spec.model_dump(mode="json"))
    view, summary = compute(out_dir)
    save_json(out_dir / "view.json", view.model_dump(mode="json"))
    save_json(out_dir / "summary.json", summary)
    return view, summary, False
