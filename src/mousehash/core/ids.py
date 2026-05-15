"""Stable identifiers and a deterministic hashing helper.

MouseHash names every dataset, manifest, view, tool run, and artifact so that
provenance is recoverable. The IDs are opaque strings; the hashing helper turns
any JSON-serializable spec into a short, deterministic suffix so two runs with
identical inputs produce identical IDs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, NewType

TargetName = NewType("TargetName", str)
DatasetId = NewType("DatasetId", str)
ManifestId = NewType("ManifestId", str)
ViewId = NewType("ViewId", str)
ToolRunId = NewType("ToolRunId", str)
ArtifactId = NewType("ArtifactId", str)
TransformationRunId = NewType("TransformationRunId", str)


def stable_hash(spec: Any, length: int = 12) -> str:
    """Return a deterministic hex digest of any JSON-serializable spec.

    Used to derive lineage hashes and content-addressable IDs. Sorted keys make
    the digest invariant to dict insertion order.
    """
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]
