from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def manifest_sha256(path: Path) -> str:
    """SHA256 of the raw on-disk YAML bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def tool_run_spec_id(
    tool_id: str,
    manifest_sha256_hex: str,
    parameters: dict[str, Any],
    input_artifacts: dict[str, Any],
) -> str:
    """Content hash for a resolved spec. 16 hex chars = 64-bit collision space.

    Two resolutions with identical (tool_id, manifest, params, inputs) yield
    identical IDs, so the DataJoint insert is naturally idempotent.
    """
    payload = _canonical_json(
        dict(
            tool_id=tool_id,
            manifest_sha256=manifest_sha256_hex,
            parameters=parameters,
            input_artifacts=input_artifacts,
        )
    )
    return hashlib.sha256(payload).hexdigest()[:16]
