"""Content hashing helpers for on-disk artifacts.

`core.ids.stable_hash` hashes JSON-serializable specs (for view/manifest IDs).
This file hashes raw bytes (for image SHA1 catalogs, npy file fingerprints).
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha1_file(path: Path, chunk_size: int = 65536) -> str:
    """SHA-1 hex digest of a file's contents. Streams in chunks for large files."""
    h = hashlib.sha1()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
