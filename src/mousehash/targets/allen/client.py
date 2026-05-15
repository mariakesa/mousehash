"""Thin glue around AllenSDK's BrainObservatoryCache.

AllenSDK is a heavy dependency that pulls in HDF5, lazy-loaded HTTP caches,
and a manifest JSON the user must place somewhere on disk. We import it
lazily so `import mousehash.targets.allen` does not require `.[allen]` to be
installed.

The manifest path resolves from `ALLEN_MANIFEST_PATH` env var by default,
matching the convention from the prior MouseHash pipeline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from mousehash.artifacts.paths import _load_dotenv_once
from mousehash.core.errors import MouseHashError


class AllenSDKMissingError(MouseHashError):
    """allensdk extra is not installed."""


def require_allensdk():
    """Import allensdk lazily. Raises AllenSDKMissingError if the extra isn't installed."""
    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    except ImportError as exc:
        raise AllenSDKMissingError(
            "allensdk is not installed. Install the Allen extra: pip install -e '.[allen]'"
        ) from exc
    return BrainObservatoryCache


def resolve_manifest_path(manifest_path: Path | str | None = None) -> Path:
    """Resolve the BrainObservatoryCache manifest JSON path.

    Precedence: explicit arg > ALLEN_MANIFEST_PATH env > raise.
    """
    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve()
    _load_dotenv_once()
    env = os.environ.get("ALLEN_MANIFEST_PATH")
    if env:
        return Path(env).expanduser().resolve()
    raise MouseHashError(
        "AllenSDK manifest path is unset. Pass manifest_path explicitly "
        "or set ALLEN_MANIFEST_PATH in your .env."
    )


@lru_cache(maxsize=4)
def get_brain_observatory_cache(manifest_path_str: str) -> Any:
    """Cache one BrainObservatoryCache per manifest path. Module-level to avoid re-opening HDF5."""
    BrainObservatoryCache = require_allensdk()
    return BrainObservatoryCache(manifest_file=manifest_path_str)
