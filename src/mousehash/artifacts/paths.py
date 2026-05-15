"""Env-resolved path roots for cache, stimuli, artifacts, reports.

Users keep a `.env` at the project root pointing `MOUSEHASH_DATA_ROOT` at a
directory of their choice (often an external drive). Every path helper below
reads that env var **lazily** — `import mousehash` does not require `.env` to
exist; only calling a path helper does.

Subroots can be overridden individually if the user wants to split caches
across disks:

    MOUSEHASH_DATA_ROOT       — required; default root for all subroots below
    MOUSEHASH_STIMULI_ROOT    — optional; defaults to <DATA_ROOT>/stimuli
    MOUSEHASH_CACHE_ROOT      — optional; defaults to <DATA_ROOT>/cache
    MOUSEHASH_REPORTS_ROOT    — optional; defaults to <DATA_ROOT>/reports
    MOUSEHASH_ARTIFACT_ROOT   — optional; defaults to <DATA_ROOT>/artifacts
    MOUSEHASH_MANIFESTS_ROOT  — optional; defaults to <DATA_ROOT>/manifests
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from mousehash.core.errors import MouseHashError


class MissingEnvError(MouseHashError):
    """Raised when a required MouseHash env var is unset."""


@lru_cache(maxsize=1)
def _load_dotenv_once() -> None:
    """Load .env once per process. lru_cache makes this idempotent + thread-safe."""
    load_dotenv()


def _env_path(name: str, default: Path | None = None) -> Path:
    _load_dotenv_once()
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    if default is not None:
        return default
    raise MissingEnvError(
        f"Required environment variable {name!r} is not set. "
        "Add it to your .env file or export it before running."
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_root() -> Path:
    """Top-level mousehash data directory. Required env: MOUSEHASH_DATA_ROOT."""
    return ensure_dir(_env_path("MOUSEHASH_DATA_ROOT"))


def stimuli_root() -> Path:
    return ensure_dir(_env_path("MOUSEHASH_STIMULI_ROOT", default=data_root() / "stimuli"))


def cache_root() -> Path:
    return ensure_dir(_env_path("MOUSEHASH_CACHE_ROOT", default=data_root() / "cache"))


def reports_root() -> Path:
    return ensure_dir(_env_path("MOUSEHASH_REPORTS_ROOT", default=data_root() / "reports"))


def artifact_root() -> Path:
    return ensure_dir(_env_path("MOUSEHASH_ARTIFACT_ROOT", default=data_root() / "artifacts"))


def manifests_root() -> Path:
    return ensure_dir(_env_path("MOUSEHASH_MANIFESTS_ROOT", default=data_root() / "manifests"))


def representations_root() -> Path:
    """Where ViT logits / probabilities / labels live, scoped per stimulus set."""
    return ensure_dir(artifact_root() / "representations")


def decompositions_root() -> Path:
    """Where PCA / NMF scores + components live, scoped per representation."""
    return ensure_dir(artifact_root() / "decompositions")
