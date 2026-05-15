"""Shared pytest fixtures.

The key fixture is `data_root_tmp`: every test that touches paths gets its
own isolated MOUSEHASH_DATA_ROOT under tmp_path. We also clear the dotenv
lru_cache so prior tests don't leak env state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest


_MOUSEHASH_ENV_VARS = [
    "MOUSEHASH_DATA_ROOT",
    "MOUSEHASH_STIMULI_ROOT",
    "MOUSEHASH_CACHE_ROOT",
    "MOUSEHASH_REPORTS_ROOT",
    "MOUSEHASH_ARTIFACT_ROOT",
    "MOUSEHASH_MANIFESTS_ROOT",
    "ALLEN_MANIFEST_PATH",
    "ALLEN_DATA",
    "ALLEN_DATA_ROOT",
]


def _reset_paths_cache() -> None:
    """Clear the lru_cache on _load_dotenv_once so each test starts fresh."""
    from mousehash.artifacts import paths as _paths
    _paths._load_dotenv_once.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_mousehash_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip MOUSEHASH_* env vars by default. Tests that need them set should
    use the `data_root_tmp` fixture or monkeypatch them explicitly.

    Crucially: stub the inner `load_dotenv` so the lazy loader does not pick up
    the repo's real `.env` (which lives next to the conftest and would otherwise
    re-populate MOUSEHASH_DATA_ROOT after we delenv'd it).
    """
    for var in _MOUSEHASH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("mousehash.artifacts.paths.load_dotenv", lambda *a, **kw: False)
    _reset_paths_cache()
    yield
    _reset_paths_cache()


@pytest.fixture
def data_root_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set MOUSEHASH_DATA_ROOT to tmp_path. All subroots default under it."""
    root = tmp_path / "mousehash_data"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MOUSEHASH_DATA_ROOT", str(root))
    _reset_paths_cache()
    return root
