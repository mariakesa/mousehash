from __future__ import annotations

from pathlib import Path

from mousehash.config import DATA_ROOT


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stimuli_root() -> Path:
    return ensure_dir(DATA_ROOT / "stimuli")


def representations_root() -> Path:
    return ensure_dir(DATA_ROOT / "representations")


def decompositions_root() -> Path:
    return ensure_dir(DATA_ROOT / "decompositions")


def reports_root() -> Path:
    return ensure_dir(DATA_ROOT / "reports")