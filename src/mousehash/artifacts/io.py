from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
