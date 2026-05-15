"""Plain-vanilla file I/O for the artifact store. No env reads here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_npy(path: Path, array: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path


def load_npy(path: Path) -> np.ndarray:
    return np.load(Path(path))


def save_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_json(path: Path) -> Any:
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def save_html(path: Path, html: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def save_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Tiny CSV writer for metric tables. Uses the first row's keys as headers."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
