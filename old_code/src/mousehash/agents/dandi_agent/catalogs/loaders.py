"""Catalog loaders for transformations and tools."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from mousehash.agents.dandi_agent.models import ToolSpec, TransformationSpec

CATALOG_DIR = Path(__file__).parent
TRANSFORMATIONS_PATH = CATALOG_DIR / "transformations.yaml"
TOOLS_PATH = CATALOG_DIR / "tools.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


@lru_cache(maxsize=1)
def load_transformations() -> dict[str, TransformationSpec]:
    """Load the transformation catalog as a dict keyed by transformation name."""
    raw = _read_yaml(TRANSFORMATIONS_PATH)
    return {entry["name"]: TransformationSpec.model_validate(entry) for entry in raw["transformations"]}


@lru_cache(maxsize=1)
def load_tools() -> dict[str, ToolSpec]:
    """Load the tool catalog as a dict keyed by tool_id."""
    raw = _read_yaml(TOOLS_PATH)
    return {entry["tool_id"]: ToolSpec.model_validate(entry) for entry in raw["tools"]}


def transformations_catalog_version() -> str:
    return _read_yaml(TRANSFORMATIONS_PATH)["catalog_version"]


def tools_catalog_version() -> str:
    return _read_yaml(TOOLS_PATH)["catalog_version"]
