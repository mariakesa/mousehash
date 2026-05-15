"""Artifact: the typed record of "what a tool produced".

Tools emit artifacts; provenance points at them. This module defines the
metadata type only. Actual I/O (writing arrays, figures, HTML reports to
disk or object storage) lives under `mousehash.artifacts/` in a later phase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from mousehash.core.ids import ArtifactId, ToolRunId


class ArtifactKind(str, Enum):
    MODEL = "model"
    TABLE = "table"
    FIGURE = "figure"
    HTML = "html"
    BUNDLE = "bundle"
    VIEW = "view"
    REPORT = "report"


class Artifact(BaseModel):
    """Metadata record for a single produced artifact."""

    artifact_id: ArtifactId
    kind: ArtifactKind
    tool_run_id: ToolRunId | None = Field(
        default=None,
        description="The tool run that produced this artifact. None for adapter-emitted resources.",
    )
    path: str = Field(description="Filesystem path or URI to the materialized artifact.")
    content_hash: str | None = Field(
        default=None,
        description="Hash of the artifact bytes for integrity checks.",
    )
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Small JSON-safe summary the agent can read without loading the artifact itself.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
