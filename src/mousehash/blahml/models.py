from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ParamType = Literal["integer", "float", "boolean", "string", "enum", "artifact_ref"]


class BlahQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ask: str
    explain: str
    examples: list[str] = Field(default_factory=list)
    default_explanation: str | None = None


class BlahRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | int | None = None
    max: float | int | None = None


class BlahParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: ParamType
    required: bool = False
    default: Any | None = None
    choices: list[Any] | None = None
    range: BlahRange | None = None
    accepted_artifact_types: list[str] | None = None
    description: str | None = None
    question: BlahQuestion


class BlahManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    display_name: str
    workflow_family: str
    priority: Literal["MVP", "soon", "later", "experimental"]
    description: dict[str, str]
    tool_binding: dict[str, str]
    datajoint: dict[str, str] | None = None
    inputs: list[BlahParameter] = Field(default_factory=list)
    parameters: list[BlahParameter] = Field(default_factory=list)
    validation: dict[str, list[str]] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)


class ResolvedSpec(BaseModel):
    """A fully resolved tool invocation built from a manifest plus user answers.

    ``tool_run_spec_id`` is a content hash so identical resolutions are
    idempotent across DataJoint inserts. ``manifest_sha256`` records the
    exact YAML this run was resolved against, so later audits can detect
    manifest drift.
    """

    model_config = ConfigDict(extra="forbid")

    tool_run_spec_id: str
    tool_id: str
    manifest_version: str
    manifest_sha256: str
    parameters: dict[str, Any]
    input_artifacts: dict[str, Any]
    question_trace: list[dict[str, Any]] = Field(default_factory=list)
    created_by: str
    created_at: datetime
