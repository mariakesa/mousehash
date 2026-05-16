"""Pydantic models backing a BlahML manifest."""

from __future__ import annotations

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
    inputs: list[BlahParameter] = Field(default_factory=list)
    parameters: list[BlahParameter] = Field(default_factory=list)
    validation: dict[str, list[str]] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
