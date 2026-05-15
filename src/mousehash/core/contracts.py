"""Tool contracts and the readiness check.

A `ToolContract` is the agent-facing declaration of what a MouseHash tool
needs and produces. The readiness engine reads contracts and manifests to
answer "can I run tool X on dataset Y, and if not, what's missing?".

Contracts intentionally live as data (loaded from YAML registries in later
phases) rather than ad-hoc Python checks — that's what lets the agent reason
about the system without executing it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mousehash.core.analysis_view import AnalysisViewKind
from mousehash.core.role_bundle import RoleName


class ToolContract(BaseModel):
    """Declarative description of a tool's role + view requirements."""

    name: str
    family: str
    required_roles: list[RoleName] = Field(default_factory=list)
    optional_roles: list[RoleName] = Field(default_factory=list)
    any_of_roles: list[RoleName] = Field(
        default_factory=list,
        description="At least one of these roles must be satisfied (e.g. stimuli OR conditions OR behavior).",
    )
    consumes_views: dict[str, AnalysisViewKind] = Field(
        default_factory=dict,
        description="Slot name -> required view kind, e.g. {'X': observation_by_feature, 'Y': observation_by_neuron}.",
    )
    produces: list[str] = Field(
        default_factory=list,
        description="Artifact kinds this tool emits, e.g. ['model_artifact', 'metric_table'].",
    )
    allowed_transformations: list[str] = Field(default_factory=list)
    default_validation: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)


class ToolReadiness(BaseModel):
    """Result of checking a tool contract against a manifest."""

    tool_name: str
    runnable: bool
    missing_required_roles: list[RoleName] = Field(default_factory=list)
    unsatisfied_any_of: list[RoleName] = Field(default_factory=list)
    satisfied_roles: list[RoleName] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def check_manifest_satisfies(
    contract: ToolContract,
    manifest: "RoleManifest",  # noqa: F821 — forward ref to avoid circular import at runtime
) -> ToolReadiness:
    """Decide whether `manifest` exposes enough roles to run `contract`.

    Returns a structured `ToolReadiness` rather than raising, because MCP
    consumers want to enumerate "what would I need to do to make this
    runnable?" — not just a yes/no.
    """
    from mousehash.core.manifests import RoleManifest  # local to avoid cycle at import time

    if not isinstance(manifest, RoleManifest):  # defensive — agent-supplied JSON could be anything
        raise TypeError(f"Expected RoleManifest, got {type(manifest).__name__}")

    satisfied = set(manifest.roles.satisfied_roles())
    missing_required = [r for r in contract.required_roles if r not in satisfied]

    unsatisfied_any_of: list[RoleName] = []
    if contract.any_of_roles:
        if not any(r in satisfied for r in contract.any_of_roles):
            unsatisfied_any_of = list(contract.any_of_roles)

    runnable = not missing_required and not unsatisfied_any_of

    return ToolReadiness(
        tool_name=contract.name,
        runnable=runnable,
        missing_required_roles=missing_required,
        unsatisfied_any_of=unsatisfied_any_of,
        satisfied_roles=sorted(satisfied),
    )
