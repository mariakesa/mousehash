"""MCP wrappers for manifest loading + readiness inspection.

The contract registry `_CONTRACTS` is the single source of truth for "what
tools does MouseHash know about?". Adding a new analysis tool = add its
`ToolContract` here (or import it from the tool's module).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mousehash.artifacts.paths import manifests_root
from mousehash.core.contracts import ToolContract, check_manifest_satisfies
from mousehash.core.errors import MouseHashError
from mousehash.core.manifests import RoleManifest
from mousehash.mcp.errors import mcp_safe
from mousehash.tools.comparison.group_comparison import GROUP_COMPARISON_CONTRACT
from mousehash.tools.decoders.logistic_decoder import LOGISTIC_DECODER_CONTRACT
from mousehash.tools.factor_models.nmf import NMF_CONTRACT
from mousehash.tools.factor_models.pca import PCA_CONTRACT
from mousehash.tools.scheduling.schedule_comparison import STIMULUS_SCHEDULE_CONTRACT


class ManifestNotFoundError(MouseHashError):
    """Requested manifest_id is not on disk under manifests_root()."""


_CONTRACTS: list[ToolContract] = [
    PCA_CONTRACT,
    NMF_CONTRACT,
    GROUP_COMPARISON_CONTRACT,
    STIMULUS_SCHEDULE_CONTRACT,
    LOGISTIC_DECODER_CONTRACT,
]
_CONTRACTS_BY_NAME: dict[str, ToolContract] = {c.name: c for c in _CONTRACTS}


def _load_manifest(manifest_id: str) -> RoleManifest:
    path = manifests_root() / f"{manifest_id}.yaml"
    if not path.exists():
        raise ManifestNotFoundError(
            f"No manifest with id {manifest_id!r} under {manifests_root()}. "
            "Build one first via allen_build_manifest."
        )
    return RoleManifest.from_yaml(path.read_text(encoding="utf-8"))


@mcp_safe
def get_manifest(manifest_id: str) -> dict[str, Any]:
    """Load the YAML manifest for the given id and return it as JSON.

    Args:
        manifest_id: id returned by allen_build_manifest (or any target's manifest builder).

    Returns:
        Full RoleManifest as a JSON-serializable dict.
    """
    manifest = _load_manifest(manifest_id)
    return manifest.model_dump(mode="json")


@mcp_safe
def list_runnable_tools(manifest_id: str) -> dict[str, Any]:
    """For every registered tool contract, report whether it can run on this manifest.

    Args:
        manifest_id: id of a previously-built role manifest.

    Returns:
        {manifest_id, readiness: [{tool_name, runnable, missing_required_roles,
            unsatisfied_any_of, satisfied_roles, family, produces}, ...]}.
    """
    manifest = _load_manifest(manifest_id)
    readiness = []
    for contract in _CONTRACTS:
        report = check_manifest_satisfies(contract, manifest).model_dump(mode="json")
        report["family"] = contract.family
        report["produces"] = contract.produces
        readiness.append(report)
    return {"manifest_id": manifest_id, "readiness": readiness}


@mcp_safe
def explain_tool_readiness(tool_name: str, manifest_id: str) -> dict[str, Any]:
    """Detailed readiness check for one tool against one manifest.

    Args:
        tool_name: registered tool name (e.g. "run_pca", "run_nmf").
        manifest_id: id of the role manifest to evaluate against.

    Returns:
        {tool_name, runnable, missing_required_roles, unsatisfied_any_of,
         satisfied_roles, contract: {family, required_roles, any_of_roles,
         optional_roles, consumes_views, produces, assumptions, failure_modes}}.
    """
    if tool_name not in _CONTRACTS_BY_NAME:
        return {
            "error": f"Unknown tool {tool_name!r}.",
            "type": "UnknownToolError",
            "details": {"known_tools": sorted(_CONTRACTS_BY_NAME)},
        }
    contract = _CONTRACTS_BY_NAME[tool_name]
    manifest = _load_manifest(manifest_id)
    readiness = check_manifest_satisfies(contract, manifest)
    return {
        **readiness.model_dump(mode="json"),
        "contract": {
            "family": contract.family,
            "required_roles": contract.required_roles,
            "any_of_roles": contract.any_of_roles,
            "optional_roles": contract.optional_roles,
            "consumes_views": {k: v.value for k, v in contract.consumes_views.items()},
            "produces": contract.produces,
            "assumptions": contract.assumptions,
            "failure_modes": contract.failure_modes,
        },
    }


def registered_contracts() -> list[ToolContract]:
    """Public accessor for the contract registry (used by resources.py)."""
    return list(_CONTRACTS)
