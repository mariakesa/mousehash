"""Read-only MCP resources.

Resources expose state without invoking a tool. Each handler returns a JSON
string (FastMCP wraps it into a resource payload).

URI scheme: `mousehash://<family>/...`.

These handlers do NOT mutate state. Building / fetching that costs anything
goes through a tool, not a resource.
"""

from __future__ import annotations

import json
from typing import Any

from mousehash.artifacts.paths import manifests_root
from mousehash.core.manifests import RoleManifest
from mousehash.mcp.manifest_tools import _CONTRACTS_BY_NAME, _load_manifest
from mousehash.mcp.target_tools import allen_list_datasets


def targets_resource() -> str:
    """List of registered targets and whether their adapter is wired up."""
    return json.dumps({"targets": [{"name": "allen", "available": True}]})


def allen_datasets_resource() -> str:
    """Allen Brain Observatory dataset references currently exposed."""
    return json.dumps(allen_list_datasets())


def manifest_resource(manifest_id: str) -> str:
    """Full role manifest for the given id, as JSON."""
    manifest = _load_manifest(manifest_id)
    return json.dumps(manifest.model_dump(mode="json"))


def tool_contract_resource(tool_name: str) -> str:
    """Declarative ToolContract for a registered tool, as JSON."""
    contract = _CONTRACTS_BY_NAME.get(tool_name)
    if contract is None:
        return json.dumps({
            "error": f"Unknown tool {tool_name!r}",
            "type": "UnknownToolError",
            "details": {"known_tools": sorted(_CONTRACTS_BY_NAME)},
        })
    payload: dict[str, Any] = contract.model_dump(mode="json")
    # Enum kinds -> strings for stable JSON
    payload["consumes_views"] = {k: v.value if hasattr(v, "value") else v
                                  for k, v in contract.consumes_views.items()}
    return json.dumps(payload)
