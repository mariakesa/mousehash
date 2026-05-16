"""Error translation for MCP wrappers.

`@mcp_safe` is the outermost decorator on every MCP-facing function. It
catches `MouseHashError` subclasses (the project's structured errors) and
returns a JSON-clean `{"error", "type", "details"}` dict. Other exceptions
bubble through to FastMCP's default handling — they're unexpected and we
want the traceback in the server log.

Agent UX win: agents reason much better against
`{"error": "Required role 'neural_data' is missing.", "type": "RoleMissingError", "details": {...}}`
than against a Python traceback string.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from mousehash.core.errors import MouseHashError


class ViewNotFoundError(MouseHashError):
    """A view_id was not found anywhere in the artifact cache."""


def _error_payload(exc: MouseHashError) -> dict[str, Any]:
    details: dict[str, Any] = {}
    # Pick up attributes set by specific error subclasses (e.g. role_name on
    # RoleMissingError, expected/got/slot on ViewKindMismatchError).
    for attr in ("role_name", "tool_name", "expected", "got", "slot"):
        value = getattr(exc, attr, None)
        if value is not None:
            details[attr] = value
    return {
        "error": str(exc),
        "type": type(exc).__name__,
        "details": details,
    }


def mcp_safe(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool: turn MouseHashError into structured dict, pass others through."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except MouseHashError as exc:
            return _error_payload(exc)
    return wrapper
