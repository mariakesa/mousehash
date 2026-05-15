"""Exception hierarchy.

Errors raised by the MouseHash core are deliberately specific so MCP tools can
translate them into structured failure responses instead of leaking tracebacks
to agents.
"""

from __future__ import annotations


class MouseHashError(Exception):
    """Base class for all MouseHash-raised errors."""


class RoleMissingError(MouseHashError):
    """A required role is absent from the supplied manifest."""

    def __init__(self, role_name: str, tool_name: str | None = None) -> None:
        self.role_name = role_name
        self.tool_name = tool_name
        suffix = f" for tool '{tool_name}'" if tool_name else ""
        super().__init__(f"Required role '{role_name}' is missing{suffix}.")


class ViewKindMismatchError(MouseHashError):
    """A tool received an AnalysisView whose kind does not match its contract."""

    def __init__(self, expected: str, got: str, slot: str | None = None) -> None:
        self.expected = expected
        self.got = got
        self.slot = slot
        where = f" in slot '{slot}'" if slot else ""
        super().__init__(f"Expected view kind '{expected}'{where}, got '{got}'.")


class ContractViolationError(MouseHashError):
    """A tool run violated its declared contract."""


class ManifestParseError(MouseHashError):
    """A target adapter could not parse its dataset into a role manifest."""
