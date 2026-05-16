"""Tests for mcp/errors.py — the mcp_safe decorator + ViewNotFoundError."""

from __future__ import annotations

import pytest

from mousehash.artifacts.paths import MissingEnvError
from mousehash.core.errors import (
    ContractViolationError,
    MouseHashError,
    RoleMissingError,
    ViewKindMismatchError,
)
from mousehash.mcp.errors import ViewNotFoundError, mcp_safe


class TestMcpSafe:
    def test_success_passes_value_through(self):
        @mcp_safe
        def f(x: int) -> dict:
            return {"value": x}
        assert f(7) == {"value": 7}

    def test_role_missing_translated(self):
        @mcp_safe
        def f() -> dict:
            raise RoleMissingError("neural_data", tool_name="run_pca")
        result = f()
        assert result["type"] == "RoleMissingError"
        assert "neural_data" in result["error"]
        assert result["details"]["role_name"] == "neural_data"
        assert result["details"]["tool_name"] == "run_pca"

    def test_view_kind_mismatch_translated(self):
        @mcp_safe
        def f() -> dict:
            raise ViewKindMismatchError(expected="rdm", got="observation_by_neuron", slot="X")
        result = f()
        assert result["type"] == "ViewKindMismatchError"
        assert result["details"]["expected"] == "rdm"
        assert result["details"]["got"] == "observation_by_neuron"
        assert result["details"]["slot"] == "X"

    def test_view_not_found_translated(self):
        @mcp_safe
        def f() -> dict:
            raise ViewNotFoundError("view_zzz not found")
        result = f()
        assert result["type"] == "ViewNotFoundError"
        assert "view_zzz" in result["error"]

    def test_missing_env_translated(self):
        @mcp_safe
        def f() -> dict:
            raise MissingEnvError("MOUSEHASH_DATA_ROOT not set")
        result = f()
        assert result["type"] == "MissingEnvError"

    def test_contract_violation_translated(self):
        @mcp_safe
        def f() -> dict:
            raise ContractViolationError("contract broken")
        result = f()
        assert result["type"] == "ContractViolationError"

    def test_non_mousehash_exceptions_bubble(self):
        @mcp_safe
        def f() -> dict:
            raise ValueError("not a mousehash error")
        with pytest.raises(ValueError, match="not a mousehash error"):
            f()

    def test_keyword_args_forwarded(self):
        @mcp_safe
        def f(*, name: str, n: int) -> dict:
            return {"name": name, "n": n}
        assert f(name="x", n=3) == {"name": "x", "n": 3}

    def test_view_not_found_subclasses_mousehash_error(self):
        assert issubclass(ViewNotFoundError, MouseHashError)
