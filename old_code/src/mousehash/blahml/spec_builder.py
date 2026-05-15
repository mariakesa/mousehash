from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mousehash.blahml.hashing import tool_run_spec_id
from mousehash.blahml.models import BlahManifest, BlahParameter, ResolvedSpec


class SpecResolutionError(ValueError):
    pass


def _coerce(value: Any, p: BlahParameter) -> Any:
    """Best-effort type coercion from string user input."""
    if value is None or not isinstance(value, str):
        return value
    try:
        if p.type == "integer":
            return int(value)
        if p.type == "float":
            return float(value)
        if p.type == "boolean":
            v = value.strip().lower()
            if v in {"true", "1", "yes", "y"}:
                return True
            if v in {"false", "0", "no", "n"}:
                return False
            raise ValueError(f"cannot interpret {value!r} as boolean")
    except ValueError as exc:
        raise SpecResolutionError(
            f"parameter {p.name!r}: {exc}"
        ) from exc
    return value


def _validate_value(p: BlahParameter, value: Any) -> None:
    if p.choices is not None and value not in p.choices:
        raise SpecResolutionError(
            f"parameter {p.name!r}: value {value!r} not in choices {p.choices}"
        )
    if p.range is not None and isinstance(value, (int, float)):
        if p.range.min is not None and value < p.range.min:
            raise SpecResolutionError(
                f"parameter {p.name!r}: value {value} below range.min {p.range.min}"
            )
        if p.range.max is not None and value > p.range.max:
            raise SpecResolutionError(
                f"parameter {p.name!r}: value {value} above range.max {p.range.max}"
            )


def build_resolved_spec(
    manifest: BlahManifest,
    answers: dict[str, Any],
    *,
    manifest_sha256: str,
    question_trace: list[dict[str, Any]] | None = None,
    created_by: str = "agent_dialogue",
) -> ResolvedSpec:
    """Merge manifest defaults with user answers into a ``ResolvedSpec``.

    Required parameters absent from ``answers`` raise ``SpecResolutionError``.
    Optional parameters fall back to their manifest default.
    """
    input_artifacts: dict[str, Any] = {}
    for p in manifest.inputs:
        if p.name in answers:
            value = _coerce(answers[p.name], p)
            _validate_value(p, value)
            input_artifacts[p.name] = value
        elif p.required:
            raise SpecResolutionError(
                f"manifest {manifest.id!r}: required input {p.name!r} not provided"
            )

    parameters: dict[str, Any] = {}
    for p in manifest.parameters:
        if p.name in answers:
            value = _coerce(answers[p.name], p)
            _validate_value(p, value)
            parameters[p.name] = value
        elif p.required:
            raise SpecResolutionError(
                f"manifest {manifest.id!r}: required parameter {p.name!r} not provided"
            )
        elif p.default is not None:
            parameters[p.name] = p.default

    spec_id = tool_run_spec_id(
        tool_id=manifest.id,
        manifest_sha256_hex=manifest_sha256,
        parameters=parameters,
        input_artifacts=input_artifacts,
    )

    return ResolvedSpec(
        tool_run_spec_id=spec_id,
        tool_id=manifest.id,
        manifest_version=manifest.version,
        manifest_sha256=manifest_sha256,
        parameters=parameters,
        input_artifacts=input_artifacts,
        question_trace=question_trace or [],
        created_by=created_by,
        created_at=datetime.now(timezone.utc),
    )
