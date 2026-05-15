from __future__ import annotations

import importlib

from mousehash.blahml.models import BlahManifest, BlahParameter


class ManifestValidationError(ValueError):
    pass


def _check_param(p: BlahParameter, *, manifest_id: str) -> None:
    where = f"manifest {manifest_id!r}, parameter {p.name!r}"

    if not p.question.ask.strip():
        raise ManifestValidationError(f"{where}: question.ask is empty")
    if not p.question.explain.strip():
        raise ManifestValidationError(f"{where}: question.explain is empty")

    if p.type == "enum":
        if not p.choices:
            raise ManifestValidationError(f"{where}: enum parameter has no choices")
        if p.default is not None and p.default not in p.choices:
            raise ManifestValidationError(
                f"{where}: default {p.default!r} not in choices {p.choices}"
            )

    if p.type == "artifact_ref" and not p.accepted_artifact_types:
        raise ManifestValidationError(
            f"{where}: artifact_ref parameter must list accepted_artifact_types"
        )

    if p.range and p.default is not None and isinstance(p.default, (int, float)):
        if p.range.min is not None and p.default < p.range.min:
            raise ManifestValidationError(
                f"{where}: default {p.default} below range.min {p.range.min}"
            )
        if p.range.max is not None and p.default > p.range.max:
            raise ManifestValidationError(
                f"{where}: default {p.default} above range.max {p.range.max}"
            )


def _check_python_function(dotted: str, *, manifest_id: str) -> None:
    if "." not in dotted:
        raise ManifestValidationError(
            f"manifest {manifest_id!r}: tool_binding.python_function "
            f"must be a dotted path, got {dotted!r}"
        )
    module_name, _, attr = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ManifestValidationError(
            f"manifest {manifest_id!r}: cannot import {module_name}: {exc}"
        ) from exc
    if not hasattr(module, attr):
        raise ManifestValidationError(
            f"manifest {manifest_id!r}: {module_name} has no attribute {attr!r}"
        )


def validate_manifest_structure(manifest: BlahManifest) -> None:
    """Stricter checks beyond the pydantic schema.

    Run at registry-load time so broken manifests fail loudly at startup
    rather than mid-dialogue.
    """
    for p in manifest.inputs:
        _check_param(p, manifest_id=manifest.id)
    for p in manifest.parameters:
        _check_param(p, manifest_id=manifest.id)

    if "python_function" not in manifest.tool_binding:
        raise ManifestValidationError(
            f"manifest {manifest.id!r}: tool_binding.python_function is required"
        )
    _check_python_function(
        manifest.tool_binding["python_function"], manifest_id=manifest.id
    )

    # All declared validation checks must be known. Imported lazily to avoid
    # a circular import (safety -> models -> validator on package init).
    from mousehash.blahml.safety import KNOWN_CHECKS

    for check_name in manifest.validation.get("checks", []):
        if check_name not in KNOWN_CHECKS:
            raise ManifestValidationError(
                f"manifest {manifest.id!r}: unknown validation check "
                f"{check_name!r}; known: {sorted(KNOWN_CHECKS)}"
            )
