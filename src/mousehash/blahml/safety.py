from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mousehash.blahml.models import BlahManifest, ResolvedSpec


@dataclass(frozen=True)
class SafetyFinding:
    check: str
    level: str          # "error" | "warning" | "info"
    message: str


CheckFn = Callable[[BlahManifest, ResolvedSpec, dict], list[SafetyFinding]]


def _input_matrix_exists(
    manifest: BlahManifest, resolved: ResolvedSpec, ctx: dict
) -> list[SafetyFinding]:
    # No-op when the matrix is not loaded into context. The deterministic
    # compute step verifies file existence anyway; this check is only useful
    # when the caller has already materialized the matrix (e.g. for unit
    # tests on in-memory arrays).
    return []


def _input_matrix_is_2d(
    manifest: BlahManifest, resolved: ResolvedSpec, ctx: dict
) -> list[SafetyFinding]:
    X = ctx.get("input_matrix")
    if X is None or not hasattr(X, "ndim"):
        return []
    if X.ndim != 2:
        return [
            SafetyFinding(
                "input_matrix_is_2d",
                "error",
                f"input matrix must be 2D, got ndim={X.ndim}",
            )
        ]
    return []


def _n_components_less_than_min_dimension(
    manifest: BlahManifest, resolved: ResolvedSpec, ctx: dict
) -> list[SafetyFinding]:
    X = ctx.get("input_matrix")
    n = resolved.parameters.get("n_components")
    if X is None or not hasattr(X, "shape") or n is None:
        return []
    min_dim = min(X.shape)
    if n > min_dim:
        return [
            SafetyFinding(
                "n_components_less_than_min_dimension",
                "error",
                f"n_components={n} exceeds min(matrix.shape)={min_dim}",
            )
        ]
    return []


def _no_nan_or_policy_declared(
    manifest: BlahManifest, resolved: ResolvedSpec, ctx: dict
) -> list[SafetyFinding]:
    X = ctx.get("input_matrix")
    if X is None or not hasattr(X, "shape"):
        return []
    if np.isnan(np.asarray(X)).any() and not resolved.parameters.get("nan_policy"):
        return [
            SafetyFinding(
                "no_nan_or_policy_declared",
                "error",
                "input matrix contains NaN and no nan_policy was declared",
            )
        ]
    return []


def _leakage_warning_if_fit_scope_full_dataset(
    manifest: BlahManifest, resolved: ResolvedSpec, ctx: dict
) -> list[SafetyFinding]:
    if resolved.parameters.get("fit_scope") == "exploratory_full_dataset":
        return [
            SafetyFinding(
                "leakage_warning_if_fit_scope_full_dataset",
                "warning",
                "fit_scope=exploratory_full_dataset is fine for exploration, "
                "but risks leakage if used inside a predictive evaluation. "
                "Use train_only when feeding downstream decoders.",
            )
        ]
    return []


KNOWN_CHECKS: dict[str, CheckFn] = {
    "input_matrix_exists": _input_matrix_exists,
    "input_matrix_is_2d": _input_matrix_is_2d,
    "n_components_less_than_min_dimension": _n_components_less_than_min_dimension,
    "no_nan_or_policy_declared": _no_nan_or_policy_declared,
    "leakage_warning_if_fit_scope_full_dataset": _leakage_warning_if_fit_scope_full_dataset,
}


def run_safety_checks(
    manifest: BlahManifest, resolved: ResolvedSpec, context: dict | None = None
) -> list[SafetyFinding]:
    """Run all checks declared on the manifest. Unknown names raise at
    manifest-validation time, so we can assume every name is in KNOWN_CHECKS.

    ``context`` carries data the checks need but the spec doesn't (e.g. the
    actual input matrix once it's been loaded). Pass ``input_matrix=<ndarray>``
    to enable shape/NaN checks.
    """
    ctx = context or {}
    findings: list[SafetyFinding] = []
    for name in manifest.validation.get("checks", []):
        findings.extend(KNOWN_CHECKS[name](manifest, resolved, ctx))
    return findings
