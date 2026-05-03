from __future__ import annotations

import json
import logging
from typing import Any

from mousehash.blahml.models import ResolvedSpec
from mousehash.blahml.safety import SafetyFinding, run_safety_checks
from mousehash.blahml.registry import ManifestRegistry

logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    pass


# Parameters that BlahML carries for scientific framing but the deterministic
# tool function does not accept as a kwarg. We strip these before dispatch.
_NON_TOOL_PARAMS: dict[str, set[str]] = {
    "run_pca": {"fit_scope"},
    "run_nmf": {"fit_scope"},
}

# Mapping of BlahML parameter names to DecompositionSpec column names.
# (Today they line up 1:1 for the columns we set; this dict is the place
# to add divergent mappings if a manifest renames a slot.)
_PCA_PARAM_TO_COLUMN = {
    "n_components": "n_components",
    "normalize_input": "normalize_input",
}
_NMF_PARAM_TO_COLUMN = {
    "n_components": "n_components",
    "init": "nmf_init",
    "solver": "nmf_solver",
    "beta_loss": "nmf_beta_loss",
    "max_iter": "nmf_max_iter",
    "tol": "nmf_tol",
    "random_state": "nmf_random_state",
    "alpha_W": "nmf_alpha_w",
    "alpha_H": "nmf_alpha_h",
    "l1_ratio": "nmf_l1_ratio",
    "temperature": "nmf_temperature",
}


def _persist_tool_run_spec(resolved: ResolvedSpec) -> None:
    from mousehash.schema.blahml import ToolRunSpec

    ToolRunSpec.insert1(
        dict(
            tool_run_spec_id=resolved.tool_run_spec_id,
            tool_id=resolved.tool_id,
            manifest_version=resolved.manifest_version,
            manifest_sha256=resolved.manifest_sha256,
            parameters_json=json.dumps(resolved.parameters, sort_keys=True),
            input_artifacts_json=json.dumps(resolved.input_artifacts, sort_keys=True),
            question_trace_json=json.dumps(resolved.question_trace, sort_keys=True),
            created_at=resolved.created_at.replace(tzinfo=None),
            created_by=resolved.created_by,
        ),
        skip_duplicates=True,
    )


def _bridge_decomposition_spec(resolved: ResolvedSpec) -> str:
    """Compile resolved.parameters into a DecompositionSpec row and return
    its primary key. Idempotent via skip_duplicates."""
    from mousehash.schema.decompositions import DecompositionSpec

    decomposition_spec_id = f"blahml_{resolved.tool_run_spec_id}"

    if resolved.tool_id == "run_pca":
        row = dict(
            decomposition_spec_id=decomposition_spec_id,
            method="pca",
            input_kind="logits",
            n_components=int(resolved.parameters[_PCA_PARAM_TO_COLUMN["n_components"]]),
            normalize_input=bool(
                resolved.parameters[_PCA_PARAM_TO_COLUMN["normalize_input"]]
            ),
            mode=resolved.parameters.get("fit_scope", "exploratory_full_dataset"),
        )
    elif resolved.tool_id == "run_nmf":
        row = dict(
            decomposition_spec_id=decomposition_spec_id,
            method="nmf",
            input_kind="probabilities",
            n_components=int(resolved.parameters["n_components"]),
            normalize_input=False,
            mode=resolved.parameters.get("fit_scope", "exploratory_full_dataset"),
        )
        for blahml_name, column in _NMF_PARAM_TO_COLUMN.items():
            if blahml_name == "n_components":
                continue
            if blahml_name in resolved.parameters:
                row[column] = resolved.parameters[blahml_name]
    else:
        raise ExecutionError(
            f"Executor has no DecompositionSpec bridge for tool {resolved.tool_id!r}"
        )

    DecompositionSpec.insert1(row, skip_duplicates=True)
    return decomposition_spec_id


def _resolve_artifact_keys(resolved: ResolvedSpec) -> dict[str, str]:
    """Pull (scene_set_id, representation_spec_id, rule_id) out of the
    resolved input_artifacts. We accept either a single ``input_artifact``
    dict with those keys, or three top-level keys."""
    ia = resolved.input_artifacts

    if "input_artifact" in ia and isinstance(ia["input_artifact"], dict):
        ref = ia["input_artifact"]
    else:
        ref = ia

    missing = [
        k for k in ("scene_set_id", "representation_spec_id", "rule_id") if k not in ref
    ]
    if missing:
        raise ExecutionError(
            f"input_artifact is missing required keys: {missing}. "
            f"Got: {ref!r}"
        )
    return {k: ref[k] for k in ("scene_set_id", "representation_spec_id", "rule_id")}


def run_from_resolved_spec(
    resolved: ResolvedSpec,
    *,
    registry: ManifestRegistry | None = None,
    safety_context: dict[str, Any] | None = None,
) -> dict:
    """Persist the audit row and dispatch the deterministic tool.

    Returns a summary dict with the ``compute_stimulus_decomposition``
    payload plus ``tool_run_spec_id`` so the agent can cite it.
    """
    registry = registry or ManifestRegistry()
    manifest = registry.by_id(resolved.tool_id).manifest

    findings = run_safety_checks(manifest, resolved, safety_context)
    errors = [f for f in findings if f.level == "error"]
    if errors:
        raise ExecutionError(
            "BlahML safety checks failed:\n"
            + "\n".join(f"  [{f.check}] {f.message}" for f in errors)
        )

    _persist_tool_run_spec(resolved)

    if resolved.tool_id in {"run_pca", "run_nmf"}:
        from mousehash.tools.decompositions.compute import compute_stimulus_decomposition

        decomposition_spec_id = _bridge_decomposition_spec(resolved)
        artifact_keys = _resolve_artifact_keys(resolved)

        summary = compute_stimulus_decomposition(
            scene_set_id=artifact_keys["scene_set_id"],
            representation_spec_id=artifact_keys["representation_spec_id"],
            rule_id=artifact_keys["rule_id"],
            decomposition_spec_id=decomposition_spec_id,
        )
    else:
        raise ExecutionError(
            f"No execution path wired for tool {resolved.tool_id!r}"
        )

    return dict(
        tool_run_spec_id=resolved.tool_run_spec_id,
        tool_id=resolved.tool_id,
        decomposition_spec_id=decomposition_spec_id,
        warnings=[
            dict(check=f.check, message=f.message)
            for f in findings
            if f.level == "warning"
        ],
        summary=summary,
    )
