"""MCP wrappers for the logistic-regression decoder + BlahML question walker.

`decoder_next_question` is the BlahML driver: Claude calls it in a loop with
the accumulated answers and presents each returned question to the user
until the manifest is fully resolved. `decode_animate_inanimate` then takes
the resolved parameters and runs the actual fit.

Both follow the standard MCP wrapper shape — primitive args in, plain dict
out — and translate `MouseHashError` subclasses to structured JSON via
`@mcp_safe`.
"""

from __future__ import annotations

import json
from typing import Any

from mousehash.blahml import load_manifest, next_question
from mousehash.mcp.errors import mcp_safe
from mousehash.mcp.views import find_view_by_id
from mousehash.tools.decoders.logistic_decoder import run_logistic_decoder

DECODER_MANIFEST_ID = "allen_logistic_decoder"


def _parse_C_grid(text: str) -> tuple[float, ...]:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if not parts:
        return (0.01, 0.1, 1.0, 10.0, 100.0)
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Could not parse C_grid {text!r} as comma-separated floats: {exc}") from exc


def _serialize_question(pending) -> dict[str, Any]:
    p = pending.parameter
    out: dict[str, Any] = {
        "name": p.name,
        "section": pending.section,
        "type": p.type,
        "required": p.required,
        "ask": p.question.ask,
        "explain": p.question.explain,
    }
    if p.choices is not None:
        out["choices"] = list(p.choices)
    if p.default is not None:
        out["default"] = p.default
    if p.range is not None:
        out["range"] = {"min": p.range.min, "max": p.range.max}
    if p.accepted_artifact_types is not None:
        out["accepted_artifact_types"] = list(p.accepted_artifact_types)
    if p.question.examples:
        out["examples"] = list(p.question.examples)
    if p.question.default_explanation is not None:
        out["default_explanation"] = p.question.default_explanation
    return out


@mcp_safe
def decoder_next_question(answers_json: str = "{}") -> dict[str, Any]:
    """Return the next BlahML question for the logistic-decoder manifest.

    Drives the interactive fit-config conversation. Claude calls this in a
    loop: take the returned `question`, ask the user, add the answer under
    `question.name` in a running answers dict, JSON-encode that dict, and
    pass it back as `answers_json`. When `resolved` is true, every required
    + scientific-intent slot has a value and `decode_animate_inanimate` can
    be called with the collected answers.

    Args:
        answers_json: JSON-encoded dict of `{parameter_name: value}` collected
            so far. Pass `"{}"` to start the conversation.

    Returns:
        {
          manifest_id: "allen_logistic_decoder",
          collected_answers: {...},
          resolved: bool,
          question: {name, section, type, ask, explain, choices?, default?, range?, ...} | null,
        }
    """
    try:
        answers = json.loads(answers_json) if answers_json else {}
    except json.JSONDecodeError as exc:
        return {
            "error": f"answers_json is not valid JSON: {exc}",
            "type": "JSONDecodeError",
            "details": {"input": answers_json},
        }
    if not isinstance(answers, dict):
        return {
            "error": f"answers_json must decode to a JSON object, got {type(answers).__name__}",
            "type": "JSONDecodeError",
            "details": {},
        }

    manifest = load_manifest(DECODER_MANIFEST_ID)
    pending = next_question(manifest, answers)
    return {
        "manifest_id": DECODER_MANIFEST_ID,
        "collected_answers": answers,
        "resolved": pending is None,
        "question": _serialize_question(pending) if pending is not None else None,
    }


@mcp_safe
def decode_animate_inanimate(
    neural_view_id: str,
    labels_view_id: str,
    split_strategy: str = "stratified_kfold",
    k_folds: int = 5,
    holdout_fraction: float = 0.25,
    penalty: str = "l2",
    class_weight: str = "balanced",
    search_hyperparams: bool = False,
    C_grid: str = "0.01,0.1,1,10,100",
    random_state: int = 0,
    n_permutations: int = 0,
) -> dict[str, Any]:
    """Fit a logistic-regression decoder predicting animate/inanimate per image.

    Loads two views by id, transposes the neural matrix so observations are
    images, then runs sklearn LogisticRegression with the requested CV
    scheme. Always refits on full data and persists coefficients.

    Idempotent: same arguments + same source-view lineage hashes -> cache hit.

    Args:
        neural_view_id: OBSERVATION_BY_FEATURE view with `event_probabilities.npy`
            (shape n_neurons x n_images). Produced by `extract_event_responses`.
        labels_view_id: OBSERVATION_BY_FEATURE view with `animate_inanimate.npy`
            (shape n_images). Produced by `extract_vit_features`.
        split_strategy: "loo" | "stratified_kfold" | "holdout".
        k_folds: number of folds when split_strategy="stratified_kfold".
        holdout_fraction: test fraction when split_strategy="holdout".
        penalty: "l1" or "l2".
        class_weight: "none" or "balanced".
        search_hyperparams: nest a GridSearchCV over C_grid inside each fold.
        C_grid: comma-separated floats; only used when search_hyperparams=true.
        random_state: seed for split + permutation.
        n_permutations: 0 to skip; otherwise the number of label-permuted
            re-runs to build a null distribution and report a p-value.

    Returns:
        {view_id, artifact_path, summary, from_cache}. `summary` carries
        cv_accuracy, cv_balanced_accuracy, chance_accuracy, p_value, n_features,
        artifact paths, and so on.
    """
    neural_view = find_view_by_id(neural_view_id)
    labels_view = find_view_by_id(labels_view_id)
    C_grid_t = _parse_C_grid(C_grid)
    output_view, summary = run_logistic_decoder(
        neural_view=neural_view,
        labels_view=labels_view,
        split_strategy=str(split_strategy),
        k_folds=int(k_folds),
        holdout_fraction=float(holdout_fraction),
        penalty=str(penalty),
        class_weight=str(class_weight),
        search_hyperparams=bool(search_hyperparams),
        C_grid=C_grid_t,
        random_state=int(random_state),
        n_permutations=int(n_permutations),
    )
    return {
        "view_id": output_view.view_id,
        "artifact_path": output_view.artifact_path,
        "summary": summary,
        "from_cache": summary.get("from_cache", False),
    }
