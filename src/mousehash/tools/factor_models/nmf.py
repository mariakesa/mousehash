"""NMF as a target-agnostic MouseHash tool.

NMF on softmax probabilities recovers nonnegative additive semantic mixture
structure: each component is an additive theme; each image's score vector
shows how much of each theme it contains.

Input contract: an `observation_by_feature` AnalysisView whose artifact_path
contains a `probabilities.npy` (non-negative). Temperature-smoothing of the
probabilities is supported via the `temperature` parameter.

Output: a tuple `(view, summary)`. The output view is an `OBSERVATION_BY_FEATURE`
view whose features are NMF components. Idempotent via `cached_computation`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mousehash.artifacts.cache import ComputationSpec, cached_computation
from mousehash.artifacts.io import load_npy, save_json, save_npy
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.contracts import ToolContract
from mousehash.core.errors import ViewKindMismatchError

logger = logging.getLogger(__name__)


NMF_CONTRACT = ToolContract(
    name="run_nmf",
    family="factor_models",
    required_roles=["stimuli"],
    optional_roles=["neural_data", "conditions", "metadata"],
    consumes_views={"X": AnalysisViewKind.OBSERVATION_BY_FEATURE},
    produces=["model", "table", "view"],
    allowed_transformations=[
        "extract_vit_features_view",
        "apply_probability_temperature",
    ],
    default_validation=["reconstruction_error_curve"],
    assumptions=[
        "Input matrix is non-negative (e.g. softmax probabilities).",
    ],
    failure_modes=[
        "Sharp probability distributions can yield degenerate components; consider temperature > 1.",
    ],
)


def apply_probability_temperature(
    probabilities: np.ndarray, temperature: float = 1.0, eps: float = 1e-12
) -> np.ndarray:
    """Apply temperature smoothing/sharpening to a probability matrix.

    Equivalent to `softmax(logits / temperature)` if the input came from
    softmax(logits), up to numerical issues. T>1 smooths, T<1 sharpens.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    P = probabilities.astype(np.float64)
    if np.any(P < 0):
        raise ValueError("probabilities must be non-negative")
    P = np.clip(P, eps, None)
    P_temp = P ** (1.0 / temperature)
    row_sums = P_temp.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("temperature-adjusted probabilities have invalid row sums")
    return P_temp / row_sums


def _solve_nmf(X: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Pure-math NMF core. Imports sklearn lazily."""
    try:
        from sklearn.decomposition import NMF
    except ImportError as exc:
        raise ImportError(
            "NMF requires scikit-learn. Install with: pip install -e '.[science]'"
        ) from exc

    nmf = NMF(**kwargs)
    scores = nmf.fit_transform(X).astype(np.float32)
    return {
        "scores": scores,
        "components": nmf.components_.astype(np.float32),
        "reconstruction_err": float(nmf.reconstruction_err_),
        "n_iter": int(nmf.n_iter_),
    }


def run_nmf(
    view: AnalysisView,
    n_components: int = 10,
    init: str = "nndsvda",
    solver: str = "cd",
    beta_loss: str = "frobenius",
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 0,
    alpha_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio: float = 0.0,
    temperature: float = 1.0,
    input_array_name: str = "probabilities",
    label: str | None = None,
) -> tuple[AnalysisView, dict[str, Any]]:
    """Run NMF on an observation_by_feature view; return (output_view, summary).

    Output directory is content-addressed via `cached_computation`: same
    `(view.lineage_hash, n_components, temperature, sklearn args)` -> same
    on-disk location; second call is a cache hit.

    `label` is informational; it does NOT participate in the cache hash.
    """
    if view.kind != AnalysisViewKind.OBSERVATION_BY_FEATURE:
        raise ViewKindMismatchError(
            expected=AnalysisViewKind.OBSERVATION_BY_FEATURE.value,
            got=view.kind.value,
            slot="X",
        )
    if view.artifact_path is None:
        raise ValueError("run_nmf: view has no artifact_path; nothing to load.")

    spec = ComputationSpec(
        family="decompositions",
        scope=view.lineage_hash,
        name="nmf",
        label=label,
        parameters={
            "n_components": int(n_components),
            "init": init,
            "solver": solver,
            "beta_loss": beta_loss,
            "max_iter": int(max_iter),
            "tol": float(tol),
            "random_state": int(random_state),
            "alpha_W": float(alpha_W),
            "alpha_H": float(alpha_H),
            "l1_ratio": float(l1_ratio),
            "temperature": float(temperature),
            "input_array_name": input_array_name,
        },
        input_fingerprints=[view.lineage_hash],
    )

    def _compute(out_dir):
        from pathlib import Path as _P
        in_path = _P(view.artifact_path) / f"{input_array_name}.npy"
        probabilities = load_npy(in_path)
        logger.info("Running NMF on %s (shape=%s, T=%.3f) -> %s",
                    in_path, probabilities.shape, temperature, out_dir)

        if temperature != 1.0:
            X = apply_probability_temperature(probabilities, temperature)
        else:
            X = probabilities.astype(np.float64)

        result = _solve_nmf(
            X,
            n_components=n_components,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
        )
        logger.info(
            "NMF: %d components, reconstruction_err=%.4f, %d iterations",
            n_components, result["reconstruction_err"], result["n_iter"],
        )

        save_npy(out_dir / "scores.npy", result["scores"])
        save_npy(out_dir / "components.npy", result["components"])
        component_stats = {
            "reconstruction_err": result["reconstruction_err"],
            "n_iter": result["n_iter"],
            "temperature": float(temperature),
        }
        save_json(out_dir / "component_stats.json", component_stats)

        output_view = AnalysisView.new(
            kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
            manifest_id=view.manifest_id,
            shape=list(result["scores"].shape),
            axes={
                "observations": view.axes.get("observations", "observations"),
                "features": "nmf_components",
            },
            source_roles=view.source_roles,
            transformation_lineage=[
                f"source:{view.lineage_hash[:12]}",
                f"nmf:n={int(n_components)}",
                f"temperature={float(temperature)}",
                f"input:{input_array_name}",
            ],
            artifact_path=str(out_dir),
            summary={
                "method": "nmf",
                "n_components": int(n_components),
                "reconstruction_err": result["reconstruction_err"],
                "temperature": float(temperature),
            },
        )
        summary = {
            "source_view_id": view.view_id,
            "source_view_lineage_hash": view.lineage_hash,
            "view_id": output_view.view_id,
            "method": "nmf",
            "input_array_name": input_array_name,
            "input_path": str(in_path),
            "n_images": int(result["scores"].shape[0]),
            "n_components": int(n_components),
            "temperature": float(temperature),
            "reconstruction_err": result["reconstruction_err"],
            "n_iter": result["n_iter"],
            "artifacts": {
                "scores": str(out_dir / "scores.npy"),
                "components": str(out_dir / "components.npy"),
                "component_stats": str(out_dir / "component_stats.json"),
            },
            "output_dir": str(out_dir),
        }
        return output_view, summary

    output_view, summary, from_cache = cached_computation(spec, _compute)
    if from_cache:
        logger.info("NMF cache hit (%s) -> %s", spec.hash(), output_view.artifact_path)
    summary["from_cache"] = from_cache
    return output_view, summary
