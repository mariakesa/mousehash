"""NMF as a target-agnostic MouseHash tool.

NMF on softmax probabilities recovers nonnegative additive semantic mixture
structure: each component is an additive theme; each image's score vector
shows how much of each theme it contains.

Input contract: an `observation_by_feature` AnalysisView whose artifact_path
contains a `probabilities.npy` (non-negative). Temperature-smoothing of the
probabilities is supported via the `temperature` parameter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mousehash.artifacts.io import load_npy, save_json, save_npy
from mousehash.artifacts.paths import decompositions_root
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
    decomposition_spec_id: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run NMF on an observation_by_feature view of softmax probabilities."""
    if view.kind != AnalysisViewKind.OBSERVATION_BY_FEATURE:
        raise ViewKindMismatchError(
            expected=AnalysisViewKind.OBSERVATION_BY_FEATURE.value,
            got=view.kind.value,
            slot="X",
        )
    if view.artifact_path is None:
        raise ValueError("run_nmf: view has no artifact_path; nothing to load.")

    spec_id = decomposition_spec_id or f"nmf_{input_array_name}_{n_components}_T{temperature}"
    in_path = Path(view.artifact_path) / f"{input_array_name}.npy"
    probabilities = load_npy(in_path)
    logger.info("Running NMF on %s (shape=%s, T=%.3f)", in_path, probabilities.shape, temperature)

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

    if output_dir is None:
        output_dir = decompositions_root() / view.lineage_hash / spec_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = save_npy(output_dir / "scores.npy", result["scores"])
    components_path = save_npy(output_dir / "components.npy", result["components"])

    component_stats = {
        "reconstruction_err": result["reconstruction_err"],
        "n_iter": result["n_iter"],
        "temperature": float(temperature),
    }
    stats_path = save_json(output_dir / "component_stats.json", component_stats)

    summary = {
        "view_id": view.view_id,
        "view_lineage_hash": view.lineage_hash,
        "decomposition_spec_id": spec_id,
        "method": "nmf",
        "input_array_name": input_array_name,
        "input_path": str(in_path),
        "n_images": int(result["scores"].shape[0]),
        "n_components": int(n_components),
        "temperature": float(temperature),
        "reconstruction_err": result["reconstruction_err"],
        "n_iter": result["n_iter"],
        "artifacts": {
            "scores": str(scores_path),
            "components": str(components_path),
            "component_stats": str(stats_path),
        },
    }
    summary_path = save_json(output_dir / "summary.json", summary)
    summary["summary_path"] = str(summary_path)
    summary["output_dir"] = str(output_dir)
    return summary
