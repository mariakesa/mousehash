"""PCA as a target-agnostic MouseHash tool.

Input contract: an `observation_by_feature` AnalysisView. The view's
artifact_path must contain a `logits.npy` (or the caller passes
input_array_name='probabilities' to use the softmax outputs).

Output: scores + components + per-component stats on disk, plus a summary
dict the caller / MCP server can pass straight through.
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


PCA_CONTRACT = ToolContract(
    name="run_pca",
    family="factor_models",
    required_roles=["stimuli"],
    optional_roles=["neural_data", "conditions", "metadata"],
    consumes_views={"X": AnalysisViewKind.OBSERVATION_BY_FEATURE},
    produces=["model", "table", "view"],
    allowed_transformations=[
        "extract_vit_features_view",
        "zscore_features",
    ],
    default_validation=["explained_variance_curve"],
    assumptions=[
        "Input feature matrix is float; normalization (z-score) is applied per-feature when normalize_input=True.",
    ],
    failure_modes=[
        "Logits dominated by a few classes can collapse variance into PC1; consider z-scoring or using probabilities.",
    ],
)


def _solve_pca(X: np.ndarray, n_components: int, normalize_input: bool) -> dict[str, Any]:
    """Pure-math PCA core. Imports sklearn lazily."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "PCA requires scikit-learn. Install with: pip install -e '.[science]'"
        ) from exc

    X = X.astype(np.float64)
    if normalize_input:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logger.info("Normalized features: mean=%.3f std=%.3f", scaler.mean_.mean(), scaler.scale_.mean())

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X).astype(np.float32)

    logger.info(
        "PCA: %d components explain %.1f%% variance",
        n_components,
        pca.explained_variance_ratio_.sum() * 100,
    )
    return {
        "scores": scores,
        "components": pca.components_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "singular_values": pca.singular_values_,
        "mean": pca.mean_.astype(np.float32),
    }


def run_pca(
    view: AnalysisView,
    n_components: int = 10,
    normalize_input: bool = True,
    input_array_name: str = "logits",
    decomposition_spec_id: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run PCA on an observation_by_feature view.

    Args:
        view: AnalysisView of kind OBSERVATION_BY_FEATURE. `view.artifact_path`
              must be a directory containing `<input_array_name>.npy`.
        n_components: number of PCs to retain.
        normalize_input: z-score features before fitting (recommended for logits).
        input_array_name: which .npy file under `artifact_path` to load.
        decomposition_spec_id: scoping id for the output directory. If None,
              derives one from the view's lineage_hash.
        output_dir: explicit output directory override. If None, derives one
              under `decompositions_root()`.

    Returns:
        dict with summary metadata + artifact paths.
    """
    if view.kind != AnalysisViewKind.OBSERVATION_BY_FEATURE:
        raise ViewKindMismatchError(
            expected=AnalysisViewKind.OBSERVATION_BY_FEATURE.value,
            got=view.kind.value,
            slot="X",
        )
    if view.artifact_path is None:
        raise ValueError("run_pca: view has no artifact_path; nothing to load.")

    spec_id = decomposition_spec_id or f"pca_{input_array_name}_{n_components}"
    in_path = Path(view.artifact_path) / f"{input_array_name}.npy"
    X = load_npy(in_path)
    logger.info("Running PCA on %s (shape=%s)", in_path, X.shape)

    result = _solve_pca(X, n_components=n_components, normalize_input=normalize_input)

    if output_dir is None:
        # Scope outputs under decompositions_root using the view's lineage
        output_dir = decompositions_root() / view.lineage_hash / spec_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = save_npy(output_dir / "scores.npy", result["scores"])
    components_path = save_npy(output_dir / "components.npy", result["components"])

    component_stats = {
        "explained_variance_ratio": result["explained_variance_ratio"].tolist(),
        "singular_values": result["singular_values"].tolist(),
        "cumulative_variance": np.cumsum(result["explained_variance_ratio"]).tolist(),
    }
    stats_path = save_json(output_dir / "component_stats.json", component_stats)

    summary = {
        "view_id": view.view_id,
        "view_lineage_hash": view.lineage_hash,
        "decomposition_spec_id": spec_id,
        "method": "pca",
        "input_array_name": input_array_name,
        "input_path": str(in_path),
        "n_images": int(result["scores"].shape[0]),
        "n_components": int(n_components),
        "normalize_input": bool(normalize_input),
        "explained_variance_ratio_total": float(result["explained_variance_ratio"].sum()),
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
