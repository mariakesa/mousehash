"""PCA as a target-agnostic MouseHash tool.

Input contract: an `observation_by_feature` AnalysisView. The view's
artifact_path must contain a `<input_array_name>.npy` (default `logits.npy`).

Output: a tuple `(view, summary)`. The view is an `OBSERVATION_BY_FEATURE`
AnalysisView whose features are PCA components and whose `artifact_path`
points at the cached output directory. The summary dict preserves the keys
the old shape carried (artifact paths, explained-variance totals, etc.) for
downstream report consumers.

This tool is on the `cached_computation` pattern: identical (source view,
n_components, normalize_input, input_array_name) -> identical view_id, second
call is a cache hit.
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
    label: str | None = None,
) -> tuple[AnalysisView, dict[str, Any]]:
    """Run PCA on an observation_by_feature view; return (output_view, summary).

    Output directory is content-addressed via `cached_computation`: same
    `(view.lineage_hash, n_components, normalize_input, input_array_name)`
    -> same on-disk location; second call is a cache hit.

    Args:
        view: source view of kind OBSERVATION_BY_FEATURE. `view.artifact_path`
              must contain `<input_array_name>.npy`.
        n_components: number of PCs to retain.
        normalize_input: z-score features before fitting (recommended for logits).
        input_array_name: which .npy file under `artifact_path` to load.
        label: informational tag; does NOT affect cache key.

    Returns:
        output_view: AnalysisView of kind OBSERVATION_BY_FEATURE whose features
                     are PCA components and whose `artifact_path` is the cache
                     directory containing `scores.npy`, `components.npy`, and
                     `component_stats.json`.
        summary:    dict with explained-variance totals + artifact paths +
                    cache flag.
    """
    if view.kind != AnalysisViewKind.OBSERVATION_BY_FEATURE:
        raise ViewKindMismatchError(
            expected=AnalysisViewKind.OBSERVATION_BY_FEATURE.value,
            got=view.kind.value,
            slot="X",
        )
    if view.artifact_path is None:
        raise ValueError("run_pca: view has no artifact_path; nothing to load.")

    spec = ComputationSpec(
        family="decompositions",
        scope=view.lineage_hash,
        name="pca",
        label=label,
        parameters={
            "n_components": int(n_components),
            "normalize_input": bool(normalize_input),
            "input_array_name": input_array_name,
        },
        input_fingerprints=[view.lineage_hash],
    )

    def _compute(out_dir):
        in_path = view.artifact_path
        from pathlib import Path as _P
        X_path = _P(in_path) / f"{input_array_name}.npy"
        X = load_npy(X_path)
        logger.info("Running PCA on %s (shape=%s) -> %s", X_path, X.shape, out_dir)

        result = _solve_pca(X, n_components=n_components, normalize_input=normalize_input)
        save_npy(out_dir / "scores.npy", result["scores"])
        save_npy(out_dir / "components.npy", result["components"])
        component_stats = {
            "explained_variance_ratio": result["explained_variance_ratio"].tolist(),
            "singular_values": result["singular_values"].tolist(),
            "cumulative_variance": np.cumsum(result["explained_variance_ratio"]).tolist(),
        }
        save_json(out_dir / "component_stats.json", component_stats)

        output_view = AnalysisView.new(
            kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
            manifest_id=view.manifest_id,
            shape=list(result["scores"].shape),
            axes={
                "observations": view.axes.get("observations", "observations"),
                "features": "pca_components",
            },
            source_roles=view.source_roles,
            transformation_lineage=[
                f"source:{view.lineage_hash[:12]}",
                f"pca:n={int(n_components)}",
                f"normalize:{bool(normalize_input)}",
                f"input:{input_array_name}",
            ],
            artifact_path=str(out_dir),
            summary={
                "method": "pca",
                "n_components": int(n_components),
                "explained_variance_ratio_total": float(result["explained_variance_ratio"].sum()),
            },
        )
        summary = {
            "source_view_id": view.view_id,
            "source_view_lineage_hash": view.lineage_hash,
            "view_id": output_view.view_id,
            "method": "pca",
            "input_array_name": input_array_name,
            "input_path": str(X_path),
            "n_images": int(result["scores"].shape[0]),
            "n_components": int(n_components),
            "normalize_input": bool(normalize_input),
            "explained_variance_ratio_total": float(result["explained_variance_ratio"].sum()),
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
        logger.info("PCA cache hit (%s) -> %s", spec.hash(), output_view.artifact_path)
    summary["from_cache"] = from_cache
    return output_view, summary
