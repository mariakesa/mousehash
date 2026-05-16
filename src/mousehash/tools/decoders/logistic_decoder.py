"""Logistic-regression decoder for image-labelled neural population activity.

Input contract:
  - `neural_view`: OBSERVATION_BY_FEATURE view shape (n_neurons, n_images),
    artifact dir holds `<neural_array_name>.npy` (default `event_probabilities`).
  - `labels_view`: OBSERVATION_BY_FEATURE view (typically the ViT features
    view), artifact dir holds `<labels_array_name>.npy` of shape (n_images,)
    with integer 0/1 labels (default `animate_inanimate`).

The decoder transposes the neural matrix to (n_images, n_neurons) so rows
are observations and columns are features for sklearn. Labels are taken
verbatim — column index of the neural view is assumed aligned with the
label vector index (it is for the Allen natural-scenes pipeline:
column i = `frame_idx == i` in `get_stimulus_table('natural_scenes')` =
image i in `fetch_natural_scene_template`).

Output: a `METRIC_TABLE` AnalysisView whose `artifact_path` holds:
    spec.json, view.json, summary.json
    cv_predictions.npy        # (n_images,)
    cv_probabilities.npy      # (n_images,)  P(class=1)
    per_fold_accuracy.npy     # (n_folds,)
    final_model_coef.npy      # (n_features,)
    final_model_intercept.npy # scalar
    permutation_accuracies.npy  (only if n_permutations > 0)

Cached via `cached_computation`: identical inputs + parameters -> cache hit.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import numpy as np

from mousehash.artifacts.cache import ComputationSpec, cached_computation
from mousehash.artifacts.io import load_npy, save_npy
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.contracts import ToolContract
from mousehash.core.errors import MouseHashError, ViewKindMismatchError

logger = logging.getLogger(__name__)


LOGISTIC_DECODER_CONTRACT = ToolContract(
    name="decode_animate_inanimate",
    family="decoders",
    required_roles=["neural_data", "stimuli"],
    optional_roles=["metadata"],
    consumes_views={
        "X": AnalysisViewKind.OBSERVATION_BY_FEATURE,
        "labels": AnalysisViewKind.OBSERVATION_BY_FEATURE,
    },
    produces=["model", "metric_table", "view"],
    allowed_transformations=[
        "extract_event_response_view",
        "extract_vit_features_view",
    ],
    default_validation=["cv_accuracy_above_chance"],
    assumptions=[
        "neural_view shape is (n_neurons, n_images); transposed internally to (n_images, n_neurons).",
        "labels are read from <labels_view.artifact_path>/animate_inanimate.npy and aligned by image index.",
    ],
    failure_modes=[
        "Tiny n_images vs huge n_neurons (~40k for Allen) — overfitting is the default; trust CV scores, not training accuracy.",
        "Permutation null is expensive when feature count is large; budget n_permutations carefully.",
    ],
)


class DecoderInputError(MouseHashError):
    """Raised when the neural / labels views don't agree on length or shape."""


def _cv_iterator(
    split_strategy: str,
    y: np.ndarray,
    *,
    k_folds: int,
    holdout_fraction: float,
    random_state: int,
) -> tuple[Iterable[tuple[np.ndarray, np.ndarray]], int]:
    """Return (iterator of (train_idx, test_idx), n_splits)."""
    try:
        from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
    except ImportError as exc:
        raise ImportError(
            "logistic_decoder requires scikit-learn. Install with: pip install -e '.[science]'"
        ) from exc

    n = len(y)
    if split_strategy == "loo":
        loo = LeaveOneOut()
        return list(loo.split(np.zeros((n, 1)), y)), n
    if split_strategy == "stratified_kfold":
        skf = StratifiedKFold(n_splits=int(k_folds), shuffle=True, random_state=int(random_state))
        return list(skf.split(np.zeros((n, 1)), y)), int(k_folds)
    if split_strategy == "holdout":
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(
            indices, test_size=float(holdout_fraction), stratify=y, random_state=int(random_state)
        )
        return [(train_idx, test_idx)], 1
    raise ValueError(f"Unknown split_strategy {split_strategy!r}; expected loo|stratified_kfold|holdout")


def _make_estimator(
    *, penalty: str, class_weight: str, random_state: int, C: float = 1.0,
):
    from sklearn.linear_model import LogisticRegression

    solver = "liblinear" if penalty == "l1" else "lbfgs"
    cw = None if class_weight == "none" else class_weight
    return LogisticRegression(
        penalty=penalty,
        C=float(C),
        class_weight=cw,
        solver=solver,
        max_iter=2000,
        random_state=int(random_state),
    )


def _fit_one_fold(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    *, penalty: str, class_weight: str, random_state: int,
    search_hyperparams: bool, C_grid: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return (predictions, P(class=1), best_C-or-None)."""
    base = _make_estimator(
        penalty=penalty, class_weight=class_weight, random_state=random_state,
    )
    if search_hyperparams:
        from sklearn.model_selection import GridSearchCV
        # Inner CV: use 3-fold or fall back to 2 if any class is too small.
        unique, counts = np.unique(y_train, return_counts=True)
        inner_splits = max(2, min(3, int(counts.min())))
        gs = GridSearchCV(
            base,
            param_grid={"C": list(C_grid)},
            cv=inner_splits,
            scoring="balanced_accuracy",
            n_jobs=1,
        )
        gs.fit(X_train, y_train)
        best_C = float(gs.best_params_["C"])
        preds = gs.predict(X_test)
        proba = gs.predict_proba(X_test)[:, 1]
        return preds, proba, best_C

    base.fit(X_train, y_train)
    preds = base.predict(X_test)
    proba = base.predict_proba(X_test)[:, 1]
    return preds, proba, None


def _run_cv(
    X: np.ndarray, y: np.ndarray,
    *, split_strategy: str, k_folds: int, holdout_fraction: float,
    penalty: str, class_weight: str, random_state: int,
    search_hyperparams: bool, C_grid: Sequence[float],
) -> dict[str, Any]:
    splits, n_splits = _cv_iterator(
        split_strategy, y,
        k_folds=k_folds, holdout_fraction=holdout_fraction, random_state=random_state,
    )
    n_images = X.shape[0]
    cv_predictions = np.full(n_images, -1, dtype=np.int32)
    cv_probabilities = np.full(n_images, np.nan, dtype=np.float32)
    per_fold_accuracy = np.zeros(n_splits, dtype=np.float32)
    best_C_per_fold: list[float] = []
    test_mask = np.zeros(n_images, dtype=bool)
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        preds, proba, best_C = _fit_one_fold(
            X[train_idx], y[train_idx], X[test_idx],
            penalty=penalty, class_weight=class_weight, random_state=random_state,
            search_hyperparams=search_hyperparams, C_grid=C_grid,
        )
        cv_predictions[test_idx] = preds.astype(np.int32)
        cv_probabilities[test_idx] = proba.astype(np.float32)
        per_fold_accuracy[fold_i] = float((preds == y[test_idx]).mean())
        test_mask[test_idx] = True
        if best_C is not None:
            best_C_per_fold.append(best_C)

    return {
        "cv_predictions": cv_predictions,
        "cv_probabilities": cv_probabilities,
        "per_fold_accuracy": per_fold_accuracy,
        "test_mask": test_mask,
        "best_C_per_fold": best_C_per_fold,
    }


def _scoring(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
    )

    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    acc = float(accuracy_score(y_true_m, y_pred_m))
    bacc = float(balanced_accuracy_score(y_true_m, y_pred_m))
    p, r, f, _ = precision_recall_fscore_support(
        y_true_m, y_pred_m, average=None, labels=[0, 1], zero_division=0,
    )
    return {
        "cv_accuracy": acc,
        "cv_balanced_accuracy": bacc,
        "precision_class0": float(p[0]),
        "precision_class1": float(p[1]),
        "recall_class0": float(r[0]),
        "recall_class1": float(r[1]),
        "f1_class0": float(f[0]),
        "f1_class1": float(f[1]),
    }


def _permutation_null(
    X: np.ndarray, y: np.ndarray,
    *, split_strategy: str, k_folds: int, holdout_fraction: float,
    penalty: str, class_weight: str, random_state: int,
    n_permutations: int,
) -> np.ndarray:
    """Re-run CV with permuted labels; return array of per-permutation accuracies."""
    rng = np.random.default_rng(int(random_state))
    accs = np.zeros(int(n_permutations), dtype=np.float32)
    for i in range(int(n_permutations)):
        y_perm = rng.permutation(y)
        res = _run_cv(
            X, y_perm,
            split_strategy=split_strategy, k_folds=k_folds, holdout_fraction=holdout_fraction,
            penalty=penalty, class_weight=class_weight, random_state=int(random_state) + i,
            search_hyperparams=False, C_grid=(1.0,),
        )
        mask = res["test_mask"]
        if mask.any():
            accs[i] = float((res["cv_predictions"][mask] == y_perm[mask]).mean())
    return accs


def run_logistic_decoder(
    neural_view: AnalysisView,
    labels_view: AnalysisView,
    split_strategy: str = "stratified_kfold",
    k_folds: int = 5,
    holdout_fraction: float = 0.25,
    penalty: str = "l2",
    class_weight: str = "balanced",
    search_hyperparams: bool = False,
    C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    random_state: int = 0,
    n_permutations: int = 0,
    neural_array_name: str = "event_probabilities",
    labels_array_name: str = "animate_inanimate",
    label: str | None = None,
) -> tuple[AnalysisView, dict[str, Any]]:
    """Fit a logistic-regression decoder on (neural_view.T, labels_view).

    Returns `(output_view, summary)`. The output view is of kind METRIC_TABLE;
    the summary dict carries CV scores, the optional permutation p-value, and
    the artifact paths.
    """
    if neural_view.kind != AnalysisViewKind.OBSERVATION_BY_FEATURE:
        raise ViewKindMismatchError(
            expected=AnalysisViewKind.OBSERVATION_BY_FEATURE.value,
            got=neural_view.kind.value,
            slot="neural_view",
        )
    if labels_view.kind != AnalysisViewKind.OBSERVATION_BY_FEATURE:
        raise ViewKindMismatchError(
            expected=AnalysisViewKind.OBSERVATION_BY_FEATURE.value,
            got=labels_view.kind.value,
            slot="labels_view",
        )
    if neural_view.artifact_path is None or labels_view.artifact_path is None:
        raise DecoderInputError("Both neural_view and labels_view must have artifact_path set.")

    from pathlib import Path as _P
    X_neurons_by_images = load_npy(_P(neural_view.artifact_path) / f"{neural_array_name}.npy")
    y = load_npy(_P(labels_view.artifact_path) / f"{labels_array_name}.npy")

    if X_neurons_by_images.ndim != 2:
        raise DecoderInputError(
            f"Expected neural array to be 2-D, got shape {X_neurons_by_images.shape}"
        )
    if y.ndim != 1:
        raise DecoderInputError(f"Expected labels to be 1-D, got shape {y.shape}")

    X = X_neurons_by_images.T.astype(np.float64)  # (n_images, n_neurons)
    y = y.astype(np.int64)
    if X.shape[0] != y.shape[0]:
        raise DecoderInputError(
            f"Neural-view n_images ({X.shape[0]}) != labels length ({y.shape[0]}); "
            "column index of the neural view must align with the label vector index."
        )

    classes, counts = np.unique(y, return_counts=True)
    if set(classes.tolist()) - {0, 1}:
        raise DecoderInputError(f"Labels must be binary 0/1; got classes {classes.tolist()}.")

    C_grid_t = tuple(float(c) for c in C_grid)

    spec = ComputationSpec(
        family="decoders",
        scope=neural_view.lineage_hash,
        name="logistic_animate_inanimate",
        label=label,
        parameters={
            "split_strategy": split_strategy,
            "k_folds": int(k_folds),
            "holdout_fraction": float(holdout_fraction),
            "penalty": penalty,
            "class_weight": class_weight,
            "search_hyperparams": bool(search_hyperparams),
            "C_grid": list(C_grid_t),
            "random_state": int(random_state),
            "n_permutations": int(n_permutations),
            "neural_array_name": neural_array_name,
            "labels_array_name": labels_array_name,
            "labels_lineage": labels_view.lineage_hash,
        },
        input_fingerprints=[neural_view.lineage_hash, labels_view.lineage_hash],
    )

    def _compute(out_dir):
        logger.info(
            "Decoder: X shape=%s y shape=%s split=%s search=%s n_perm=%d",
            X.shape, y.shape, split_strategy, search_hyperparams, n_permutations,
        )
        cv = _run_cv(
            X, y,
            split_strategy=split_strategy, k_folds=int(k_folds), holdout_fraction=float(holdout_fraction),
            penalty=penalty, class_weight=class_weight, random_state=int(random_state),
            search_hyperparams=bool(search_hyperparams), C_grid=C_grid_t,
        )
        scores = _scoring(y, cv["cv_predictions"], cv["test_mask"])

        save_npy(out_dir / "cv_predictions.npy", cv["cv_predictions"])
        save_npy(out_dir / "cv_probabilities.npy", cv["cv_probabilities"])
        save_npy(out_dir / "per_fold_accuracy.npy", cv["per_fold_accuracy"])

        # Always refit on full data for coefficient interpretation.
        final = _make_estimator(
            penalty=penalty, class_weight=class_weight, random_state=int(random_state),
        )
        final.fit(X, y)
        coef = final.coef_.astype(np.float32).reshape(-1)
        save_npy(out_dir / "final_model_coef.npy", coef)
        save_npy(out_dir / "final_model_intercept.npy", final.intercept_.astype(np.float32))

        chance = float(max(counts) / counts.sum())

        perm_block: dict[str, Any] = {}
        if int(n_permutations) > 0:
            logger.info("Permutation null: %d shuffles", int(n_permutations))
            perm_accs = _permutation_null(
                X, y,
                split_strategy=split_strategy, k_folds=int(k_folds), holdout_fraction=float(holdout_fraction),
                penalty=penalty, class_weight=class_weight, random_state=int(random_state),
                n_permutations=int(n_permutations),
            )
            save_npy(out_dir / "permutation_accuracies.npy", perm_accs)
            p_value = float((1 + int(np.sum(perm_accs >= scores["cv_accuracy"]))) / (1 + int(n_permutations)))
            perm_block = {
                "p_value": p_value,
                "permutation_mean": float(perm_accs.mean()),
                "permutation_std": float(perm_accs.std()),
                "permutation_path": str(out_dir / "permutation_accuracies.npy"),
            }
        else:
            perm_block = {"p_value": None}

        best_C_mean = (
            float(np.mean(cv["best_C_per_fold"])) if cv["best_C_per_fold"] else None
        )

        summary: dict[str, Any] = {
            "method": "logistic_regression",
            "split_strategy": split_strategy,
            "k_folds": int(k_folds) if split_strategy == "stratified_kfold" else None,
            "holdout_fraction": float(holdout_fraction) if split_strategy == "holdout" else None,
            "penalty": penalty,
            "class_weight": class_weight,
            "search_hyperparams": bool(search_hyperparams),
            "n_features": int(X.shape[1]),
            "n_images": int(X.shape[0]),
            "n_class0": int(counts[classes.tolist().index(0)]) if 0 in classes.tolist() else 0,
            "n_class1": int(counts[classes.tolist().index(1)]) if 1 in classes.tolist() else 0,
            "chance_accuracy": chance,
            "best_C_mean": best_C_mean,
            **scores,
            **perm_block,
            "source_neural_view_id": neural_view.view_id,
            "source_labels_view_id": labels_view.view_id,
            "artifacts": {
                "cv_predictions": str(out_dir / "cv_predictions.npy"),
                "cv_probabilities": str(out_dir / "cv_probabilities.npy"),
                "per_fold_accuracy": str(out_dir / "per_fold_accuracy.npy"),
                "final_model_coef": str(out_dir / "final_model_coef.npy"),
                "final_model_intercept": str(out_dir / "final_model_intercept.npy"),
            },
            "output_dir": str(out_dir),
        }
        if int(n_permutations) > 0:
            summary["artifacts"]["permutation_accuracies"] = str(out_dir / "permutation_accuracies.npy")

        output_view = AnalysisView.new(
            kind=AnalysisViewKind.METRIC_TABLE,
            manifest_id=neural_view.manifest_id,
            shape=[1, 12],
            axes={"observations": "summary", "features": "metrics"},
            source_roles=neural_view.source_roles,
            transformation_lineage=[
                f"neural:{neural_view.lineage_hash[:12]}",
                f"labels:{labels_view.lineage_hash[:12]}",
                f"logreg:{penalty}",
                f"cv:{split_strategy}",
                f"search:{bool(search_hyperparams)}",
                f"perm:{int(n_permutations)}",
            ],
            artifact_path=str(out_dir),
            summary={
                "method": "logistic_regression",
                "cv_accuracy": scores["cv_accuracy"],
                "cv_balanced_accuracy": scores["cv_balanced_accuracy"],
                "chance_accuracy": chance,
                "p_value": perm_block.get("p_value"),
            },
        )
        summary["view_id"] = output_view.view_id
        return output_view, summary

    output_view, summary, from_cache = cached_computation(spec, _compute)
    if from_cache:
        logger.info("Decoder cache hit (%s) -> %s", spec.hash(), output_view.artifact_path)
    summary["from_cache"] = from_cache
    return output_view, summary
