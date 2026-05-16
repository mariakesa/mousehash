"""Two-group statistical comparison of any (n, k) feature matrix by a (n,) label vector.

Target-agnostic. The cached wrapper `compare_jpeg_animate_inanimate_views` is
a specific specialization for the JPEG-sizes × animate/inanimate question, but
the math (`compare_groups_by_label`) works on any feature matrix + binary label.

Stats per feature: Welch's t, Mann-Whitney U, Cohen's d. Bonferroni-corrected
min-p across features. Sample-size guard (NaN out features where either group
has < 3 samples).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from mousehash.artifacts.cache import ComputationSpec, cached_computation
from mousehash.artifacts.io import load_npy, save_json
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.contracts import ToolContract

logger = logging.getLogger(__name__)

_MIN_GROUP_SIZE = 3


GROUP_COMPARISON_CONTRACT = ToolContract(
    name="compare_groups_by_label",
    family="comparison",
    required_roles=["stimuli"],
    any_of_roles=["conditions", "behavior", "metadata"],
    consumes_views={"features": AnalysisViewKind.OBSERVATION_BY_FEATURE},
    produces=["metric_table", "figure"],
    allowed_transformations=[
        "extract_jpeg_size_view",
        "extract_vit_features_view",
    ],
    default_validation=["welch_t", "mann_whitney_u", "cohens_d", "bonferroni"],
    assumptions=[
        "features and labels are aligned per-observation (row i in features corresponds to label i).",
        "labels are binary (0 / 1); other integer codings work but only the two extremes are compared.",
    ],
    failure_modes=[
        "Tiny group sizes (< 3 per feature) make per-feature p-values unstable; results are NaN'd out.",
        "Heavy outliers in one group can drag Welch's t-test; Mann-Whitney U is the robust backup.",
    ],
)


# ---------- Pure math ----------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation (Hedges-style denominator).

    Returns 0.0 when both groups have variance 0 (identical) and NaN when
    sample sizes are too small to estimate variance.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0.0:
        return 0.0 if mean_a == mean_b else float("inf") * (1 if mean_a > mean_b else -1)
    return (mean_a - mean_b) / pooled


def compare_groups_by_label(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: tuple[str, str] = ("group_a", "group_b"),
    feature_names: list[str] | None = None,
    units: str = "",
    feature_axis: str = "features",
) -> dict[str, Any]:
    """Per-feature Welch's t + Mann-Whitney U + Cohen's d between two label groups.

    Args:
        features: (n, k) numeric feature matrix.
        labels: (n,) array; 1 -> label_names[0], 0 -> label_names[1].
        label_names: ("group_a_name", "group_b_name"). Group A is `labels == 1`.
        feature_names: optional per-feature labels (length k). Defaults to "f{i}".
        units: short unit string for the summary (e.g. "kB", "bytes").
        feature_axis: human axis name for the plot (e.g. "jpeg_quality_levels").

    Returns the comparison result dict described in the plan.
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError as exc:
        raise ImportError(
            "scipy is required for group comparison. Install with: pip install -e '.[science]'"
        ) from exc

    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D (n, k); got shape {features.shape}")
    if labels.ndim != 1 or len(labels) != features.shape[0]:
        raise ValueError(
            f"labels must be 1D and match n_observations; got {labels.shape} vs n={features.shape[0]}"
        )

    n, k = features.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(k)]
    elif len(feature_names) != k:
        raise ValueError(f"feature_names length {len(feature_names)} != n_features {k}")

    mask_a = labels.astype(bool)
    mask_b = ~mask_a
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())

    per_feature: list[dict[str, Any]] = []
    p_welch_list: list[float] = []
    cohens_d_list: list[float] = []
    direction_votes = {"a_larger": 0, "b_larger": 0, "tied": 0}

    for i in range(k):
        col = features[:, i]
        a = col[mask_a]
        b = col[mask_b]
        sufficient = (len(a) >= _MIN_GROUP_SIZE) and (len(b) >= _MIN_GROUP_SIZE)

        mean_a = float(np.mean(a)) if len(a) else float("nan")
        mean_b = float(np.mean(b)) if len(b) else float("nan")
        median_a = float(np.median(a)) if len(a) else float("nan")
        median_b = float(np.median(b)) if len(b) else float("nan")
        std_a = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
        std_b = float(np.std(b, ddof=1)) if len(b) > 1 else float("nan")

        delta = mean_a - mean_b
        delta_pct = (100.0 * delta / mean_b) if (sufficient and mean_b != 0) else float("nan")

        if sufficient:
            d = _cohens_d(a, b)
            cohens_d_list.append(d)
            try:
                p_welch = float(scipy_stats.ttest_ind(a, b, equal_var=False).pvalue)
            except Exception:
                p_welch = float("nan")
            try:
                p_mwu = float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except Exception:
                p_mwu = float("nan")
            if not math.isnan(p_welch):
                p_welch_list.append(p_welch)
            if mean_a > mean_b:
                direction_votes["a_larger"] += 1
            elif mean_a < mean_b:
                direction_votes["b_larger"] += 1
            else:
                direction_votes["tied"] += 1
        else:
            d = float("nan")
            p_welch = float("nan")
            p_mwu = float("nan")

        per_feature.append({
            "feature_name": feature_names[i],
            "n_animate": len(a) if label_names[0] == "animate" else len(a),
            "n_inanimate": len(b) if label_names[1] == "inanimate" else len(b),
            f"n_{label_names[0]}": len(a),
            f"n_{label_names[1]}": len(b),
            f"mean_{label_names[0]}": mean_a,
            f"mean_{label_names[1]}": mean_b,
            f"median_{label_names[0]}": median_a,
            f"median_{label_names[1]}": median_b,
            f"std_{label_names[0]}": std_a,
            f"std_{label_names[1]}": std_b,
            "delta_mean": delta,
            "delta_mean_pct": delta_pct,
            "cohens_d": d,
            "p_welch": p_welch,
            "p_mannwhitneyu": p_mwu,
            "sufficient_samples": sufficient,
        })

    # Overall summaries
    n_tests = len(p_welch_list)
    min_p = min(p_welch_list) if p_welch_list else float("nan")
    min_p_bonf = min(1.0, min_p * n_tests) if (n_tests > 0 and not math.isnan(min_p)) else float("nan")
    median_d = float(np.median(cohens_d_list)) if cohens_d_list else float("nan")

    direction: str
    if direction_votes["a_larger"] > direction_votes["b_larger"] and direction_votes["a_larger"] > direction_votes["tied"]:
        direction = f"{label_names[0]}_larger"
    elif direction_votes["b_larger"] > direction_votes["a_larger"] and direction_votes["b_larger"] > direction_votes["tied"]:
        direction = f"{label_names[1]}_larger"
    else:
        direction = "mixed"

    result: dict[str, Any] = {
        f"n_{label_names[0]}": n_a,
        f"n_{label_names[1]}": n_b,
        "feature_axis": feature_axis,
        "units": units,
        "label_names": list(label_names),
        "per_feature": per_feature,
        "overall": {
            "min_p_welch": min_p,
            "min_p_welch_bonferroni": min_p_bonf,
            "median_cohens_d": median_d,
            "direction": direction,
            "n_tests": n_tests,
        },
    }
    result["summary"] = interpret_comparison(result)
    return result


# ---------- Plain-English summary ----------

def interpret_comparison(result: dict[str, Any]) -> str:
    """One-to-two sentence summary, friendly to chat agents."""
    label_a, label_b = result["label_names"]
    units = result.get("units") or ""
    units_suffix = f" {units}" if units else ""
    overall = result["overall"]
    per_feature = result["per_feature"]
    feature_names = [f["feature_name"] for f in per_feature]

    median_d = overall.get("median_cohens_d", float("nan"))
    min_p = overall.get("min_p_welch", float("nan"))
    min_p_bonf = overall.get("min_p_welch_bonferroni", float("nan"))
    direction = overall.get("direction", "mixed")

    if overall["n_tests"] == 0:
        return (
            f"Insufficient samples to compare {label_a} vs {label_b}: every feature had "
            f"< {_MIN_GROUP_SIZE} observations in at least one group."
        )

    # Mean percent difference, averaged over sufficient features
    valid_deltas = [f["delta_mean_pct"] for f in per_feature if f["sufficient_samples"] and not math.isnan(f["delta_mean_pct"])]
    avg_pct = float(np.mean(valid_deltas)) if valid_deltas else float("nan")

    # Find feature with strongest |Cohen's d|
    valid = [(f, abs(f["cohens_d"])) for f in per_feature if f["sufficient_samples"] and not math.isnan(f["cohens_d"])]
    if valid:
        strongest, _ = max(valid, key=lambda kv: kv[1])
        strongest_name = strongest["feature_name"]
        strongest_d = strongest["cohens_d"]
        strongest_p = strongest["p_welch"]
    else:
        strongest_name = strongest_d = strongest_p = None

    if direction == f"{label_a}_larger":
        comp = f"{label_a} images produce JPEG bytes {abs(avg_pct):.1f}% larger on average than {label_b} images"
    elif direction == f"{label_b}_larger":
        comp = f"{label_b} images produce JPEG bytes {abs(avg_pct):.1f}% larger on average than {label_a} images"
    else:
        comp = f"{label_a} and {label_b} show mixed direction across {result['feature_axis']}"

    qstr = f"across {result['feature_axis']} {feature_names}"
    detail = f"(median Cohen's d = {median_d:.2f}"
    if strongest_name is not None and not math.isnan(strongest_d):
        detail += f", strongest at {result['feature_axis'].split('_')[0]}={strongest_name}: d={strongest_d:.2f}, p={strongest_p:.3g}"
    if not math.isnan(min_p_bonf):
        detail += f", smallest p={min_p:.3g} (Bonferroni {min_p_bonf:.3g})"
    detail += ")."
    _ = units_suffix  # currently informational; kept for future expansion
    return f"{comp} {qstr} {detail}"


# ---------- Plot ----------

def make_comparison_plot(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: tuple[str, str],
    feature_names: list[str],
    units: str,
    output_path: Path,
    title: str,
) -> Path:
    """Self-contained interactive HTML grouped boxplot. One box per (feature, group)."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Comparison plots require plotly. Install with: pip install -e '.[viz]'"
        ) from exc

    features = np.asarray(features)
    labels = np.asarray(labels).astype(bool)

    fig = go.Figure()
    color_a = "#e06c75"
    color_b = "#61afef"

    # Build long-form arrays: one entry per (image, feature).
    for label_val, color, name in [(True, color_a, label_names[0]), (False, color_b, label_names[1])]:
        mask = labels if label_val else ~labels
        if not mask.any():
            continue
        sub = features[mask]
        xs: list[str] = []
        ys: list[float] = []
        for feat_idx, feat_name in enumerate(feature_names):
            col = sub[:, feat_idx]
            xs.extend([feat_name] * len(col))
            ys.extend(col.tolist())
        fig.add_trace(
            go.Box(
                x=xs, y=ys, name=name, marker_color=color,
                boxmean=True,
            )
        )

    fig.update_layout(
        title=title,
        boxmode="group",
        xaxis_title="quality level",
        yaxis_title=f"size ({units})" if units else "value",
        template="plotly_dark",
        legend_title="group",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
    return output_path


# ---------- Cached MCP-ready wrapper ----------

def compare_jpeg_animate_inanimate_views(
    jpeg_view: AnalysisView,
    vit_view: AnalysisView,
    plot: bool = True,
) -> dict[str, Any]:
    """Compare JPEG sizes (from `jpeg_view`) between animate/inanimate groups (from `vit_view`).

    Uses `cached_computation` keyed on both lineage hashes — re-running with the
    same view pair is a cache hit.

    Args:
        jpeg_view: AnalysisView produced by `extract_jpeg_size_view`. Must have
            `sizes_kb.npy` under `artifact_path` and a `qualities` list in `summary`.
        vit_view: AnalysisView produced by `extract_vit_features_view`. Must have
            `animate_inanimate.npy` under `artifact_path`.
        plot: render an interactive HTML grouped boxplot under the cache dir.

    Returns the comparison result dict (with `from_cache` flag and `artifacts.plot_html`).
    """
    if jpeg_view.artifact_path is None:
        raise ValueError("jpeg_view has no artifact_path; can't load sizes_kb.npy.")
    if vit_view.artifact_path is None:
        raise ValueError("vit_view has no artifact_path; can't load animate_inanimate.npy.")

    spec = ComputationSpec(
        family="comparisons",
        scope=jpeg_view.lineage_hash,
        name="jpeg_animate_inanimate",
        parameters={
            "statistical_methods": ["welch", "mannwhitneyu", "cohens_d"],
            "plot": bool(plot),
        },
        input_fingerprints=[jpeg_view.lineage_hash, vit_view.lineage_hash],
    )

    sizes_kb_path = Path(jpeg_view.artifact_path) / "sizes_kb.npy"
    ai_path = Path(vit_view.artifact_path) / "animate_inanimate.npy"
    qualities = jpeg_view.summary.get("qualities") or []

    def _compute(out_dir: Path):
        sizes_kb = load_npy(sizes_kb_path)
        animate_inanimate = load_npy(ai_path).astype(np.int8)
        if sizes_kb.shape[0] != animate_inanimate.shape[0]:
            raise ValueError(
                f"Row mismatch: sizes_kb has {sizes_kb.shape[0]} rows but "
                f"animate_inanimate has {animate_inanimate.shape[0]} labels."
            )

        feature_names = [str(int(q)) for q in qualities] if qualities else None
        result = compare_groups_by_label(
            features=sizes_kb,
            labels=animate_inanimate,
            label_names=("animate", "inanimate"),
            feature_names=feature_names,
            units="kB",
            feature_axis="jpeg_quality_levels",
        )
        result["input"] = {
            "feature_view_id": jpeg_view.view_id,
            "label_source_view_id": vit_view.view_id,
            "label_source_array_path": str(ai_path),
            "feature_array_path": str(sizes_kb_path),
        }

        artifacts: dict[str, Any] = {}
        if plot:
            plot_path = make_comparison_plot(
                features=sizes_kb,
                labels=animate_inanimate.astype(bool),
                label_names=("animate", "inanimate"),
                feature_names=feature_names or [str(i) for i in range(sizes_kb.shape[1])],
                units="kB",
                output_path=out_dir / "comparison.html",
                title=f"JPEG size by animate / inanimate ({jpeg_view.view_id[:14]})",
            )
            artifacts["plot_html"] = str(plot_path)
        result["artifacts"] = artifacts

        save_json(out_dir / "result.json", result)

        view = AnalysisView.new(
            kind=AnalysisViewKind.METRIC_TABLE,
            manifest_id=jpeg_view.manifest_id,
            shape=[sizes_kb.shape[1], 1],
            axes={"rows": "feature", "cols": "comparison_metrics"},
            source_roles=jpeg_view.source_roles,
            transformation_lineage=[
                f"jpeg:{jpeg_view.lineage_hash[:12]}",
                f"labels:{vit_view.lineage_hash[:12]}",
                "compare_groups:welch+mwu+cohens_d",
            ],
            artifact_path=str(out_dir),
            summary={
                "n_features": sizes_kb.shape[1],
                "n_animate": int(animate_inanimate.sum()),
                "n_inanimate": int(len(animate_inanimate) - animate_inanimate.sum()),
                "median_cohens_d": result["overall"]["median_cohens_d"],
                "min_p_welch_bonferroni": result["overall"]["min_p_welch_bonferroni"],
                "direction": result["overall"]["direction"],
            },
        )
        return view, result

    view, summary, from_cache = cached_computation(spec, _compute)
    if from_cache:
        logger.info("Comparison cache hit (%s) -> %s", spec.hash(), view.artifact_path)
    summary["from_cache"] = from_cache
    summary["view_id"] = view.view_id
    summary["artifact_path"] = view.artifact_path
    return summary
