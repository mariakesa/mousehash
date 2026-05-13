"""Prototype standardized RSA path for spike-trial datasets.

This path assumes:

- spike times are available in the NWB units table
- trials are available in the NWB trials table
- a categorical trial column exists with repeated labels

The prototype builds a trial x unit count matrix using a fixed response window
from a chosen alignment column, computes a neural RDM across trials, builds a
categorical target RDM from trial labels, and compares them with RSA.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

from mousehash.agents.dandi_agent.catalogs.loaders import load_tools
from mousehash.agents.dandi_agent.models import EvidenceBackedRoleManifest
from mousehash.agents.dandi_agent.readiness import compute_tool_readiness


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RSA_TOOL_ID = "run_rsa"
DEFAULT_OUTPUT_DIR = Path("RSAAnalysis") / "outputs" / "spike_trial_rsa"
PREFERRED_LABEL_COLUMNS = (
    "trial_type",
    "type",
    "condition",
    "condition_label",
    "stimulus_type",
    "stim_type",
    "choice",
    "response",
    "task_label",
    "stim_present",
)
PREFERRED_ALIGN_COLUMNS = (
    "start_time",
    "cue_start_time",
    "stimulus_onset",
    "stim_on_time",
)
EXCLUDED_LABEL_COLUMNS = {
    "start_time",
    "stop_time",
    "id",
    "index",
    "trial_index",
    "timestamps",
}


def load_manifest(manifest_path: Path) -> EvidenceBackedRoleManifest:
    return EvidenceBackedRoleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )


def _manifest_requires_spike_trial_path(manifest: EvidenceBackedRoleManifest) -> None:
    rsa_tool = load_tools()[RSA_TOOL_ID]
    report = compute_tool_readiness(manifest, rsa_tool)
    if report.status != "ready":
        raise ValueError(
            f"Manifest is not RSA-ready (status={report.status}): {report.rationale}"
        )
    if manifest.status("neural_data.spikes") not in ("present", "likely_present"):
        raise ValueError("Manifest does not support spike-based RSA.")
    if manifest.status("time_organization.trials") not in ("present", "likely_present"):
        raise ValueError("Manifest does not contain trial structure.")
    if not manifest.nwb_path:
        raise ValueError("Manifest has no local NWB path.")


def _load_nwb_tables(nwb_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        if nwbfile.units is None:
            raise ValueError("NWB file does not contain a units table.")
        if nwbfile.trials is None:
            raise ValueError("NWB file does not contain a trials table.")
        units_df = nwbfile.units.to_dataframe()
        trials_df = nwbfile.trials.to_dataframe()
    return units_df, trials_df


def _normalize_trial_values(series: pd.Series) -> pd.Series:
    cleaned = series.map(lambda value: "1" if value is True else "0" if value is False else value)
    cleaned = cleaned.astype(str).str.strip()
    cleaned = cleaned.mask(cleaned.isin(["", "nan", "None"]))
    return cleaned


def _select_align_column(trials_df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in trials_df.columns:
            raise ValueError(f"Requested align column {requested!r} not in trials table.")
        return requested
    for column in PREFERRED_ALIGN_COLUMNS:
        if column in trials_df.columns:
            return column
    raise ValueError("Could not find a supported alignment column in the trials table.")


def _resolve_time_series(trials_df: pd.DataFrame, column: str) -> np.ndarray:
    values = trials_df[column].to_numpy(dtype=np.float64)
    if column == "start_time" or "start_time" not in trials_df.columns:
        return values

    trial_starts = trials_df["start_time"].to_numpy(dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(trial_starts)
    if not finite.any():
        return values

    if np.nanmedian(values[finite]) < 0.5 * np.nanmedian(trial_starts[finite]):
        return trial_starts + values
    return values


def _resolve_stop_times(trials_df: pd.DataFrame, align_times: np.ndarray, response_window_s: float) -> np.ndarray:
    fallback = align_times + float(response_window_s)
    if "stop_time" not in trials_df.columns:
        return fallback

    raw_stop = trials_df["stop_time"].to_numpy(dtype=np.float64)
    if np.isfinite(raw_stop).any():
        if "start_time" in trials_df.columns:
            trial_starts = trials_df["start_time"].to_numpy(dtype=np.float64)
            finite = np.isfinite(raw_stop) & np.isfinite(trial_starts)
            if finite.any() and np.nanmedian(raw_stop[finite]) < 0.5 * np.nanmedian(trial_starts[finite]):
                raw_stop = trial_starts + raw_stop
        return np.where(np.isfinite(raw_stop), np.minimum(fallback, raw_stop), fallback)

    return fallback


def _is_categorical_candidate(series: pd.Series) -> bool:
    values = _normalize_trial_values(series).dropna()
    if len(values) < 4:
        return False
    unique_values = values.unique()
    if len(unique_values) < 2 or len(unique_values) > max(20, len(values) // 2):
        return False
    counts = values.value_counts()
    if counts.max() < 2:
        return False
    if len(counts) < 2:
        return False
    same_label_pairs = sum(int(count) * (int(count) - 1) // 2 for count in counts)
    return same_label_pairs > 0


def choose_label_column(trials_df: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in trials_df.columns:
            raise ValueError(f"Requested label column {requested!r} not in trials table.")
        if not _is_categorical_candidate(trials_df[requested]):
            raise ValueError(f"Requested label column {requested!r} is not suitable for categorical RSA.")
        return requested

    for column in PREFERRED_LABEL_COLUMNS:
        if column in trials_df.columns and _is_categorical_candidate(trials_df[column]):
            return column

    fallback_columns = [
        col for col in map(str, trials_df.columns)
        if col not in EXCLUDED_LABEL_COLUMNS and _is_categorical_candidate(trials_df[col])
    ]
    if not fallback_columns:
        raise ValueError("No categorical trial label column was suitable for RSA.")
    return fallback_columns[0]


def _trial_mask(trials_df: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(trials_df), dtype=bool)
    if "is_good" in trials_df.columns:
        is_good = _normalize_trial_values(trials_df["is_good"]).fillna("0")
        mask &= is_good.isin({"1", "true", "True"}).to_numpy()
    return mask


def _extract_spike_count_matrix(
    units_df: pd.DataFrame,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    if "spike_times" not in units_df.columns:
        raise ValueError("Units table does not expose a spike_times column.")
    counts = np.zeros((len(starts), len(units_df)), dtype=np.float32)
    for unit_index, spike_times in enumerate(units_df["spike_times"]):
        unit_spikes = np.asarray(spike_times, dtype=np.float64)
        left = np.searchsorted(unit_spikes, starts, side="left")
        right = np.searchsorted(unit_spikes, stops, side="left")
        counts[:, unit_index] = right - left
    return counts


def _distance_metric_name(metric: str) -> str:
    supported = {"correlation", "euclidean", "cosine"}
    if metric not in supported:
        raise ValueError(f"Unsupported distance metric {metric!r}; expected one of {sorted(supported)}")
    return metric


def _neural_rdm(trial_matrix: np.ndarray, metric: str) -> np.ndarray:
    if trial_matrix.shape[0] < 3:
        raise ValueError("Need at least 3 trials to compute a usable RDM.")
    if trial_matrix.shape[1] < 1:
        raise ValueError("Need at least 1 unit to compute a neural RDM.")
    if metric == "correlation" and trial_matrix.shape[1] == 1:
        metric = "euclidean"
    condensed = pdist(trial_matrix, metric=_distance_metric_name(metric))
    return squareform(condensed)


def _target_rdm(labels: np.ndarray) -> np.ndarray:
    return (labels[:, None] != labels[None, :]).astype(np.float32)


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    tri = np.triu_indices_from(matrix, k=1)
    return matrix[tri]


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _safe_median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _compute_meta_features(
    counts: np.ndarray,
    labels: np.ndarray,
    neural_rdm: np.ndarray,
    rsa_statistic: float,
    p_value: float,
) -> dict[str, Any]:
    trial_totals = counts.sum(axis=1)
    unit_totals = counts.sum(axis=0)
    nonzero_fraction = float(np.count_nonzero(counts) / counts.size) if counts.size else 0.0

    label_counts = Counter(labels)
    category_sizes = np.asarray(list(label_counts.values()), dtype=np.float64)
    label_probs = category_sizes / category_sizes.sum()
    label_entropy_bits = float(
        -np.sum(label_probs * np.log2(label_probs))
    ) if label_probs.size else 0.0

    same_mask = labels[:, None] == labels[None, :]
    upper_mask = np.triu(np.ones_like(same_mask, dtype=bool), k=1)
    within_mask = same_mask & upper_mask
    between_mask = (~same_mask) & upper_mask
    within_distances = neural_rdm[within_mask]
    between_distances = neural_rdm[between_mask]

    return {
        "n_trials": int(counts.shape[0]),
        "n_units": int(counts.shape[1]),
        "n_categories": int(len(label_counts)),
        "min_category_size": int(category_sizes.min()) if category_sizes.size else 0,
        "max_category_size": int(category_sizes.max()) if category_sizes.size else 0,
        "category_balance_ratio": float(category_sizes.min() / category_sizes.max()) if category_sizes.size else 0.0,
        "label_entropy_bits": label_entropy_bits,
        "mean_spikes_per_trial": _safe_mean(trial_totals),
        "std_spikes_per_trial": _safe_std(trial_totals),
        "median_spikes_per_trial": _safe_median(trial_totals),
        "mean_spikes_per_unit": _safe_mean(unit_totals),
        "std_spikes_per_unit": _safe_std(unit_totals),
        "median_spikes_per_unit": _safe_median(unit_totals),
        "active_unit_fraction": float(np.mean(unit_totals > 0)) if unit_totals.size else 0.0,
        "nonzero_count_fraction": nonzero_fraction,
        "mean_neural_distance": _safe_mean(_upper_triangle(neural_rdm)),
        "std_neural_distance": _safe_std(_upper_triangle(neural_rdm)),
        "mean_within_label_distance": _safe_mean(within_distances),
        "mean_between_label_distance": _safe_mean(between_distances),
        "distance_separation": _safe_mean(between_distances) - _safe_mean(within_distances),
        "rsa_statistic": float(rsa_statistic),
        "p_value": float(p_value),
    }


def _compute_rsa_statistic(neural_rdm: np.ndarray, target_rdm: np.ndarray, method: str) -> float:
    neural_vec = _upper_triangle(neural_rdm)
    target_vec = _upper_triangle(target_rdm)
    if np.allclose(target_vec, target_vec[0]):
        raise ValueError("Target RDM is constant; choose a label column with repeated categories.")
    if method == "spearman":
        stat, _ = spearmanr(neural_vec, target_vec)
    elif method == "pearson":
        stat, _ = pearsonr(neural_vec, target_vec)
    else:
        raise ValueError("Unsupported RSA correlation; expected 'spearman' or 'pearson'.")
    return float(0.0 if np.isnan(stat) else stat)


def _permutation_test(
    neural_rdm: np.ndarray,
    labels: np.ndarray,
    method: str,
    n_permutations: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    observed = _compute_rsa_statistic(neural_rdm, _target_rdm(labels), method)
    null_stats = np.zeros(n_permutations, dtype=np.float32)
    for index in range(n_permutations):
        permuted = rng.permutation(labels)
        null_stats[index] = _compute_rsa_statistic(neural_rdm, _target_rdm(permuted), method)
    return observed, null_stats


def run_spike_trial_rsa(
    manifest_path: Path,
    *,
    output_dir: Path,
    item_column: str | None = None,
    align_column: str | None = None,
    response_window_s: float = 0.4,
    distance_metric: str = "correlation",
    rsa_correlation: str = "spearman",
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    _manifest_requires_spike_trial_path(manifest)

    nwb_path = Path(str(manifest.nwb_path)).expanduser()
    units_df, trials_df = _load_nwb_tables(nwb_path)

    label_column = choose_label_column(trials_df, requested=item_column)
    align_column_name = _select_align_column(trials_df, requested=align_column)
    mask = _trial_mask(trials_df)
    labels = _normalize_trial_values(trials_df[label_column])
    mask &= labels.notna().to_numpy()

    trial_starts = _resolve_time_series(trials_df, align_column_name)
    trial_stops = _resolve_stop_times(trials_df, trial_starts, response_window_s)
    valid_window = trial_stops > trial_starts
    mask &= valid_window

    if mask.sum() < 4:
        raise ValueError("Not enough valid trials remain after filtering.")

    filtered_trials = trials_df.loc[mask].copy()
    filtered_labels = labels.loc[mask].astype(str).to_numpy()
    counts = _extract_spike_count_matrix(units_df, trial_starts[mask], trial_stops[mask])
    if counts.shape[1] == 0:
        raise ValueError("No units available for spike-count RSA.")

    label_counts = Counter(filtered_labels)
    same_label_pairs = sum(count * (count - 1) // 2 for count in label_counts.values())
    if len(label_counts) < 2 or same_label_pairs == 0:
        raise ValueError("Categorical label column does not yield repeated categories for RSA.")

    neural_rdm = _neural_rdm(counts, distance_metric)
    target_rdm = _target_rdm(filtered_labels)
    rsa_statistic, null_stats = _permutation_test(
        neural_rdm,
        filtered_labels,
        rsa_correlation,
        n_permutations,
        seed,
    )
    p_value = float((1 + np.sum(np.abs(null_stats) >= abs(rsa_statistic))) / (1 + len(null_stats)))

    dataset_key = f"{manifest.dandiset_id or 'unknown'}__{manifest.asset_id or manifest_path.stem}"
    out_dir = output_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "trial_spike_counts.npy", counts)
    np.save(out_dir / "neural_rdm.npy", neural_rdm)
    np.save(out_dir / "target_rdm.npy", target_rdm)
    np.save(out_dir / "null_rsa.npy", null_stats)

    trial_metadata = filtered_trials.copy()
    trial_metadata.insert(0, "trial_label", filtered_labels)
    trial_metadata.to_csv(out_dir / "trial_metadata.csv", index=False)

    summary = {
        "manifest_path": str(manifest_path),
        "nwb_path": str(nwb_path),
        "dandiset_id": manifest.dandiset_id,
        "asset_id": manifest.asset_id,
        "item_column": label_column,
        "align_column": align_column_name,
        "response_window_s": float(response_window_s),
        "distance_metric": distance_metric,
        "rsa_correlation": rsa_correlation,
        "n_permutations": int(n_permutations),
        "n_trials": int(counts.shape[0]),
        "n_units": int(counts.shape[1]),
        "label_counts": dict(label_counts),
        "rsa_statistic": rsa_statistic,
        "p_value": p_value,
        "meta_features": _compute_meta_features(
            counts,
            filtered_labels,
            neural_rdm,
            rsa_statistic,
            p_value,
        ),
        "output_dir": str(out_dir),
        "artifacts": {
            "trial_spike_counts": str(out_dir / "trial_spike_counts.npy"),
            "neural_rdm": str(out_dir / "neural_rdm.npy"),
            "target_rdm": str(out_dir / "target_rdm.npy"),
            "null_rsa": str(out_dir / "null_rsa.npy"),
            "trial_metadata": str(out_dir / "trial_metadata.csv"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--item-column", default=None)
    parser.add_argument("--align-column", default=None)
    parser.add_argument("--response-window-s", type=float, default=0.4)
    parser.add_argument("--distance-metric", choices=["correlation", "euclidean", "cosine"], default="correlation")
    parser.add_argument("--rsa-correlation", choices=["spearman", "pearson"], default="spearman")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_spike_trial_rsa(
        args.manifest_path,
        output_dir=args.output_dir,
        item_column=args.item_column,
        align_column=args.align_column,
        response_window_s=args.response_window_s,
        distance_metric=args.distance_metric,
        rsa_correlation=args.rsa_correlation,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
