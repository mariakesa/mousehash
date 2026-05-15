#!/usr/bin/env python3
"""
MouseHash exploratory visualization tools for DANDI/NWB spike-behavior-optogenetics files.

Targeted first at DANDI 000011-style files with:
  - /units spike_times
  - /intervals/trials with task/trial_instruction/outcome/early_lick columns
  - /acquisition/BehavioralEvents/{sample, delay, go}
  - /acquisition/BehavioralTimeSeries/lick_trace
  - /general/optogenetics sites such as left/right/bilateral ALM

This script is intentionally exploratory. It does not fit final statistical models.
It materializes simple analysis views and saves artifacts:
  - trial table CSV
  - unit summary CSV
  - event-aligned raster plots
  - PSTH plots grouped by trial columns
  - PCA neural trajectories
  - lick trace overview
  - artifact manifest JSON

Install:
  pip install pynwb h5py numpy pandas matplotlib scikit-learn

Examples:
  python mousehash_explore_nwb_visuals.py /path/to/file.nwb
  python mousehash_explore_nwb_visuals.py /path/to/file.nwb --align go --group-by trial_instruction
  python mousehash_explore_nwb_visuals.py /path/to/file.nwb --out-dir ./mh_artifacts --max-units 10

Design:
  Roles -> Transformations -> AnalysisViews -> Visualization tools -> Artifacts
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pynwb import NWBHDF5IO
except ImportError as exc:
    raise SystemExit("PyNWB is required. Install with: pip install pynwb h5py") from exc

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    PCA = None
    StandardScaler = None


@dataclass
class UnitSpikes:
    unit_id: Any
    spike_times: np.ndarray
    quality: Optional[Any] = None
    cell_type: Optional[Any] = None
    electrode_group: Optional[Any] = None
    n_spikes: int = 0
    firing_rate_hz: Optional[float] = None


def safe_name(text: Any) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unnamed"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    try:
        return np.asarray(x[:])
    except Exception:
        try:
            return np.asarray(x)
        except Exception:
            return np.array([])


def decode_value(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    return v


def decode_array(arr: Iterable[Any]) -> List[Any]:
    return [decode_value(x) for x in list(arr)]


def maybe_series_values(col: Any) -> List[Any]:
    try:
        data = col.data
        return decode_array(to_numpy(data))
    except Exception:
        try:
            return decode_array(col[:])
        except Exception:
            try:
                return decode_array(col)
            except Exception:
                return []


def compact_unique(values: Sequence[Any], max_items: int = 20) -> List[Any]:
    out = []
    seen = set()
    for v in values:
        key = str(v)
        if key not in seen:
            out.append(v)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def save_fig(path: Path, dpi: int = 160) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def read_trials(nwbfile: Any) -> pd.DataFrame:
    if getattr(nwbfile, "trials", None) is None:
        return pd.DataFrame()
    trials = nwbfile.trials
    try:
        df = trials.to_dataframe().reset_index()
    except Exception:
        data: Dict[str, List[Any]] = {}
        for name in getattr(trials, "colnames", []):
            try:
                data[name] = maybe_series_values(trials[name])
            except Exception:
                pass
        df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(decode_value)
    return df


def read_event_times(nwbfile: Any) -> Dict[str, np.ndarray]:
    events: Dict[str, np.ndarray] = {}
    acquisition = getattr(nwbfile, "acquisition", {}) or {}
    for name, obj in acquisition.items():
        obj_type = obj.__class__.__name__
        if obj_type == "BehavioralEvents":
            try:
                for ts_name, ts in obj.time_series.items():
                    timestamps = to_numpy(getattr(ts, "timestamps", None))
                    if timestamps.size:
                        events[str(ts_name)] = timestamps.astype(float)
            except Exception:
                pass
        if hasattr(obj, "timestamps") and any(k in str(name).lower() for k in ["sample", "delay", "go", "lick", "stim"]):
            timestamps = to_numpy(getattr(obj, "timestamps", None))
            if timestamps.size:
                events[str(name)] = timestamps.astype(float)
    return events


def read_lick_trace(nwbfile: Any) -> Optional[pd.DataFrame]:
    acquisition = getattr(nwbfile, "acquisition", {}) or {}
    for _, obj in acquisition.items():
        if obj.__class__.__name__ != "BehavioralTimeSeries":
            continue
        try:
            for ts_name, ts in obj.time_series.items():
                if "lick" in str(ts_name).lower():
                    data = to_numpy(getattr(ts, "data", None)).squeeze()
                    timestamps = to_numpy(getattr(ts, "timestamps", None)).squeeze()
                    if data.size and timestamps.size and data.size == timestamps.size:
                        return pd.DataFrame({"time": timestamps.astype(float), "lick_trace": data})
        except Exception:
            pass
    return None


def infer_recording_duration(nwbfile: Any) -> Optional[float]:
    candidates: List[float] = []
    try:
        df = read_trials(nwbfile)
        if not df.empty and "stop_time" in df.columns:
            candidates.append(float(pd.to_numeric(df["stop_time"], errors="coerce").max()))
    except Exception:
        pass
    try:
        for ev in read_event_times(nwbfile).values():
            if ev.size:
                candidates.append(float(np.nanmax(ev)))
    except Exception:
        pass
    try:
        units = getattr(nwbfile, "units", None)
        if units is not None:
            flat = to_numpy(units["spike_times"].data).astype(float)
            if flat.size:
                candidates.append(float(np.nanmax(flat)))
    except Exception:
        pass
    candidates = [x for x in candidates if np.isfinite(x) and x > 0]
    return max(candidates) if candidates else None


def read_units(nwbfile: Any) -> List[UnitSpikes]:
    units = getattr(nwbfile, "units", None)
    if units is None:
        return []
    unit_spikes: List[UnitSpikes] = []
    try:
        df = units.to_dataframe().reset_index()
    except Exception:
        df = pd.DataFrame()
    if not df.empty and "spike_times" in df.columns:
        duration = infer_recording_duration(nwbfile)
        for _, row in df.iterrows():
            st = np.asarray(row["spike_times"], dtype=float)
            unit_id = decode_value(row.get("id", len(unit_spikes)))
            quality = decode_value(row.get("quality", None))
            cell_type = decode_value(row.get("cell_type", None))
            electrode_group = decode_value(row.get("electrode_group", None))
            fr = float(len(st) / duration) if duration and duration > 0 else None
            unit_spikes.append(UnitSpikes(unit_id, st, quality, cell_type, str(electrode_group) if electrode_group is not None else None, int(len(st)), fr))
        return unit_spikes
    try:
        spike_times = to_numpy(units["spike_times"].data).astype(float)
        spike_idx = to_numpy(units["spike_times_index"].data).astype(int)
        ids = maybe_series_values(units["id"]) if "id" in units.colnames else list(range(len(spike_idx)))
        starts = np.r_[0, spike_idx[:-1]]
        duration = infer_recording_duration(nwbfile)
        for i, (a, b) in enumerate(zip(starts, spike_idx)):
            st = spike_times[a:b]
            fr = float(len(st) / duration) if duration and duration > 0 else None
            unit_spikes.append(UnitSpikes(decode_value(ids[i]) if i < len(ids) else i, st, n_spikes=int(len(st)), firing_rate_hz=fr))
    except Exception as exc:
        warnings.warn(f"Could not read units/spike_times: {exc}")
    return unit_spikes


def align_event_to_trials(trials: pd.DataFrame, event_times: np.ndarray) -> np.ndarray:
    if trials.empty or event_times.size == 0 or not {"start_time", "stop_time"}.issubset(trials.columns):
        return np.full(len(trials), np.nan)
    starts = pd.to_numeric(trials["start_time"], errors="coerce").to_numpy(float)
    stops = pd.to_numeric(trials["stop_time"], errors="coerce").to_numpy(float)
    aligned = np.full(len(trials), np.nan)
    event_times = np.asarray(event_times, dtype=float)
    for i, (a, b) in enumerate(zip(starts, stops)):
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        hits = event_times[(event_times >= a) & (event_times <= b)]
        if hits.size:
            aligned[i] = hits[0]
    return aligned


def choose_alignment_times(trials: pd.DataFrame, events: Dict[str, np.ndarray], align: str) -> Tuple[str, np.ndarray]:
    align = align.lower()
    for name, times in events.items():
        if name.lower() == align:
            return name, align_event_to_trials(trials, times)
    aliases = {
        "go_cue": ["go", "cue"],
        "go": ["go"],
        "sample": ["sample"],
        "delay": ["delay"],
        "photostim": ["photostim", "laser", "stim"],
        "lick": ["lick"],
        "trial_start": ["trial_start", "start_time"],
    }
    keys = aliases.get(align, [align])
    for name, times in events.items():
        lname = name.lower()
        if any(k in lname for k in keys):
            return name, align_event_to_trials(trials, times)
    if "start_time" in trials.columns:
        return "trial_start", pd.to_numeric(trials["start_time"], errors="coerce").to_numpy(float)
    raise ValueError(f"Could not infer alignment times for align={align!r}")


def make_binned_spike_tensor(units: Sequence[UnitSpikes], align_times: np.ndarray, window: Tuple[float, float], bin_size: float) -> Tuple[np.ndarray, np.ndarray]:
    n_trials = len(align_times)
    n_units = len(units)
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2.0
    X = np.zeros((n_trials, len(centers), n_units), dtype=float)
    for ti, t0 in enumerate(align_times):
        if not np.isfinite(t0):
            X[ti, :, :] = np.nan
            continue
        lo, hi = t0 + window[0], t0 + window[1]
        for ui, unit in enumerate(units):
            st = unit.spike_times
            rel = st[(st >= lo) & (st <= hi)] - t0
            counts, _ = np.histogram(rel, bins=edges)
            X[ti, :, ui] = counts / bin_size
    return X, centers


def make_epoch_response_matrix(units: Sequence[UnitSpikes], align_times: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    n_trials, n_units = len(align_times), len(units)
    duration = window[1] - window[0]
    Y = np.zeros((n_trials, n_units), dtype=float)
    for ti, t0 in enumerate(align_times):
        if not np.isfinite(t0):
            Y[ti, :] = np.nan
            continue
        lo, hi = t0 + window[0], t0 + window[1]
        for ui, unit in enumerate(units):
            st = unit.spike_times
            Y[ti, ui] = np.sum((st >= lo) & (st <= hi)) / duration
    return Y


def plot_lick_trace(lick_df: pd.DataFrame, out: Path, max_seconds: Optional[float] = 300.0) -> Path:
    df = lick_df.copy()
    if max_seconds is not None:
        t0 = float(df["time"].min())
        df = df[df["time"] <= t0 + max_seconds]
    plt.figure(figsize=(12, 3))
    plt.plot(df["time"], df["lick_trace"], linewidth=0.8)
    plt.xlabel("time (s)")
    plt.ylabel("lick_trace")
    plt.title("BehavioralTimeSeries: lick_trace")
    save_fig(out)
    return out


def plot_trial_event_overview(trials: pd.DataFrame, events: Dict[str, np.ndarray], out: Path) -> Path:
    plt.figure(figsize=(12, 5))
    if {"start_time", "stop_time"}.issubset(trials.columns):
        starts = pd.to_numeric(trials["start_time"], errors="coerce").to_numpy(float)
        stops = pd.to_numeric(trials["stop_time"], errors="coerce").to_numpy(float)
        idx = np.arange(len(trials))
        plt.hlines(idx, starts, stops, alpha=0.5, linewidth=1, label="trials")
    for name in [k for k in events.keys() if k.lower() in {"sample", "delay", "go"}]:
        aligned = align_event_to_trials(trials, events[name])
        valid = np.isfinite(aligned)
        plt.scatter(aligned[valid], np.arange(len(trials))[valid], s=8, label=name)
    plt.xlabel("time (s)")
    plt.ylabel("trial")
    plt.title("Trial intervals and task events")
    plt.legend(loc="best", fontsize=8)
    save_fig(out)
    return out


def plot_raster(units: Sequence[UnitSpikes], trials: pd.DataFrame, align_times: np.ndarray, align_label: str, out: Path, window: Tuple[float, float], group_by: Optional[str], max_units: int, max_trials: int) -> Path:
    sorted_units = sorted(units, key=lambda u: u.n_spikes, reverse=True)[:max_units]
    n_trials = min(len(trials), max_trials)
    if group_by and group_by in trials.columns:
        labels = trials[group_by].astype(str).fillna("NA").to_numpy()[:n_trials]
        order = np.argsort(labels)
        ylabel = f"trial sorted by {group_by}"
    else:
        order = np.arange(n_trials)
        ylabel = "trial"
    fig_h = max(3.0, 1.8 * len(sorted_units))
    fig, axes = plt.subplots(len(sorted_units), 1, figsize=(12, fig_h), sharex=True)
    if len(sorted_units) == 1:
        axes = [axes]
    for ax, unit in zip(axes, sorted_units):
        for rank, ti in enumerate(order):
            t0 = align_times[ti]
            if not np.isfinite(t0):
                continue
            st = unit.spike_times
            rel = st[(st >= t0 + window[0]) & (st <= t0 + window[1])] - t0
            if rel.size:
                ax.vlines(rel, rank + 0.1, rank + 0.9, linewidth=0.5)
        ax.axvline(0, linestyle="--", linewidth=1)
        title = f"unit {unit.unit_id} | spikes={unit.n_spikes}"
        if unit.quality is not None:
            title += f" | quality={unit.quality}"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel(f"time from {align_label} (s)")
    fig.suptitle(f"Raster aligned to {align_label}", y=1.01)
    save_fig(out)
    return out


def plot_psth(units: Sequence[UnitSpikes], trials: pd.DataFrame, align_times: np.ndarray, align_label: str, out: Path, window: Tuple[float, float], bin_size: float, group_by: Optional[str], max_units: int) -> Path:
    sorted_units = sorted(units, key=lambda u: u.n_spikes, reverse=True)[:max_units]
    X, centers = make_binned_spike_tensor(sorted_units, align_times, window, bin_size)
    fig_h = max(3.0, 1.8 * len(sorted_units))
    fig, axes = plt.subplots(len(sorted_units), 1, figsize=(12, fig_h), sharex=True)
    if len(sorted_units) == 1:
        axes = [axes]
    if group_by and group_by in trials.columns:
        labels = trials[group_by].astype(str).fillna("NA").to_numpy()
        groups = compact_unique(labels, max_items=8)
    else:
        labels = np.array(["all"] * len(trials))
        groups = ["all"]
    for ui, (ax, unit) in enumerate(zip(axes, sorted_units)):
        for g in groups:
            mask = labels == str(g)
            if not np.any(mask):
                continue
            y = np.nanmean(X[mask, :, ui], axis=0)
            ax.plot(centers, y, linewidth=1, label=str(g))
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_ylabel("Hz")
        ax.set_title(f"unit {unit.unit_id} PSTH", fontsize=9)
    axes[-1].set_xlabel(f"time from {align_label} (s)")
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle(f"PSTH aligned to {align_label}" + (f", grouped by {group_by}" if group_by else ""), y=1.01)
    save_fig(out)
    return out


def plot_population_psth(units: Sequence[UnitSpikes], trials: pd.DataFrame, align_times: np.ndarray, align_label: str, out: Path, window: Tuple[float, float], bin_size: float, group_by: Optional[str]) -> Path:
    X, centers = make_binned_spike_tensor(units, align_times, window, bin_size)
    if group_by and group_by in trials.columns:
        labels = trials[group_by].astype(str).fillna("NA").to_numpy()
        groups = compact_unique(labels, max_items=10)
    else:
        labels = np.array(["all"] * len(trials))
        groups = ["all"]
    plt.figure(figsize=(12, 4))
    for g in groups:
        mask = labels == str(g)
        if np.any(mask):
            y = np.nanmean(X[mask, :, :], axis=(0, 2))
            plt.plot(centers, y, linewidth=1.5, label=str(g))
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel(f"time from {align_label} (s)")
    plt.ylabel("population mean firing rate (Hz)")
    plt.title("Population PSTH" + (f" grouped by {group_by}" if group_by else ""))
    plt.legend(loc="best", fontsize=8)
    save_fig(out)
    return out


def plot_condition_response_heatmap(units: Sequence[UnitSpikes], trials: pd.DataFrame, align_times: np.ndarray, align_label: str, out: Path, response_window: Tuple[float, float], group_by: Optional[str]) -> Optional[Path]:
    if not group_by or group_by not in trials.columns:
        return None
    Y = make_epoch_response_matrix(units, align_times, response_window)
    labels = trials[group_by].astype(str).fillna("NA").to_numpy()
    groups = compact_unique(labels, max_items=20)
    rows, row_names = [], []
    for g in groups:
        mask = labels == str(g)
        if np.any(mask):
            rows.append(np.nanmean(Y[mask, :], axis=0))
            row_names.append(str(g))
    if not rows:
        return None
    M = np.vstack(rows)
    plt.figure(figsize=(max(6, len(units) * 0.45), max(3, len(row_names) * 0.4)))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="mean firing rate (Hz)")
    plt.xlabel("unit")
    plt.ylabel(group_by)
    plt.yticks(np.arange(len(row_names)), row_names)
    plt.title(f"Mean response {response_window}s from {align_label}, grouped by {group_by}")
    save_fig(out)
    return out


def plot_pca_trajectories(units: Sequence[UnitSpikes], trials: pd.DataFrame, align_times: np.ndarray, align_label: str, out: Path, window: Tuple[float, float], bin_size: float, group_by: Optional[str], min_trials_per_group: int = 3) -> Optional[Path]:
    if PCA is None or StandardScaler is None:
        warnings.warn("scikit-learn not installed; skipping PCA trajectories.")
        return None
    if len(units) < 2:
        warnings.warn("Need at least 2 units for PCA trajectories; skipping.")
        return None
    X, centers = make_binned_spike_tensor(units, align_times, window, bin_size)
    valid_trials = np.isfinite(X).all(axis=(1, 2))
    Xv = X[valid_trials]
    if Xv.shape[0] < 2:
        warnings.warn("Too few valid trials for PCA trajectories.")
        return None
    flat = Xv.reshape(-1, len(units))
    if np.nanstd(flat) == 0:
        warnings.warn("Population matrix has zero variance; skipping PCA.")
        return None
    flat_z = StandardScaler().fit_transform(flat)
    n_components = min(3, len(units), flat_z.shape[0])
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(flat_z).reshape(Xv.shape[0], len(centers), n_components)
    valid_trial_df = trials.iloc[np.where(valid_trials)[0]].reset_index(drop=True)
    if group_by and group_by in valid_trial_df.columns:
        labels = valid_trial_df[group_by].astype(str).fillna("NA").to_numpy()
        groups = [g for g in compact_unique(labels, max_items=10) if np.sum(labels == str(g)) >= min_trials_per_group]
    else:
        labels = np.array(["all"] * Xv.shape[0])
        groups = ["all"]
    plt.figure(figsize=(7, 6))
    for g in groups:
        mask = labels == str(g)
        if np.any(mask):
            traj = np.nanmean(Z[mask, :, :], axis=0)
            plt.plot(traj[:, 0], traj[:, 1], linewidth=1.5, label=str(g))
            plt.scatter(traj[0, 0], traj[0, 1], s=25, marker="o")
            plt.scatter(traj[-1, 0], traj[-1, 1], s=25, marker="x")
    evr = getattr(pca, "explained_variance_ratio_", np.array([]))
    plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)" if evr.size > 0 else "PC1")
    plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)" if evr.size > 1 else "PC2")
    plt.title(f"Population PCA trajectories aligned to {align_label}")
    plt.legend(loc="best", fontsize=8)
    save_fig(out)
    save_json(out.with_suffix(".pca_summary.json"), {
        "explained_variance_ratio": evr.tolist(),
        "n_valid_trials": int(Xv.shape[0]),
        "n_time_bins": int(len(centers)),
        "n_units": int(len(units)),
        "group_by": group_by,
        "groups": [str(g) for g in groups],
        "window": list(window),
        "bin_size": bin_size,
    })
    return out


def write_trial_table(trials: pd.DataFrame, events: Dict[str, np.ndarray], out: Path) -> Path:
    df = trials.copy()
    for name, times in events.items():
        if name.lower() in {"sample", "delay", "go"}:
            df[f"{safe_name(name)}_time"] = align_event_to_trials(df, times)
    df.to_csv(out, index=False)
    return out


def write_unit_summary(units: Sequence[UnitSpikes], out: Path) -> Path:
    rows = []
    for u in units:
        rows.append({
            "unit_id": u.unit_id,
            "n_spikes": u.n_spikes,
            "firing_rate_hz": u.firing_rate_hz,
            "quality": u.quality,
            "cell_type": u.cell_type,
            "electrode_group": u.electrode_group,
            "first_spike_time": float(np.min(u.spike_times)) if u.spike_times.size else None,
            "last_spike_time": float(np.max(u.spike_times)) if u.spike_times.size else None,
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def summarize_nwb(nwbfile: Any, nwb_path: Path, trials: pd.DataFrame, events: Dict[str, np.ndarray], units: Sequence[UnitSpikes]) -> Dict[str, Any]:
    return {
        "source_file": str(nwb_path),
        "identifier": getattr(nwbfile, "identifier", None),
        "session_start_time": str(getattr(nwbfile, "session_start_time", None)),
        "experiment_description": getattr(nwbfile, "experiment_description", None),
        "institution": getattr(nwbfile, "institution", None),
        "subject": {
            "subject_id": getattr(getattr(nwbfile, "subject", None), "subject_id", None),
            "species": getattr(getattr(nwbfile, "subject", None), "species", None),
            "genotype": getattr(getattr(nwbfile, "subject", None), "genotype", None),
            "sex": getattr(getattr(nwbfile, "subject", None), "sex", None),
        },
        "n_trials": int(len(trials)),
        "trial_columns": list(map(str, trials.columns)),
        "event_names": {k: int(len(v)) for k, v in events.items()},
        "n_units": int(len(units)),
        "n_spikes_total": int(sum(u.n_spikes for u in units)),
        "recording_duration_s": infer_recording_duration(nwbfile),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate exploratory MouseHash visual artifacts from an NWB file.")
    parser.add_argument("nwb", type=Path, help="Path to NWB file.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Artifact output directory.")
    parser.add_argument("--align", default="go", help="Alignment event: go, sample, delay, trial_start, photostim, etc.")
    parser.add_argument("--group-by", default="trial_instruction", help="Trial column for grouping PSTHs/trajectories.")
    parser.add_argument("--window", nargs=2, type=float, default=(-2.0, 2.0), metavar=("START", "STOP"))
    parser.add_argument("--bin-size", type=float, default=0.05)
    parser.add_argument("--response-window", nargs=2, type=float, default=(0.0, 1.0), metavar=("START", "STOP"))
    parser.add_argument("--max-units", type=int, default=10)
    parser.add_argument("--max-trials", type=int, default=120)
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    nwb_path = args.nwb.expanduser().resolve()
    if not nwb_path.exists():
        raise SystemExit(f"NWB file not found: {nwb_path}")
    out_dir = args.out_dir
    if out_dir is None:
        stem_dir = nwb_path.with_suffix("")
        out_dir = stem_dir.parent / f"{stem_dir.name}_mousehash_visuals"
    out_dir = ensure_dir(out_dir)
    artifacts: List[Dict[str, Any]] = []

    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        trials = read_trials(nwbfile)
        events = read_event_times(nwbfile)
        units = read_units(nwbfile)
        lick_df = read_lick_trace(nwbfile)
        if trials.empty:
            raise SystemExit("No trials table found. This script expects /intervals/trials.")
        if not units:
            raise SystemExit("No units/spike_times found. This script expects /units.")
        align_label, align_times = choose_alignment_times(trials, events, args.align)

        trial_csv = write_trial_table(trials, events, out_dir / "trial_table_with_events.csv")
        artifacts.append({"type": "analysis_view", "name": "trial_table_with_events", "path": str(trial_csv)})
        unit_csv = write_unit_summary(units, out_dir / "unit_summary.csv")
        artifacts.append({"type": "analysis_view", "name": "unit_summary", "path": str(unit_csv)})
        summary = summarize_nwb(nwbfile, nwb_path, trials, events, units)
        save_json(out_dir / "nwb_exploration_summary.json", summary)
        artifacts.append({"type": "summary", "name": "nwb_exploration_summary", "path": str(out_dir / "nwb_exploration_summary.json")})

        for name, func in [
            ("trial_event_overview", lambda: plot_trial_event_overview(trials, events, out_dir / "trial_event_overview.png")),
        ]:
            try:
                p = func()
                artifacts.append({"type": "figure", "name": name, "path": str(p)})
            except Exception as exc:
                artifacts.append({"type": "warning", "name": f"{name}_failed", "message": str(exc)})

        if lick_df is not None:
            try:
                p = plot_lick_trace(lick_df, out_dir / "lick_trace_overview.png")
                artifacts.append({"type": "figure", "name": "lick_trace_overview", "path": str(p)})
            except Exception as exc:
                artifacts.append({"type": "warning", "name": "lick_trace_failed", "message": str(exc)})

        group_by = args.group_by if args.group_by in trials.columns else None
        if args.group_by and group_by is None:
            artifacts.append({"type": "warning", "name": "group_by_missing", "message": f"Requested group-by column {args.group_by!r} not found; available columns: {list(trials.columns)}"})

        plot_calls = [
            ("raster_by_trial", lambda: plot_raster(units, trials, align_times, align_label, out_dir / f"raster_align-{safe_name(align_label)}_group-{safe_name(group_by or 'none')}.png", tuple(args.window), group_by, args.max_units, args.max_trials)),
            ("unit_psth", lambda: plot_psth(units, trials, align_times, align_label, out_dir / f"unit_psth_align-{safe_name(align_label)}_group-{safe_name(group_by or 'none')}.png", tuple(args.window), args.bin_size, group_by, args.max_units)),
            ("population_psth", lambda: plot_population_psth(units, trials, align_times, align_label, out_dir / f"population_psth_align-{safe_name(align_label)}_group-{safe_name(group_by or 'none')}.png", tuple(args.window), args.bin_size, group_by)),
            ("condition_response_heatmap", lambda: plot_condition_response_heatmap(units, trials, align_times, align_label, out_dir / f"condition_response_heatmap_align-{safe_name(align_label)}_group-{safe_name(group_by or 'none')}.png", tuple(args.response_window), group_by)),
        ]
        if not args.skip_pca:
            plot_calls.append(("pca_trajectories", lambda: plot_pca_trajectories(units, trials, align_times, align_label, out_dir / f"pca_trajectories_align-{safe_name(align_label)}_group-{safe_name(group_by or 'none')}.png", tuple(args.window), args.bin_size, group_by)))
        for name, func in plot_calls:
            try:
                p = func()
                if p is not None:
                    artifacts.append({"type": "figure", "name": name, "path": str(p)})
            except Exception as exc:
                artifacts.append({"type": "warning", "name": f"{name}_failed", "message": str(exc)})

    manifest = {
        "source_file": str(nwb_path),
        "out_dir": str(out_dir),
        "parameters": {
            "align": args.align,
            "resolved_align_label": align_label,
            "group_by": group_by,
            "window": list(args.window),
            "bin_size": args.bin_size,
            "response_window": list(args.response_window),
            "max_units": args.max_units,
            "max_trials": args.max_trials,
        },
        "analysis_views": {
            "trial_table": "trial_table_with_events.csv",
            "unit_summary": "unit_summary.csv",
            "binned_spike_tensor": {
                "materialized_in_memory": True,
                "shape_convention": "trials x time_bins x units",
                "produced_by": [
                    "alignment.align_spikes_to_trial_events",
                    "segmentation.extract_event_locked_windows",
                    "binning.bin_spikes",
                ],
            },
        },
        "artifacts": artifacts,
        "notes": [
            "Exploratory only; not train/test-safe.",
            "Perturbation labels may require paper-level or per-trial metadata disambiguation.",
            "For 000011-like files with few units, PCA trajectories are descriptive rather than definitive.",
        ],
    }
    save_json(out_dir / "mousehash_visual_artifacts_manifest.json", manifest)
    if not args.quiet:
        print(f"[ok] wrote artifacts to: {out_dir}")
        for a in artifacts:
            if a.get("type") in {"figure", "analysis_view", "summary"}:
                print(f"  - {a['name']}: {a['path']}")
            elif a.get("type") == "warning":
                print(f"  ! {a['name']}: {a.get('message')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
