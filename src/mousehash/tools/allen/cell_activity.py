from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mousehash.artifacts.io import load_json, load_npy, save_json, save_npy
from mousehash.artifacts.paths import reports_root
from mousehash.tools.allen.stimulus_fetch import fetch_natural_scene_template
from mousehash.tools.representations.animate_inanimate import (
    derive_animate_inanimate,
    derive_top1,
)
from mousehash.tools.representations.vit_imagenet import run_vit_on_frames

DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "cpu"
DEFAULT_THRESHOLD_MAX_CLASS_IDX = 397

ANIMATE_COLOR = "#1f77b4"
INANIMATE_COLOR = "#ff7f0e"
CONTEXT_COLOR = "#b0b7c3"
VANILLA_COLOR = "#2f4858"


def slice_trace_time_range(
    timestamps: np.ndarray,
    dff_trace: np.ndarray,
    *,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
    timepoint_labels: np.ndarray | None = None,
) -> dict:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    dff_trace = np.asarray(dff_trace, dtype=np.float32)

    if len(timestamps) != len(dff_trace):
        raise ValueError("timestamps and dff_trace must have the same length")
    if timepoint_labels is not None and len(timepoint_labels) != len(timestamps):
        raise ValueError("timepoint_labels must have the same length as timestamps")
    if time_start_s is not None and time_end_s is not None and float(time_end_s) <= float(time_start_s):
        raise ValueError("time_end_s must be greater than time_start_s")

    mask = np.ones(len(timestamps), dtype=bool)
    if time_start_s is not None:
        mask &= timestamps >= float(time_start_s)
    if time_end_s is not None:
        mask &= timestamps <= float(time_end_s)

    sliced_labels = None if timepoint_labels is None else np.asarray(timepoint_labels, dtype=np.int8)[mask]
    return {
        "timestamps": timestamps[mask],
        "dff_trace": dff_trace[mask],
        "timepoint_labels": sliced_labels,
        "time_start_s": None if time_start_s is None else float(time_start_s),
        "time_end_s": None if time_end_s is None else float(time_end_s),
    }


def _slugify_cache_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "default"


def _natural_scene_label_cache_paths(
    threshold_max_class_idx: int,
    model_name: str,
    device: str,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    if cache_dir is None:
        cache_dir = reports_root() / "cell_activity" / "cache"

    cache_dir = Path(cache_dir)
    stem = (
        "natural_scenes_animate_inanimate"
        f"__model_{_slugify_cache_component(model_name)}"
        f"__device_{_slugify_cache_component(device)}"
        f"__threshold_{int(threshold_max_class_idx)}"
    )
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _require_allensdk():
    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    except ImportError as exc:
        raise ImportError(
            "allensdk is required for Allen cell activity tools. "
            "Install it with: pip install allensdk"
        ) from exc

    return BrainObservatoryCache


def _normalize_single_trace(dff_traces: object) -> np.ndarray:
    traces = np.asarray(dff_traces, dtype=object)

    if traces.ndim == 0:
        return np.asarray(traces.item(), dtype=np.float32)

    if traces.ndim == 1:
        first = traces[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return np.asarray(first, dtype=np.float32)
        return np.asarray(traces, dtype=np.float32)

    return np.asarray(traces[0], dtype=np.float32)


def _select_natural_scenes_experiment(boc, cell_specimen_id: int) -> dict:
    experiments = boc.get_ophys_experiments(
        cell_specimen_ids=[int(cell_specimen_id)],
        stimuli=["natural_scenes"],
    )
    if not experiments:
        raise ValueError(
            "No natural-scenes ophys experiment found for "
            f"cell_specimen_id={cell_specimen_id}"
        )

    experiments = sorted(experiments, key=lambda row: int(row["id"]))
    return experiments[0]


def _lookup_cell_record(boc, cell_specimen_id: int) -> dict:
    cells = pd.DataFrame.from_records(boc.get_cell_specimens())
    matches = cells[cells["cell_specimen_id"].astype(int) == int(cell_specimen_id)]
    if matches.empty:
        raise ValueError(f"Cell specimen {cell_specimen_id} was not found in the manifest")
    return matches.iloc[0].to_dict()


def compute_natural_scene_animate_inanimate_labels(
    manifest_path: Path,
    threshold_max_class_idx: int = DEFAULT_THRESHOLD_MAX_CLASS_IDX,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
) -> dict:
    frames = fetch_natural_scene_template(Path(manifest_path))
    _, probabilities = run_vit_on_frames(
        frames,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    top1 = derive_top1(probabilities)
    animate_inanimate = derive_animate_inanimate(top1, threshold_max_class_idx)
    return {
        "animate_inanimate": animate_inanimate.astype(np.int8),
        "top1": top1.astype(np.int32),
        "n_images": int(len(animate_inanimate)),
        "threshold_max_class_idx": int(threshold_max_class_idx),
        "model_name": model_name,
        "device": device,
    }


def load_or_compute_natural_scene_animate_inanimate_labels(
    manifest_path: Path,
    threshold_max_class_idx: int = DEFAULT_THRESHOLD_MAX_CLASS_IDX,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    *,
    cache_dir: Path | None = None,
    force_recompute: bool = False,
) -> dict:
    labels_path, metadata_path = _natural_scene_label_cache_paths(
        threshold_max_class_idx=threshold_max_class_idx,
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )

    if not force_recompute and labels_path.exists() and metadata_path.exists():
        metadata = load_json(metadata_path)
        animate_inanimate = load_npy(labels_path).astype(np.int8)
        return {
            "animate_inanimate": animate_inanimate,
            "top1": None,
            "n_images": int(metadata["n_images"]),
            "threshold_max_class_idx": int(metadata["threshold_max_class_idx"]),
            "model_name": str(metadata["model_name"]),
            "device": str(metadata["device"]),
            "cache_path": str(labels_path),
            "cache_metadata_path": str(metadata_path),
            "from_cache": True,
        }

    label_data = compute_natural_scene_animate_inanimate_labels(
        manifest_path=manifest_path,
        threshold_max_class_idx=threshold_max_class_idx,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    save_npy(labels_path, label_data["animate_inanimate"])
    save_json(
        metadata_path,
        {
            "n_images": int(label_data["n_images"]),
            "threshold_max_class_idx": int(label_data["threshold_max_class_idx"]),
            "model_name": label_data["model_name"],
            "device": label_data["device"],
        },
    )
    label_data["cache_path"] = str(labels_path)
    label_data["cache_metadata_path"] = str(metadata_path)
    label_data["from_cache"] = False
    return label_data


def build_timepoint_scene_labels(
    n_timepoints: int,
    stimulus_table: pd.DataFrame,
    animate_inanimate: np.ndarray,
) -> np.ndarray:
    labels = np.full(int(n_timepoints), -1, dtype=np.int8)
    if stimulus_table.empty:
        return labels

    required_columns = {"start", "end", "frame"}
    missing = required_columns.difference(stimulus_table.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Stimulus table missing required columns: {missing_cols}")

    n_frames = len(animate_inanimate)
    for row in stimulus_table.itertuples(index=False):
        if pd.isna(row.frame):
            continue

        frame_idx = int(row.frame)
        if frame_idx < 0 or frame_idx >= n_frames:
            continue

        start_idx = max(int(row.start), 0)
        end_idx = min(int(row.end), int(n_timepoints))
        if end_idx <= start_idx:
            continue

        labels[start_idx:end_idx] = int(animate_inanimate[frame_idx])

    return labels


def fetch_cell_natural_scenes_trace(
    manifest_path: Path,
    cell_specimen_id: int,
) -> dict:
    BrainObservatoryCache = _require_allensdk()
    manifest_path = Path(manifest_path)
    boc = BrainObservatoryCache(manifest_file=str(manifest_path))

    cell_record = _lookup_cell_record(boc, cell_specimen_id)
    experiment = _select_natural_scenes_experiment(boc, cell_specimen_id)
    experiment_id = int(experiment["id"])
    data_set = boc.get_ophys_experiment_data(experiment_id)
    timestamps, dff_traces = data_set.get_dff_traces(cell_specimen_ids=[int(cell_specimen_id)])
    dff_trace = _normalize_single_trace(dff_traces)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) != len(dff_trace):
        raise ValueError(
            "AllenSDK returned mismatched dF/F timestamps and trace lengths: "
            f"timestamps={len(timestamps)} trace={len(dff_trace)}"
        )

    stimulus_table = data_set.get_stimulus_table(stimulus_name="natural_scenes").copy()
    return {
        "cell_specimen_id": int(cell_specimen_id),
        "cell_record": cell_record,
        "experiment_id": experiment_id,
        "experiment_container_id": int(experiment.get("experiment_container_id", -1)),
        "timestamps": timestamps,
        "dff_trace": dff_trace.astype(np.float32),
        "stimulus_table": stimulus_table,
    }


def build_cell_dff_plot(
    timestamps: np.ndarray,
    dff_trace: np.ndarray,
    output_path: Path,
    *,
    title: str,
    line_color: str = VANILLA_COLOR,
    trace_name: str = "dF/F",
) -> Path:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    dff_trace = np.asarray(dff_trace, dtype=np.float32)

    if len(timestamps) != len(dff_trace):
        raise ValueError("timestamps and dff_trace must have the same length")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=dff_trace,
            mode="lines",
            name=trace_name,
            line={"color": line_color, "width": 2},
        )
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="dF/F",
        hovermode="x unified",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def build_cell_dff_matplotlib_plot(
    timestamps: np.ndarray,
    dff_trace: np.ndarray,
    output_path: Path,
    *,
    title: str,
    line_color: str = VANILLA_COLOR,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps = np.asarray(timestamps, dtype=np.float64)
    dff_trace = np.asarray(dff_trace, dtype=np.float32)

    if len(timestamps) != len(dff_trace):
        raise ValueError("timestamps and dff_trace must have the same length")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.plot(timestamps, dff_trace, color=line_color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_cell_dff_animate_inanimate_plot(
    timestamps: np.ndarray,
    dff_trace: np.ndarray,
    timepoint_labels: np.ndarray,
    output_path: Path,
    *,
    title: str,
    animate_color: str = ANIMATE_COLOR,
    inanimate_color: str = INANIMATE_COLOR,
    context_color: str = CONTEXT_COLOR,
) -> Path:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    dff_trace = np.asarray(dff_trace, dtype=np.float32)
    timepoint_labels = np.asarray(timepoint_labels, dtype=np.int8)

    if not (len(timestamps) == len(dff_trace) == len(timepoint_labels)):
        raise ValueError("timestamps, dff_trace, and timepoint_labels must have the same length")

    animate_mask = timepoint_labels == 1
    inanimate_mask = timepoint_labels == 0
    context_mask = timepoint_labels < 0

    def masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.where(mask, values, np.nan)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=masked(dff_trace, context_mask),
            mode="lines",
            name="outside natural scenes",
            line={"color": context_color, "width": 1},
            opacity=0.45,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=masked(dff_trace, inanimate_mask),
            mode="lines",
            name="inanimate scene",
            line={"color": inanimate_color, "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=masked(dff_trace, animate_mask),
            mode="lines",
            name="animate scene",
            line={"color": animate_color, "width": 2},
        )
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="dF/F",
        hovermode="x unified",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def build_cell_dff_animate_inanimate_matplotlib_plot(
    timestamps: np.ndarray,
    dff_trace: np.ndarray,
    timepoint_labels: np.ndarray,
    output_path: Path,
    *,
    title: str,
    animate_color: str = ANIMATE_COLOR,
    inanimate_color: str = INANIMATE_COLOR,
    context_color: str = CONTEXT_COLOR,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps = np.asarray(timestamps, dtype=np.float64)
    dff_trace = np.asarray(dff_trace, dtype=np.float32)
    timepoint_labels = np.asarray(timepoint_labels, dtype=np.int8)

    if not (len(timestamps) == len(dff_trace) == len(timepoint_labels)):
        raise ValueError("timestamps, dff_trace, and timepoint_labels must have the same length")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.plot(
        timestamps,
        np.where(timepoint_labels < 0, dff_trace, np.nan),
        color=context_color,
        linewidth=0.8,
        alpha=0.6,
        label="outside natural scenes",
    )
    ax.plot(
        timestamps,
        np.where(timepoint_labels == 0, dff_trace, np.nan),
        color=inanimate_color,
        linewidth=1.6,
        label="inanimate scene",
    )
    ax.plot(
        timestamps,
        np.where(timepoint_labels == 1, dff_trace, np.nan),
        color=animate_color,
        linewidth=1.6,
        label="animate scene",
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def analyze_cell_dff_against_animate_inanimate(
    manifest_path: Path,
    cell_specimen_id: int,
    *,
    threshold_max_class_idx: int = DEFAULT_THRESHOLD_MAX_CLASS_IDX,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    output_path: Path | None = None,
    cache_dir: Path | None = None,
    force_recompute_labels: bool = False,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
    animate_color: str = ANIMATE_COLOR,
    inanimate_color: str = INANIMATE_COLOR,
    context_color: str = CONTEXT_COLOR,
) -> dict:
    trace_data = fetch_cell_natural_scenes_trace(manifest_path, cell_specimen_id)
    label_data = load_or_compute_natural_scene_animate_inanimate_labels(
        manifest_path=manifest_path,
        threshold_max_class_idx=threshold_max_class_idx,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        force_recompute=force_recompute_labels,
    )

    timepoint_labels = build_timepoint_scene_labels(
        n_timepoints=len(trace_data["timestamps"]),
        stimulus_table=trace_data["stimulus_table"],
        animate_inanimate=label_data["animate_inanimate"],
    )
    sliced = slice_trace_time_range(
        trace_data["timestamps"],
        trace_data["dff_trace"],
        time_start_s=time_start_s,
        time_end_s=time_end_s,
        timepoint_labels=timepoint_labels,
    )

    if output_path is None:
        output_path = (
            reports_root()
            / "cell_activity"
            / f"cell_{int(cell_specimen_id)}_natural_scenes_animate_inanimate.html"
        )
    png_output_path = Path(output_path).with_suffix(".png")

    plot_path = build_cell_dff_animate_inanimate_plot(
        timestamps=sliced["timestamps"],
        dff_trace=sliced["dff_trace"],
        timepoint_labels=sliced["timepoint_labels"],
        output_path=Path(output_path),
        title=(
            f"Cell {int(cell_specimen_id)} dF/F during natural scenes "
            "(blue=animate, orange=inanimate)"
        ),
        animate_color=animate_color,
        inanimate_color=inanimate_color,
        context_color=context_color,
    )
    png_plot_path = build_cell_dff_animate_inanimate_matplotlib_plot(
        timestamps=sliced["timestamps"],
        dff_trace=sliced["dff_trace"],
        timepoint_labels=sliced["timepoint_labels"],
        output_path=png_output_path,
        title=(
            f"Cell {int(cell_specimen_id)} dF/F during natural scenes "
            "(blue=animate, orange=inanimate)"
        ),
        animate_color=animate_color,
        inanimate_color=inanimate_color,
        context_color=context_color,
    )

    assert sliced["timepoint_labels"] is not None
    n_animate = int(np.count_nonzero(sliced["timepoint_labels"] == 1))
    n_inanimate = int(np.count_nonzero(sliced["timepoint_labels"] == 0))
    n_other = int(np.count_nonzero(sliced["timepoint_labels"] < 0))

    return {
        "cell_specimen_id": int(cell_specimen_id),
        "experiment_id": trace_data["experiment_id"],
        "experiment_container_id": trace_data["experiment_container_id"],
        "plot_path": str(plot_path),
        "plot_png_path": str(png_plot_path),
        "n_timepoints": int(len(sliced["timestamps"])),
        "n_animate_timepoints": n_animate,
        "n_inanimate_timepoints": n_inanimate,
        "n_other_timepoints": n_other,
        "threshold_max_class_idx": int(threshold_max_class_idx),
        "model_name": model_name,
        "device": device,
        "label_cache_path": label_data["cache_path"],
        "label_cache_metadata_path": label_data["cache_metadata_path"],
        "labels_from_cache": bool(label_data["from_cache"]),
        "time_start_s": sliced["time_start_s"],
        "time_end_s": sliced["time_end_s"],
        "animate_color": animate_color,
        "inanimate_color": inanimate_color,
        "context_color": context_color,
    }


def analyze_cell_dff_vanilla(
    manifest_path: Path,
    cell_specimen_id: int,
    *,
    output_path: Path | None = None,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
    line_color: str = VANILLA_COLOR,
) -> dict:
    trace_data = fetch_cell_natural_scenes_trace(manifest_path, cell_specimen_id)
    sliced = slice_trace_time_range(
        trace_data["timestamps"],
        trace_data["dff_trace"],
        time_start_s=time_start_s,
        time_end_s=time_end_s,
    )

    if output_path is None:
        output_path = (
            reports_root()
            / "cell_activity"
            / f"cell_{int(cell_specimen_id)}_natural_scenes_vanilla.html"
        )
    png_output_path = Path(output_path).with_suffix(".png")

    plot_path = build_cell_dff_plot(
        timestamps=sliced["timestamps"],
        dff_trace=sliced["dff_trace"],
        output_path=Path(output_path),
        title=f"Cell {int(cell_specimen_id)} dF/F during natural scenes",
        line_color=line_color,
    )
    png_plot_path = build_cell_dff_matplotlib_plot(
        timestamps=sliced["timestamps"],
        dff_trace=sliced["dff_trace"],
        output_path=png_output_path,
        title=f"Cell {int(cell_specimen_id)} dF/F during natural scenes",
        line_color=line_color,
    )

    return {
        "cell_specimen_id": int(cell_specimen_id),
        "experiment_id": trace_data["experiment_id"],
        "experiment_container_id": trace_data["experiment_container_id"],
        "plot_path": str(plot_path),
        "plot_png_path": str(png_plot_path),
        "n_timepoints": int(len(sliced["timestamps"])),
        "time_start_s": sliced["time_start_s"],
        "time_end_s": sliced["time_end_s"],
        "line_color": line_color,
    }