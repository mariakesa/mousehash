from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mousehash.tools.allen.cell_activity import (
    build_cell_dff_plot,
    build_cell_dff_matplotlib_plot,
    build_timepoint_scene_labels,
    load_or_compute_natural_scene_animate_inanimate_labels,
    slice_trace_time_range,
)


def test_build_timepoint_scene_labels_marks_animate_inanimate_and_other() -> None:
    stimulus_table = pd.DataFrame(
        {
            "start": [0, 3, 5, 8],
            "end": [3, 5, 8, 11],
            "frame": [2, -1, 1, 0],
        }
    )

    labels = build_timepoint_scene_labels(
        n_timepoints=10,
        stimulus_table=stimulus_table,
        animate_inanimate=np.array([1, 0, 0], dtype=np.int8),
    )

    assert labels.tolist() == [0, 0, 0, -1, -1, 0, 0, 0, 1, 1]


def test_load_or_compute_scene_labels_reuses_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def fake_compute(*args, **kwargs):
        calls["count"] += 1
        return {
            "animate_inanimate": np.array([1, 0, 1], dtype=np.int8),
            "top1": np.array([5, 600, 12], dtype=np.int32),
            "n_images": 3,
            "threshold_max_class_idx": 397,
            "model_name": "fake-model",
            "device": "cpu",
        }

    monkeypatch.setattr(
        "mousehash.tools.allen.cell_activity.compute_natural_scene_animate_inanimate_labels",
        fake_compute,
    )

    first = load_or_compute_natural_scene_animate_inanimate_labels(
        manifest_path=tmp_path / "manifest.json",
        model_name="fake-model",
        cache_dir=tmp_path,
    )
    second = load_or_compute_natural_scene_animate_inanimate_labels(
        manifest_path=tmp_path / "manifest.json",
        model_name="fake-model",
        cache_dir=tmp_path,
    )

    assert calls["count"] == 1
    assert first["from_cache"] is False
    assert second["from_cache"] is True
    assert second["cache_path"].endswith(".npy")
    assert np.array_equal(second["animate_inanimate"], np.array([1, 0, 1], dtype=np.int8))


def test_build_cell_dff_plot_writes_html(tmp_path: Path) -> None:
    output_path = tmp_path / "vanilla_plot.html"

    result = build_cell_dff_plot(
        timestamps=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        dff_trace=np.array([0.1, 0.2, 0.15], dtype=np.float32),
        output_path=output_path,
        title="Vanilla trace",
    )

    assert result == output_path
    html = output_path.read_text(encoding="utf-8")
    assert "Vanilla trace" in html
    assert "plotly" in html.lower()


def test_build_cell_dff_plot_uses_custom_color(tmp_path: Path) -> None:
    output_path = tmp_path / "vanilla_plot_custom.html"

    build_cell_dff_plot(
        timestamps=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        dff_trace=np.array([0.1, 0.2, 0.15], dtype=np.float32),
        output_path=output_path,
        title="Vanilla trace",
        line_color="#123456",
    )

    html = output_path.read_text(encoding="utf-8")
    assert "#123456" in html


def test_build_cell_dff_matplotlib_plot_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "vanilla_plot.png"

    result = build_cell_dff_matplotlib_plot(
        timestamps=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        dff_trace=np.array([0.1, 0.2, 0.15], dtype=np.float32),
        output_path=output_path,
        title="Vanilla trace",
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_slice_trace_time_range_clips_trace_and_labels() -> None:
    sliced = slice_trace_time_range(
        timestamps=np.array([1000.0, 1499.0, 1500.0, 1750.0, 2000.0, 2100.0]),
        dff_trace=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        time_start_s=1500.0,
        time_end_s=2000.0,
        timepoint_labels=np.array([-1, 0, 1, 1, 0, -1], dtype=np.int8),
    )

    assert sliced["timestamps"].tolist() == [1500.0, 1750.0, 2000.0]
    assert np.allclose(sliced["dff_trace"], np.array([0.3, 0.4, 0.5], dtype=np.float32))
    assert sliced["timepoint_labels"].tolist() == [1, 1, 0]


def test_slice_trace_time_range_rejects_invalid_bounds() -> None:
    try:
        slice_trace_time_range(
            timestamps=np.array([0.0, 1.0]),
            dff_trace=np.array([0.1, 0.2], dtype=np.float32),
            time_start_s=2.0,
            time_end_s=1.0,
        )
    except ValueError as exc:
        assert "time_end_s must be greater" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid time bounds")