from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_nmf_report(
    scores: np.ndarray,
    components: np.ndarray,
    animate_inanimate: np.ndarray,
    image_paths: list[Path],
    output_path: Path,
    reconstruction_err: float | None = None,
    title: str = "NMF Explorer",
    class_labels: list[str] | None = None,
) -> None:
    """Generate a self-contained interactive HTML NMF explorer.

    The report contains:
    - Component 0 vs Component 1 scatter colored by animate/inanimate.
    - Heatmap of all image scores across all components.
    - Per-component top-10 class loadings (non-negative by construction).

    Args:
        scores:             (n_images, n_components) NMF activation matrix W.
        components:         (n_components, n_classes) basis matrix H.
        animate_inanimate:  (n_images,) int8 array (1=animate, 0=inanimate).
        image_paths:        Ordered list of image file paths for hover labels.
        output_path:        Destination HTML file.
        reconstruction_err: Optional Frobenius reconstruction error for subtitle.
        title:              Page / report title.
    """
    import plotly.graph_objects as go

    n_components = scores.shape[1]
    labels = ["animate" if a else "inanimate" for a in animate_inanimate]
    color_map = {"animate": "#e06c75", "inanimate": "#61afef"}
    hover = [f"img {i}<br>{Path(p).name}" for i, p in enumerate(image_paths)]

    figs: list[go.Figure] = []

    # 1. Component 0 vs 1 scatter
    fig_scatter = go.Figure()
    for label, marker_color in color_map.items():
        mask = np.array(labels) == label
        fig_scatter.add_trace(
            go.Scatter(
                x=scores[mask, 0].tolist(),
                y=scores[mask, 1].tolist(),
                mode="markers",
                name=label,
                marker=dict(color=marker_color, size=8, opacity=0.8),
                text=[hover[i] for i in np.where(mask)[0]],
                hovertemplate="%{text}<br>C0=%{x:.3f}, C1=%{y:.3f}<extra></extra>",
            )
        )
    subtitle = f" (reconstruction err={reconstruction_err:.4f})" if reconstruction_err else ""
    fig_scatter.update_layout(
        title=f"NMF component 0 vs 1{subtitle}",
        xaxis_title="Component 0",
        yaxis_title="Component 1",
        template="plotly_dark",
        legend_title="Category",
    )
    figs.append(fig_scatter)

    # 2. Score heatmap: images × components
    fig_heat = go.Figure(
        go.Heatmap(
            z=scores.tolist(),
            x=[f"C{i}" for i in range(n_components)],
            y=[f"img {i}" for i in range(scores.shape[0])],
            colorscale="Viridis",
            hovertemplate="img=%{y}, comp=%{x}<br>score=%{z:.3f}<extra></extra>",
        )
    )
    fig_heat.update_layout(
        title="Image × component activation heatmap",
        xaxis_title="NMF component",
        yaxis_title="Image index",
        template="plotly_dark",
    )
    figs.append(fig_heat)

    # 3. Top class loadings per component
    use_labels = class_labels is not None
    x_axis_title = "ImageNet class" if use_labels else "ImageNet class index"
    for c_idx in range(min(n_components, 5)):
        loadings = components[c_idx]
        top_idx = np.argsort(loadings)[-15:][::-1]
        top_vals = loadings[top_idx]

        if use_labels:
            x_ticks = [class_labels[int(i)] for i in top_idx]
            hover_lbls = [f"cls {int(i)}: {class_labels[int(i)]}" for i in top_idx]
        else:
            x_ticks = [f"cls {int(i)}" for i in top_idx]
            hover_lbls = x_ticks

        fig_load = go.Figure(
            go.Bar(
                x=x_ticks,
                y=top_vals.tolist(),
                marker_color="#98c379",
                customdata=hover_lbls,
                hovertemplate="%{customdata}<br>loading=%{y:.3f}<extra></extra>",
            )
        )
        fig_load.update_layout(
            title=f"NMF component {c_idx} — top 15 class loadings",
            xaxis_title=x_axis_title,
            yaxis_title="Loading",
            template="plotly_dark",
        )
        figs.append(fig_load)

    # Assemble
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_parts = [
        f"<html><head><title>{title}</title></head>",
        "<body style='background:#282c34;color:#abb2bf;font-family:sans-serif;'>",
        f"<h1 style='padding:16px'>{title}</h1>",
    ]
    for i, fig in enumerate(figs):
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("Wrote NMF report to %s", output_path)
