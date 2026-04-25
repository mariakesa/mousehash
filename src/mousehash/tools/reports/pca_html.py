from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_pca_report(
    scores: np.ndarray,
    components: np.ndarray,
    explained_variance_ratio: np.ndarray,
    animate_inanimate: np.ndarray,
    image_paths: list[Path],
    output_path: Path,
    title: str = "PCA Explorer",
    class_labels: list[str] | None = None,
) -> None:
    """Generate a self-contained interactive HTML PCA explorer.

    The report contains:
    - Scree plot of explained variance.
    - PC1 vs PC2 scatter colored by animate/inanimate, with image-path hover.
    - Per-component top-10 positive and negative class-index loadings (bar chart).

    Args:
        scores:                   (n_images, n_components) PCA projections.
        components:               (n_components, n_classes) component matrix.
        explained_variance_ratio: (n_components,) explained variance fractions.
        animate_inanimate:        (n_images,) int8 array (1=animate, 0=inanimate).
        image_paths:              Ordered list of image file paths for hover labels.
        output_path:              Destination HTML file.
        title:                    Page / report title.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_components = scores.shape[1]
    labels = ["animate" if a else "inanimate" for a in animate_inanimate]
    color_map = {"animate": "#e06c75", "inanimate": "#61afef"}
    colors = [color_map[l] for l in labels]
    hover = [f"img {i}<br>{Path(p).name}" for i, p in enumerate(image_paths)]

    figs: list[go.Figure] = []

    # 1. Scree plot
    fig_scree = go.Figure(
        go.Bar(
            x=[f"PC{i+1}" for i in range(n_components)],
            y=(explained_variance_ratio * 100).tolist(),
            marker_color="#98c379",
        )
    )
    fig_scree.update_layout(
        title="Explained variance per PC",
        xaxis_title="Component",
        yaxis_title="% variance",
        template="plotly_dark",
    )
    figs.append(fig_scree)

    # 2. PC1 vs PC2 scatter
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
                hovertemplate="%{text}<br>PC1=%{x:.2f}, PC2=%{y:.2f}<extra></extra>",
            )
        )
    fig_scatter.update_layout(
        title="PC1 vs PC2 (colored by animate/inanimate)",
        xaxis_title=f"PC1 ({explained_variance_ratio[0]:.1%} var)",
        yaxis_title=f"PC2 ({explained_variance_ratio[1]:.1%} var)",
        template="plotly_dark",
        legend_title="Category",
    )
    figs.append(fig_scatter)

    # 3. Top loading bar charts for each PC
    use_labels = class_labels is not None
    x_axis_title = "ImageNet class" if use_labels else "ImageNet class index"
    for pc_idx in range(min(n_components, 5)):
        loadings = components[pc_idx]
        top_pos = np.argsort(loadings)[-10:][::-1]
        top_neg = np.argsort(loadings)[:10]
        top_idx = np.concatenate([top_pos, top_neg])
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
                marker_color=["#e06c75" if v > 0 else "#61afef" for v in top_vals],
                customdata=hover_lbls,
                hovertemplate="%{customdata}<br>loading=%{y:.3f}<extra></extra>",
            )
        )
        fig_load.update_layout(
            title=f"PC{pc_idx+1} top loadings (pos/neg)",
            xaxis_title=x_axis_title,
            yaxis_title="Loading",
            template="plotly_dark",
        )
        figs.append(fig_load)

    # Assemble into one HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_parts = [
        f"<html><head><title>{title}</title></head>",
        "<body style='background:#282c34;color:#abb2bf;font-family:sans-serif;'>",
        f"<h1 style='padding:16px'>{title}</h1>",
    ]
    for i, fig in enumerate(figs):
        include_js = i == 0  # embed Plotly JS only once
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("Wrote PCA report to %s", output_path)
