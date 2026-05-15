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
    image_thumbs: list[str] | None = None,
) -> None:
    """Generate a self-contained interactive HTML PCA explorer.

    The report contains:
    - Scree plot of explained variance.
    - Scatter colored by animate/inanimate, with X/Y component dropdowns.
    - Per-component top-10 positive and negative class loadings for every PC.

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

    # 2. Interactive scatter: pick X / Y PC from dropdowns
    labels_arr = np.array(labels)
    masks = {label: labels_arr == label for label in color_map}
    hovers = {label: [hover[i] for i in np.where(masks[label])[0]] for label in color_map}
    indices = {label: np.where(masks[label])[0].tolist() for label in color_map}
    per_class_scores = {label: scores[masks[label]] for label in color_map}

    init_x = 0
    init_y = 1 if n_components > 1 else 0

    def pc_axis_title(idx: int) -> str:
        return f"PC{idx + 1} ({explained_variance_ratio[idx]:.1%} var)"

    fig_scatter = go.Figure()
    for label, marker_color in color_map.items():
        fig_scatter.add_trace(
            go.Scatter(
                x=per_class_scores[label][:, init_x].tolist(),
                y=per_class_scores[label][:, init_y].tolist(),
                mode="markers",
                name=label,
                marker=dict(color=marker_color, size=8, opacity=0.8),
                text=hovers[label],
                customdata=indices[label],
                hovertemplate="%{text}<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
            )
        )

    trace_order = list(color_map.keys())
    x_buttons = [
        dict(
            label=f"PC{i + 1}",
            method="update",
            args=[
                {"x": [per_class_scores[lbl][:, i].tolist() for lbl in trace_order]},
                {"xaxis.title.text": pc_axis_title(i)},
            ],
        )
        for i in range(n_components)
    ]
    y_buttons = [
        dict(
            label=f"PC{i + 1}",
            method="update",
            args=[
                {"y": [per_class_scores[lbl][:, i].tolist() for lbl in trace_order]},
                {"yaxis.title.text": pc_axis_title(i)},
            ],
        )
        for i in range(n_components)
    ]

    fig_scatter.update_layout(
        title="PCA scatter — pick X / Y components (colored by animate/inanimate)",
        xaxis_title=pc_axis_title(init_x),
        yaxis_title=pc_axis_title(init_y),
        template="plotly_dark",
        legend_title="Category",
        margin=dict(t=120),
        updatemenus=[
            dict(
                buttons=x_buttons,
                direction="down",
                x=0.13, xanchor="left",
                y=1.16, yanchor="top",
                active=init_x,
                showactive=True,
            ),
            dict(
                buttons=y_buttons,
                direction="down",
                x=0.33, xanchor="left",
                y=1.16, yanchor="top",
                active=init_y,
                showactive=True,
            ),
        ],
        annotations=[
            dict(text="X PC:", x=0.07, xref="paper", y=1.14, yref="paper",
                 showarrow=False, xanchor="right"),
            dict(text="Y PC:", x=0.27, xref="paper", y=1.14, yref="paper",
                 showarrow=False, xanchor="right"),
        ],
    )
    figs.append(fig_scatter)

    # 3. Top loading bar charts for each PC
    use_labels = class_labels is not None
    x_axis_title = "ImageNet class" if use_labels else "ImageNet class index"
    for pc_idx in range(n_components):
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
    scatter_idx = 1  # figs[0] is the scree plot, figs[1] is the scatter
    div_ids = {scatter_idx: "pca-scatter"} if image_thumbs is not None else {}
    for i, fig in enumerate(figs):
        include_js = i == 0  # embed Plotly JS only once
        html_parts.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs=include_js,
                **({"div_id": div_ids[i]} if i in div_ids else {}),
            )
        )
    if image_thumbs is not None:
        from mousehash.tools.reports.nmf_html import _hover_thumb_html
        html_parts.append(_hover_thumb_html(image_thumbs, ["pca-scatter"]))
    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("Wrote PCA report to %s", output_path)
