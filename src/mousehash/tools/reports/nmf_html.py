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
    image_thumbs: list[str] | None = None,
) -> None:
    """Generate a self-contained interactive HTML NMF explorer.

    The report contains:
    - Scatter colored by animate/inanimate, with X/Y component dropdowns.
    - Heatmap of all image scores across all components.
    - Per-component top-15 class loadings for every component.

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

    # 1. Interactive scatter: pick X / Y component from dropdowns
    labels_arr = np.array(labels)
    masks = {label: labels_arr == label for label in color_map}
    hovers = {label: [hover[i] for i in np.where(masks[label])[0]] for label in color_map}
    indices = {label: np.where(masks[label])[0].tolist() for label in color_map}
    # Per-class scores stacked once: shape (n_in_class, n_components) per label.
    per_class_scores = {label: scores[masks[label]] for label in color_map}

    init_x = 0
    init_y = 1 if n_components > 1 else 0
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
                hovertemplate="%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
            )
        )

    trace_order = list(color_map.keys())  # matches add_trace order above
    x_buttons = [
        dict(
            label=f"C{i}",
            method="update",
            args=[
                {"x": [per_class_scores[lbl][:, i].tolist() for lbl in trace_order]},
                {"xaxis.title.text": f"Component {i}"},
            ],
        )
        for i in range(n_components)
    ]
    y_buttons = [
        dict(
            label=f"C{i}",
            method="update",
            args=[
                {"y": [per_class_scores[lbl][:, i].tolist() for lbl in trace_order]},
                {"yaxis.title.text": f"Component {i}"},
            ],
        )
        for i in range(n_components)
    ]

    subtitle = f" (reconstruction err={reconstruction_err:.4f})" if reconstruction_err else ""
    fig_scatter.update_layout(
        title=f"NMF scatter — pick X / Y components{subtitle}",
        xaxis_title=f"Component {init_x}",
        yaxis_title=f"Component {init_y}",
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
            dict(text="X comp:", x=0.07, xref="paper", y=1.14, yref="paper",
                 showarrow=False, xanchor="right"),
            dict(text="Y comp:", x=0.27, xref="paper", y=1.14, yref="paper",
                 showarrow=False, xanchor="right"),
        ],
    )
    figs.append(fig_scatter)

    # 2. Score heatmap: images × components
    n_images = scores.shape[0]
    heat_customdata = [[i] * n_components for i in range(n_images)]
    fig_heat = go.Figure(
        go.Heatmap(
            z=scores.tolist(),
            x=[f"C{i}" for i in range(n_components)],
            y=[f"img {i}" for i in range(n_images)],
            customdata=heat_customdata,
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
    for c_idx in range(n_components):
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
    div_ids = {0: "nmf-scatter", 1: "nmf-heatmap"} if image_thumbs is not None else {}
    for i, fig in enumerate(figs):
        html_parts.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs=(i == 0),
                **({"div_id": div_ids[i]} if i in div_ids else {}),
            )
        )
    if image_thumbs is not None:
        html_parts.append(_hover_thumb_html(image_thumbs, ["nmf-scatter", "nmf-heatmap"]))
    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("Wrote NMF report to %s", output_path)


def _hover_thumb_html(image_thumbs: list[str], target_ids: list[str]) -> str:
    """Floating <img> + Plotly hover handler that swaps it to the hovered image's thumbnail."""
    import json
    return (
        "<img id=\"hover-thumb\" alt=\"\" "
        "style=\"position:fixed;display:none;pointer-events:none;"
        "border:1px solid #abb2bf;background:#000;z-index:9999;max-width:240px;\">"
        "<script>(function(){"
        f"const THUMBS={json.dumps(image_thumbs)};"
        f"const TARGET_IDS={json.dumps(target_ids)};"
        "const tip=document.getElementById('hover-thumb');"
        "function show(evt,idx){"
        "if(typeof idx!=='number'||idx<0||idx>=THUMBS.length)return;"
        "tip.src=THUMBS[idx];"
        "tip.style.display='block';"
        "const e=evt.event;"
        "const cx=(e&&e.clientX!=null)?e.clientX:0;"
        "const cy=(e&&e.clientY!=null)?e.clientY:0;"
        "tip.style.left=(cx+14)+'px';"
        "tip.style.top=(cy+14)+'px';"
        "}"
        "function hide(){tip.style.display='none';}"
        "TARGET_IDS.forEach(function(id){"
        "const div=document.getElementById(id);"
        "if(!div||!div.on){console.warn('hover-thumb: no .on on',id);return;}"
        "div.on('plotly_hover',function(evt){"
        "const p=evt.points[0];"
        "console.debug('hover',id,'cd=',p.customdata,'type=',p.data&&p.data.type);"
        "show(evt,p.customdata);"
        "});"
        "div.on('plotly_unhover',hide);"
        "});"
        "})();</script>"
    )
