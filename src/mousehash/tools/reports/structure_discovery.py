"""Structure-discovery HTML reports for PCA / NMF decompositions.

Each report is a self-contained HTML file (Plotly inlined once, image
thumbnails inlined as base64). The combined `generate_structure_discovery_report`
wraps both PCA and NMF sections together with a manifest summary header so
the bundle is auditable on its own.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

import numpy as np

from mousehash.artifacts.io import load_json, load_npy, save_html, save_json
from mousehash.artifacts.paths import reports_root
from mousehash.core.manifests import RoleManifest
from mousehash.transformations.labeling import load_imagenet_labels

logger = logging.getLogger(__name__)


def _encode_thumbnails(image_paths: list[Path]) -> list[str]:
    """Read each image and return a list of self-contained base64 data URIs."""
    uris: list[str] = []
    for p in image_paths:
        mime, _ = mimetypes.guess_type(p.name)
        mime = mime or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        uris.append(f"data:{mime};base64,{b64}")
    return uris


def _hover_thumb_html(
    image_thumbs: list[str],
    target_ids: list[str],
    heatmap_thumb_threshold: float | None = None,
    heatmap_component_indices: list[list[int]] | None = None,
    heatmap_thumb_max_images: int | None = None,
) -> str:
    """Hover-tooltip script: single-image thumbs on scatter, thumb-grid on heatmap cells."""
    return (
        "<img id=\"hover-thumb\" alt=\"\" "
        "style=\"position:fixed;display:none;pointer-events:none;"
        "border:1px solid #abb2bf;background:#000;z-index:9999;max-width:240px;\">"
        "<div id=\"hover-thumb-grid\" "
        "style=\"position:fixed;display:none;pointer-events:none;padding:10px;"
        "border:1px solid #abb2bf;background:rgba(0,0,0,0.92);z-index:9999;"
        "max-width:360px;max-height:260px;overflow:auto;box-shadow:0 10px 24px rgba(0,0,0,0.35);\">"
        "<div id=\"hover-thumb-grid-title\" style=\"margin:0 0 8px 0;font:600 12px sans-serif;color:#abb2bf;\"></div>"
        "<div id=\"hover-thumb-grid-body\" "
        "style=\"display:grid;grid-template-columns:repeat(4,minmax(0,72px));gap:6px;\"></div>"
        "</div>"
        "<script>(function(){"
        f"const THUMBS={json.dumps(image_thumbs)};"
        f"const TARGET_IDS={json.dumps(target_ids)};"
        f"const HEATMAP_THUMB_THRESHOLD={json.dumps(heatmap_thumb_threshold)};"
        f"const HEATMAP_COMPONENT_INDICES={json.dumps(heatmap_component_indices)};"
        f"const HEATMAP_THUMB_MAX_IMAGES={json.dumps(heatmap_thumb_max_images)};"
        "const tip=document.getElementById('hover-thumb');"
        "const grid=document.getElementById('hover-thumb-grid');"
        "const gridTitle=document.getElementById('hover-thumb-grid-title');"
        "const gridBody=document.getElementById('hover-thumb-grid-body');"
        "function getHeatmapComponentIndex(point){"
        "if(typeof point.x==='string'){const m=point.x.match(/^C(\\d+)$/);if(m){return Number(m[1]);}}"
        "if(Array.isArray(point.pointNumber)&&typeof point.pointNumber[1]==='number'){return point.pointNumber[1];}"
        "return null;"
        "}"
        "function position(el,evt){"
        "const e=evt.event;"
        "const cx=(e&&e.clientX!=null)?e.clientX:0;"
        "const cy=(e&&e.clientY!=null)?e.clientY:0;"
        "el.style.left=(cx+14)+'px';"
        "el.style.top=(cy+14)+'px';"
        "}"
        "function show(evt,idx){"
        "if(typeof idx!=='number'||!Number.isFinite(idx)||idx<0||idx>=THUMBS.length){tip.style.display='none';return;}"
        "grid.style.display='none';"
        "tip.src=THUMBS[idx];"
        "tip.style.display='block';"
        "position(tip,evt);"
        "}"
        "function showGrid(evt,compIdx,indices){"
        "tip.style.display='none';"
        "if(!Array.isArray(indices)||indices.length===0){grid.style.display='none';return;}"
        "gridTitle.textContent='Component C'+compIdx+' thumbnails (score > '+HEATMAP_THUMB_THRESHOLD+', max '+HEATMAP_THUMB_MAX_IMAGES+')';"
        "gridBody.innerHTML=indices.map(function(idx){"
        "if(typeof idx!=='number'||!Number.isFinite(idx)||idx<0||idx>=THUMBS.length){return '';}"
        "return '<img src=\"'+THUMBS[idx]+'\" alt=\"img '+idx+'\" title=\"img '+idx+'\" style=\"display:block;width:72px;height:auto;border:1px solid #4b5263;background:#111;\">';"
        "}).join('');"
        "if(!gridBody.innerHTML){grid.style.display='none';return;}"
        "grid.style.display='block';"
        "position(grid,evt);"
        "}"
        "function hide(){tip.style.display='none';grid.style.display='none';}"
        "TARGET_IDS.forEach(function(id){"
        "const div=document.getElementById(id);"
        "if(!div||!div.on){console.warn('hover-thumb: no .on on',id);return;}"
        "div.on('plotly_hover',function(evt){"
        "const p=evt.points[0];"
        "if(p.data&&p.data.type==='heatmap'&&Array.isArray(HEATMAP_COMPONENT_INDICES)){"
        "showGrid(evt,getHeatmapComponentIndex(p),HEATMAP_COMPONENT_INDICES[getHeatmapComponentIndex(p)]);"
        "return;"
        "}"
        "show(evt,p.customdata);"
        "});"
        "div.on('plotly_unhover',hide);"
        "});"
        "})();</script>"
    )


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Reports require plotly. Install with: pip install -e '.[viz]'") from exc
    return go


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
) -> Path:
    """Self-contained interactive HTML PCA report.

    Contents: scree plot, scatter colored by animate/inanimate with X/Y PC
    dropdowns, per-PC top-10 positive / negative class loadings.
    """
    go = _require_plotly()

    n_components = scores.shape[1]
    labels = ["animate" if a else "inanimate" for a in animate_inanimate]
    color_map = {"animate": "#e06c75", "inanimate": "#61afef"}
    hover = [f"img {i}<br>{Path(p).name}" for i, p in enumerate(image_paths)]

    figs: list[Any] = []

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
            dict(buttons=x_buttons, direction="down", x=0.13, xanchor="left",
                 y=1.16, yanchor="top", active=init_x, showactive=True),
            dict(buttons=y_buttons, direction="down", x=0.33, xanchor="left",
                 y=1.16, yanchor="top", active=init_y, showactive=True),
        ],
        annotations=[
            dict(text="X PC:", x=0.07, xref="paper", y=1.14, yref="paper",
                 showarrow=False, xanchor="right"),
            dict(text="Y PC:", x=0.27, xref="paper", y=1.14, yref="paper",
                 showarrow=False, xanchor="right"),
        ],
    )
    figs.append(fig_scatter)

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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_parts = [
        f"<html><head><title>{title}</title></head>",
        "<body style='background:#282c34;color:#abb2bf;font-family:sans-serif;'>",
        f"<h1 style='padding:16px'>{title}</h1>",
    ]
    scatter_idx = 1
    div_ids = {scatter_idx: "pca-scatter"} if image_thumbs is not None else {}
    for i, fig in enumerate(figs):
        include_js = i == 0
        html_parts.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs=include_js,
                **({"div_id": div_ids[i]} if i in div_ids else {}),
            )
        )
    if image_thumbs is not None:
        html_parts.append(_hover_thumb_html(image_thumbs, ["pca-scatter"]))
    html_parts.append("</body></html>")
    save_html(output_path, "\n".join(html_parts))
    logger.info("Wrote PCA report to %s", output_path)
    return output_path


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
    heatmap_thumb_threshold: float = 0.01,
) -> Path:
    """Self-contained interactive HTML NMF report.

    Contents: scatter with X/Y component dropdowns, image × component
    activation heatmap, top-15 class loadings per component.
    """
    go = _require_plotly()

    heatmap_thumb_max_images = 8
    n_components = scores.shape[1]
    labels = ["animate" if a else "inanimate" for a in animate_inanimate]
    color_map = {"animate": "#e06c75", "inanimate": "#61afef"}
    hover = [f"img {i}<br>{Path(p).name}" for i, p in enumerate(image_paths)]

    figs: list[Any] = []

    labels_arr = np.array(labels)
    masks = {label: labels_arr == label for label in color_map}
    hovers = {label: [hover[i] for i in np.where(masks[label])[0]] for label in color_map}
    indices = {label: np.where(masks[label])[0].tolist() for label in color_map}
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

    trace_order = list(color_map.keys())
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
            dict(buttons=x_buttons, direction="down", x=0.13, xanchor="left",
                 y=1.16, yanchor="top", active=init_x, showactive=True),
            dict(buttons=y_buttons, direction="down", x=0.33, xanchor="left",
                 y=1.16, yanchor="top", active=init_y, showactive=True),
        ],
        annotations=[
            dict(text="X comp:", x=0.07, xref="paper", y=1.14, yref="paper",
                 showarrow=False, xanchor="right"),
            dict(text="Y comp:", x=0.27, xref="paper", y=1.14, yref="paper",
                 showarrow=False, xanchor="right"),
        ],
    )
    figs.append(fig_scatter)

    n_images = scores.shape[0]
    heat_component_indices = []
    for comp_idx in range(n_components):
        active_idx = np.flatnonzero(scores[:, comp_idx] > heatmap_thumb_threshold)
        ranked_idx = active_idx[np.argsort(scores[active_idx, comp_idx])[::-1]]
        heat_component_indices.append(ranked_idx[:heatmap_thumb_max_images].tolist())
    fig_heat = go.Figure(
        go.Heatmap(
            z=scores.tolist(),
            x=[f"C{i}" for i in range(n_components)],
            y=[f"img {i}" for i in range(n_images)],
            colorscale="Viridis",
            hovertemplate="img=%{y}, comp=%{x}<br>score=%{z:.3f}<extra></extra>",
        )
    )
    fig_heat.update_layout(
        title=(
            "Image × component activation heatmap"
            f" (thumbnail tile threshold > {heatmap_thumb_threshold:g}, max {heatmap_thumb_max_images})"
        ),
        xaxis_title="NMF component",
        yaxis_title="Image index",
        template="plotly_dark",
    )
    figs.append(fig_heat)

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

    output_path = Path(output_path)
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
        html_parts.append(
            _hover_thumb_html(
                image_thumbs,
                ["nmf-scatter", "nmf-heatmap"],
                heatmap_thumb_threshold=heatmap_thumb_threshold,
                heatmap_component_indices=heat_component_indices,
                heatmap_thumb_max_images=heatmap_thumb_max_images,
            )
        )
    html_parts.append("</body></html>")
    save_html(output_path, "\n".join(html_parts))
    logger.info("Wrote NMF report to %s", output_path)
    return output_path


def generate_structure_discovery_report(
    manifest: RoleManifest,
    pca_summary: dict[str, Any],
    nmf_summary: dict[str, Any],
    image_catalog: list[dict[str, Any]],
    animate_inanimate: np.ndarray,
    output_dir: Path | None = None,
    title_prefix: str | None = None,
) -> dict[str, Any]:
    """Render PCA + NMF reports plus a small index page summarizing the manifest.

    Args:
        manifest: RoleManifest the analyses ran against.
        pca_summary / nmf_summary: dicts returned by `tools.factor_models.run_pca` and `run_nmf`.
        image_catalog: per-image rows from the manifest's stimulus catalog.
        animate_inanimate: (n_images,) int8 labels.
        output_dir: where to write. Defaults to <reports_root>/<manifest_id>/.
        title_prefix: optional human title prefix (e.g. dataset label).

    Returns:
        dict with paths to pca_report.html, nmf_report.html, index.html.
    """
    if output_dir is None:
        output_dir = reports_root() / manifest.manifest_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [Path(row["image_path"]) for row in image_catalog]
    image_thumbs = _encode_thumbnails(image_paths)
    class_labels = load_imagenet_labels()

    pca_scores = load_npy(Path(pca_summary["artifacts"]["scores"]))
    pca_components = load_npy(Path(pca_summary["artifacts"]["components"]))
    pca_stats = load_json(Path(pca_summary["artifacts"]["component_stats"]))
    pca_evr = np.array(pca_stats["explained_variance_ratio"])

    title = title_prefix or manifest.dataset.label or manifest.manifest_id

    pca_path = build_pca_report(
        scores=pca_scores,
        components=pca_components,
        explained_variance_ratio=pca_evr,
        animate_inanimate=animate_inanimate,
        image_paths=image_paths,
        output_path=output_dir / "pca_report.html",
        title=f"PCA Explorer — {title}",
        class_labels=class_labels,
        image_thumbs=image_thumbs,
    )

    nmf_scores = load_npy(Path(nmf_summary["artifacts"]["scores"]))
    nmf_components = load_npy(Path(nmf_summary["artifacts"]["components"]))
    nmf_path = build_nmf_report(
        scores=nmf_scores,
        components=nmf_components,
        animate_inanimate=animate_inanimate,
        image_paths=image_paths,
        output_path=output_dir / "nmf_report.html",
        reconstruction_err=nmf_summary.get("reconstruction_err"),
        title=f"NMF Explorer — {title}",
        class_labels=class_labels,
        image_thumbs=image_thumbs,
    )

    index_html = _render_index_page(title=title, manifest=manifest,
                                    pca_summary=pca_summary, nmf_summary=nmf_summary)
    index_path = save_html(output_dir / "index.html", index_html)

    bundle_summary = {
        "manifest_id": manifest.manifest_id,
        "output_dir": str(output_dir),
        "reports": {
            "index": str(index_path),
            "pca": str(pca_path),
            "nmf": str(nmf_path),
        },
        "pca_summary": pca_summary,
        "nmf_summary": nmf_summary,
    }
    save_json(output_dir / "report_bundle.json", bundle_summary)
    return bundle_summary


def _render_index_page(
    title: str,
    manifest: RoleManifest,
    pca_summary: dict[str, Any],
    nmf_summary: dict[str, Any],
) -> str:
    role_rows = []
    for name in ["conditions", "stimuli", "behavior", "neural_data", "time_organization", "metadata"]:
        role = manifest.roles.get(name)
        role_rows.append(
            f"<tr><td>{name}</td><td>{role.status.value}</td><td>{role.confidence.value}</td>"
            f"<td>{len(role.evidence)} evidence item(s)</td></tr>"
        )
    return (
        f"<html><head><title>Structure discovery — {title}</title></head>"
        "<body style='background:#282c34;color:#abb2bf;font-family:sans-serif;padding:24px;'>"
        f"<h1>Structure discovery report — {title}</h1>"
        f"<p><b>manifest_id:</b> {manifest.manifest_id}<br>"
        f"<b>dataset:</b> {manifest.dataset.target}/{manifest.dataset.dataset_id} "
        f"({manifest.dataset.dataset_version})</p>"
        "<h2>Roles</h2>"
        "<table style='border-collapse:collapse;color:#abb2bf;'>"
        "<tr><th style='text-align:left;padding-right:16px'>role</th>"
        "<th style='text-align:left;padding-right:16px'>status</th>"
        "<th style='text-align:left;padding-right:16px'>confidence</th>"
        "<th style='text-align:left'>evidence</th></tr>"
        + "\n".join(role_rows) +
        "</table>"
        "<h2>Decompositions</h2>"
        f"<p>PCA: {pca_summary['n_components']} components, "
        f"total explained variance = {pca_summary['explained_variance_ratio_total']:.3f}<br>"
        f"NMF: {nmf_summary['n_components']} components, "
        f"reconstruction err = {nmf_summary['reconstruction_err']:.4f}, "
        f"T = {nmf_summary['temperature']}</p>"
        "<h2>Reports</h2>"
        "<ul>"
        "<li><a style='color:#61afef' href='pca_report.html'>PCA explorer</a></li>"
        "<li><a style='color:#61afef' href='nmf_report.html'>NMF explorer</a></li>"
        "</ul>"
        "</body></html>"
    )
