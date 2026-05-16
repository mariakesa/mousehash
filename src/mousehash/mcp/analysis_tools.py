"""MCP wrappers for analysis tools: PCA + NMF.

These take a source `view_id` (e.g. the view returned by `extract_vit_features`),
resolve it via `find_view_by_id`, and call the cached `run_pca` / `run_nmf`
factor-model implementations. Both return the output view id + summary.
"""

from __future__ import annotations

from typing import Any

from mousehash.mcp.errors import mcp_safe
from mousehash.mcp.views import find_view_by_id
from mousehash.tools.comparison.group_comparison import compare_jpeg_animate_inanimate_views
from mousehash.tools.factor_models.nmf import run_nmf as _run_nmf
from mousehash.tools.factor_models.pca import run_pca as _run_pca


@mcp_safe
def run_pca(
    view_id: str,
    n_components: int = 10,
    normalize_input: bool = True,
    input_array_name: str = "logits",
    label: str = "",
) -> dict[str, Any]:
    """Run PCA on an OBSERVATION_BY_FEATURE view; return the output view + summary.

    Idempotent: same `(view_id, n_components, normalize_input, input_array_name)`
    -> cache hit.

    Args:
        view_id: source view id (e.g. from extract_vit_features).
        n_components: number of principal components to retain.
        normalize_input: z-score features before fitting (recommended for ViT logits).
        input_array_name: which .npy file under the source view's artifact_path to
            load. ViT views expose "logits" and "probabilities". Defaults to "logits".
        label: informational tag; does NOT affect cache key.

    Returns:
        {view_id, artifact_path, summary, from_cache}. The output view's kind is
        OBSERVATION_BY_FEATURE with features = pca_components.
    """
    source_view = find_view_by_id(view_id)
    out_view, summary = _run_pca(
        view=source_view,
        n_components=int(n_components),
        normalize_input=bool(normalize_input),
        input_array_name=input_array_name,
        label=label or None,
    )
    return {
        "view_id": out_view.view_id,
        "artifact_path": out_view.artifact_path,
        "summary": summary,
        "from_cache": summary.get("from_cache", False),
    }


@mcp_safe
def run_nmf(
    view_id: str,
    n_components: int = 10,
    temperature: float = 1.0,
    input_array_name: str = "probabilities",
    label: str = "",
) -> dict[str, Any]:
    """Run NMF on a non-negative OBSERVATION_BY_FEATURE view; return output view + summary.

    Idempotent: same `(view_id, n_components, temperature, ...)` -> cache hit.

    Args:
        view_id: source view id (e.g. from extract_vit_features).
        n_components: number of NMF components to retain.
        temperature: PIL softmax temperature (1.0 = unchanged; >1 smooths, <1 sharpens).
        input_array_name: which .npy under source view's artifact_path. NMF needs
            non-negative input; "probabilities" is the canonical choice for ViT views.
        label: informational tag; does NOT affect cache key.

    Returns:
        {view_id, artifact_path, summary, from_cache}. Output view kind is
        OBSERVATION_BY_FEATURE with features = nmf_components.
    """
    source_view = find_view_by_id(view_id)
    out_view, summary = _run_nmf(
        view=source_view,
        n_components=int(n_components),
        temperature=float(temperature),
        input_array_name=input_array_name,
        label=label or None,
    )
    return {
        "view_id": out_view.view_id,
        "artifact_path": out_view.artifact_path,
        "summary": summary,
        "from_cache": summary.get("from_cache", False),
    }


@mcp_safe
def compare_jpeg_by_animate_inanimate(
    jpeg_view_id: str,
    vit_view_id: str,
    plot: bool = True,
) -> dict[str, Any]:
    """Statistically compare JPEG byte sizes between animate and inanimate images.

    Per-quality Welch's t + Mann-Whitney U + Cohen's d, plus Bonferroni-corrected
    min-p across qualities and a plain-English `summary` string. Optionally
    renders an interactive HTML grouped boxplot.

    Idempotent via `cached_computation`: same `(jpeg_view_id, vit_view_id, plot)`
    -> cache hit.

    Args:
        jpeg_view_id: view returned by `extract_jpeg_sizes`.
        vit_view_id: view returned by `extract_vit_features` (supplies
            `animate_inanimate.npy`).
        plot: render an interactive HTML grouped boxplot under the cache dir.

    Returns:
        Comparison result dict: `per_feature` stats (one entry per quality),
        `overall` summary numbers, plain-English `summary`, `input` provenance,
        `artifacts.plot_html` (if `plot=True`), `view_id`, `artifact_path`, `from_cache`.
    """
    jpeg_view = find_view_by_id(jpeg_view_id)
    vit_view = find_view_by_id(vit_view_id)
    return compare_jpeg_animate_inanimate_views(
        jpeg_view=jpeg_view, vit_view=vit_view, plot=bool(plot),
    )
