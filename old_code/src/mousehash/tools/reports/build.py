from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _labels_for_feature_space(feature_space: str) -> list[str] | None:
    """Return per-class labels for known feature spaces, or None to fall back to indices."""
    if feature_space == "imagenet_classifier_output":
        from mousehash.utils.imagenet import load_imagenet_labels
        return load_imagenet_labels()
    return None


def _encode_thumbnails(image_paths: list[Path]) -> list[str]:
    """Read each image and return a list of self-contained base64 data URIs."""
    import base64
    import mimetypes

    uris: list[str] = []
    for p in image_paths:
        mime, _ = mimetypes.guess_type(p.name)
        mime = mime or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        uris.append(f"data:{mime};base64,{b64}")
    return uris


def build_decomposition_report(
    scene_set_id: str,
    representation_spec_id: str,
    rule_id: str,
    decomposition_spec_id: str,
    force: bool = False,
) -> dict:
    """Generate an HTML explorer for a stored decomposition and register it in DataJoint.

    Dispatches to :func:`build_pca_report` or :func:`build_nmf_report` based on
    the ``method`` stored in ``DecompositionSpec``.

    Returns:
        Summary dict with ``report_path`` and ``report_type``.
    """
    from mousehash.artifacts.io import load_json, load_npy
    from mousehash.artifacts.paths import reports_root
    from mousehash.schema.decompositions import DecompositionSpec, StimulusDecomposition
    from mousehash.schema.representations import (
        AnimateInanimateRule,
        RepresentationSpec,
        StimulusRepresentation,
    )
    from mousehash.schema.stimuli import StimulusImage
    from mousehash.schema.reports import StimulusDecompositionReport

    rep_key = dict(
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        rule_id=rule_id,
    )
    decomp_key = dict(**rep_key, decomposition_spec_id=decomposition_spec_id)

    spec = (DecompositionSpec & {"decomposition_spec_id": decomposition_spec_id}).fetch1()
    method = spec["method"]
    report_type = f"{method}_explorer"

    report_key = dict(**decomp_key, report_type=report_type)
    if (StimulusDecompositionReport & report_key) and not force:
        logger.info("Report already exists for %s, skipping.", decomposition_spec_id)
        return (StimulusDecompositionReport & report_key).fetch1()

    decomp = (StimulusDecomposition & decomp_key).fetch1()
    rep = (StimulusRepresentation & rep_key).fetch1()
    rep_spec = (RepresentationSpec & {"representation_spec_id": representation_spec_id}).fetch1()
    class_labels = _labels_for_feature_space(rep_spec["feature_space"])

    scores = load_npy(Path(decomp["scores_path"]))
    components = load_npy(Path(decomp["components_path"]))
    animate_inanimate = load_npy(Path(rep["animate_inanimate_path"]))
    component_stats = load_json(Path(decomp["component_stats_path"]))

    image_rows = (StimulusImage & {"scene_set_id": scene_set_id}).fetch(
        "image_idx", "image_path", as_dict=True, order_by="image_idx"
    )
    image_paths = [Path(r["image_path"]) for r in image_rows]
    image_thumbs = _encode_thumbnails(image_paths)

    out_dir = (
        reports_root()
        / scene_set_id
        / representation_spec_id
        / decomposition_spec_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{method}_explorer.html"
    summary_path = out_dir / f"{method}_report_summary.json"

    if method == "pca":
        import numpy as np
        from mousehash.tools.reports.pca_html import build_pca_report
        build_pca_report(
            scores=scores,
            components=components,
            explained_variance_ratio=np.array(component_stats["explained_variance_ratio"]),
            animate_inanimate=animate_inanimate,
            image_paths=image_paths,
            output_path=report_path,
            title=f"PCA Explorer — {scene_set_id}",
            class_labels=class_labels,
            image_thumbs=image_thumbs,
        )
    elif method == "nmf":
        from mousehash.tools.reports.nmf_html import build_nmf_report
        build_nmf_report(
            scores=scores,
            components=components,
            animate_inanimate=animate_inanimate,
            image_paths=image_paths,
            output_path=report_path,
            reconstruction_err=component_stats.get("reconstruction_err"),
            title=f"NMF Explorer — {scene_set_id}",
            class_labels=class_labels,
            image_thumbs=image_thumbs,
        )
    else:
        raise ValueError(f"No report builder for method: {method!r}")

    from mousehash.artifacts.io import save_json
    summary = dict(
        scene_set_id=scene_set_id,
        decomposition_spec_id=decomposition_spec_id,
        report_type=report_type,
        report_path=str(report_path),
    )
    save_json(summary_path, summary)

    StimulusDecompositionReport.insert1(
        dict(
            **decomp_key,
            report_type=report_type,
            report_path=str(report_path),
            summary_path=str(summary_path),
        ),
        skip_duplicates=True,
    )
    logger.info("Registered report: %s", report_path)
    return summary
