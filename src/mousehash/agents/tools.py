from __future__ import annotations

from smolagents import tool

from mousehash.schema.queries import (
    DEFAULT_DECOMP_SPECS,
    DEFAULT_RULE_ID,
    DEFAULT_SPEC_ID,
    format_status,
    next_pipeline_step,
    pipeline_status,
)


@tool
def check_pipeline_status(scene_set_id: str) -> str:
    """Check which pipeline stages have been completed for a given scene set.

    Returns a human-readable status string showing whether ingestion,
    representation extraction, decompositions, and HTML reports are done.

    Args:
        scene_set_id: The scene set identifier, e.g. 'allen_natural_scenes_v1'.
    """
    status = pipeline_status(scene_set_id)
    step = next_pipeline_step(status)
    text = format_status(status)
    text += f"\n\nNext step: {step if step else 'all done'}"
    return text


@tool
def run_ingestion(manifest_path: str, scene_set_id: str) -> str:
    """Fetch Allen natural scenes from AllenSDK and register them in DataJoint.

    Downloads the stimulus template, saves each image as a PNG, computes SHA1
    hashes, and inserts records into AllenNaturalSceneSet and
    AllenNaturalSceneImage.  Safe to re-run (idempotent).

    Args:
        manifest_path: Absolute path to the BrainObservatoryCache manifest JSON.
        scene_set_id:  Identifier for this scene set, e.g. 'allen_natural_scenes_v1'.
    """
    from pathlib import Path
    from mousehash.tools.allen.natural_scenes import ingest_natural_scenes

    summary = ingest_natural_scenes(
        manifest_path=Path(manifest_path),
        scene_set_id=scene_set_id,
    )
    return (
        f"Ingestion complete. scene_set_id={summary['scene_set_id']}, "
        f"n_images={summary['n_images']}, image_dir={summary['image_dir']}"
    )


@tool
def run_representations(
    scene_set_id: str,
    representation_spec_id: str = DEFAULT_SPEC_ID,
    rule_id: str = DEFAULT_RULE_ID,
) -> str:
    """Run the ViT ImageNet classifier on an ingested scene set.

    Loads images from DataJoint, runs the ViT model in batches, saves logits,
    probabilities, top-1 class indices, and animate/inanimate labels to disk,
    and registers the result in StimulusRepresentation.  Idempotent.

    Args:
        scene_set_id:           Scene set to process.
        representation_spec_id: Which RepresentationSpec to use.
        rule_id:                Which AnimateInanimateRule to apply.
    """
    from mousehash.tools.representations.vit_imagenet import compute_stimulus_representation

    summary = compute_stimulus_representation(
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        rule_id=rule_id,
    )
    return (
        f"Representations done. n_images={summary['n_images']}, "
        f"n_animate={summary['n_animate']}, n_inanimate={summary['n_inanimate']}"
    )


@tool
def run_decompositions(
    scene_set_id: str,
    representation_spec_id: str = DEFAULT_SPEC_ID,
    rule_id: str = DEFAULT_RULE_ID,
) -> str:
    """Run PCA (on logits) and NMF (on probabilities) for a scene set.

    Uses the decomposition specs seeded by seed_lookup_tables.py.  Both
    decompositions are run and registered in StimulusDecomposition.  Idempotent.

    Args:
        scene_set_id:           Scene set to decompose.
        representation_spec_id: Representation to decompose.
        rule_id:                Animate/inanimate rule used for the representation.
    """
    from mousehash.tools.decompositions.compute import compute_stimulus_decomposition

    results = []
    for decomp_spec_id in DEFAULT_DECOMP_SPECS:
        summary = compute_stimulus_decomposition(
            scene_set_id=scene_set_id,
            representation_spec_id=representation_spec_id,
            rule_id=rule_id,
            decomposition_spec_id=decomp_spec_id,
        )
        method = summary["method"]
        extra = (
            f"var={summary['explained_variance_ratio_total']:.1%}"
            if method == "pca"
            else f"err={summary['reconstruction_err']:.4f}"
        )
        results.append(f"{decomp_spec_id}: {extra}")

    return "Decompositions done. " + " | ".join(results)


@tool
def run_reports(
    scene_set_id: str,
    representation_spec_id: str = DEFAULT_SPEC_ID,
    rule_id: str = DEFAULT_RULE_ID,
) -> str:
    """Generate PCA and NMF interactive HTML explorer reports.

    Reads stored decomposition arrays and produces self-contained HTML files
    registered in StimulusDecompositionReport.  Idempotent.

    Args:
        scene_set_id:           Scene set whose reports to build.
        representation_spec_id: Representation the decompositions are based on.
        rule_id:                Animate/inanimate rule used.
    """
    from mousehash.tools.reports.build import build_decomposition_report

    paths = []
    for decomp_spec_id in DEFAULT_DECOMP_SPECS:
        summary = build_decomposition_report(
            scene_set_id=scene_set_id,
            representation_spec_id=representation_spec_id,
            rule_id=rule_id,
            decomposition_spec_id=decomp_spec_id,
        )
        paths.append(summary["report_path"])

    return "Reports built:\n" + "\n".join(f"  {p}" for p in paths)


@tool
def get_analysis_summary(
    scene_set_id: str,
    representation_spec_id: str = DEFAULT_SPEC_ID,
) -> str:
    """Read stored analysis results and return key statistics about the data.

    Loads the representation summary (animate/inanimate counts, top-1 class
    distribution) and both decomposition summaries (PCA explained variance,
    NMF reconstruction error) from disk.  Does not require a DataJoint
    connection — reads the JSON files saved during computation.

    Args:
        scene_set_id:           Scene set to summarise.
        representation_spec_id: Which representation to read summaries for.
    """
    import json
    from pathlib import Path
    from mousehash.artifacts.paths import decompositions_root, representations_root

    lines: list[str] = []

    # Representation summary
    rep_summary_path = (
        representations_root() / scene_set_id / representation_spec_id / "summary.json"
    )
    if rep_summary_path.exists():
        rep = json.loads(rep_summary_path.read_text())
        lines += [
            f"Representation ({representation_spec_id}):",
            f"  n_images    = {rep.get('n_images')}",
            f"  n_animate   = {rep.get('n_animate')}",
            f"  n_inanimate = {rep.get('n_inanimate')}",
            f"  top-1 class distribution (class_idx: count):",
        ]
        dist = rep.get("top1_distribution", {})
        for cls, cnt in sorted(dist.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    class {cls}: {cnt} images")
    else:
        lines.append("Representation summary not found — run compute_representations first.")

    lines.append("")

    # Decomposition summaries
    for decomp_spec_id in DEFAULT_DECOMP_SPECS:
        decomp_summary_path = (
            decompositions_root()
            / scene_set_id
            / representation_spec_id
            / decomp_spec_id
            / "summary.json"
        )
        if decomp_summary_path.exists():
            d = json.loads(decomp_summary_path.read_text())
            method = d.get("method", "?")
            lines.append(f"Decomposition ({decomp_spec_id}):")
            lines.append(f"  method      = {method}")
            lines.append(f"  n_components= {d.get('n_components')}")
            if "explained_variance_ratio_total" in d:
                lines.append(f"  total var explained = {d['explained_variance_ratio_total']:.1%}")
            if "reconstruction_err" in d:
                lines.append(f"  reconstruction err  = {d['reconstruction_err']:.4f}")
        else:
            lines.append(f"Decomposition summary not found for {decomp_spec_id}.")

    return "\n".join(lines)


ALL_TOOLS = [
    check_pipeline_status,
    get_analysis_summary,
    run_ingestion,
    run_representations,
    run_decompositions,
    run_reports,
]
