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


ALL_TOOLS = [
    check_pipeline_status,
    run_ingestion,
    run_representations,
    run_decompositions,
    run_reports,
]
