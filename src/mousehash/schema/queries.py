from __future__ import annotations

DEFAULT_SPEC_ID = "vit_b16_imagenet_cpu"
DEFAULT_RULE_ID = "imagenet_top1_leq_397"
DEFAULT_DECOMP_SPECS = [
    "pca_logits_10_exploratory",
    "nmf_probs_10_exploratory",
]


def pipeline_status(
    scene_set_id: str,
    representation_spec_id: str = DEFAULT_SPEC_ID,
    rule_id: str = DEFAULT_RULE_ID,
    decomp_spec_ids: list[str] = DEFAULT_DECOMP_SPECS,
) -> dict:
    """Return a structured snapshot of what pipeline stages exist in DataJoint.

    All DataJoint imports are deferred so this module can be imported without
    a live database connection (the query only runs when called).

    Returns a dict with keys:
        ``scene_set_ingested``   – bool
        ``n_images``             – int or None
        ``representations_done`` – bool
        ``decompositions``       – dict[spec_id -> bool]
        ``reports``              – dict[spec_id -> bool]
    """
    from mousehash.schema.decompositions import StimulusDecomposition
    from mousehash.schema.reports import StimulusDecompositionReport
    from mousehash.schema.representations import StimulusRepresentation
    from mousehash.schema.stimuli import StimulusImage, StimulusSet

    scene_set_key = {"scene_set_id": scene_set_id}
    rep_key = dict(
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        rule_id=rule_id,
    )

    ingested = bool(StimulusSet & scene_set_key)
    n_images = None
    if ingested:
        n_images = len(StimulusImage & scene_set_key)

    reps_done = bool(StimulusRepresentation & rep_key)

    decomps = {}
    reports = {}
    for spec_id in decomp_spec_ids:
        decomp_key = dict(**rep_key, decomposition_spec_id=spec_id)
        decomps[spec_id] = bool(StimulusDecomposition & decomp_key)

        # report_type mirrors the method name
        method = spec_id.split("_")[0]  # "pca" or "nmf"
        report_key = dict(**decomp_key, report_type=f"{method}_explorer")
        reports[spec_id] = bool(StimulusDecompositionReport & report_key)

    return dict(
        scene_set_id=scene_set_id,
        scene_set_ingested=ingested,
        n_images=n_images,
        representations_done=reps_done,
        decompositions=decomps,
        reports=reports,
    )


def format_status(status: dict) -> str:
    """Format a pipeline_status dict as a human-readable string."""
    lines = [
        f"Pipeline status for scene_set_id='{status['scene_set_id']}'",
        f"  Ingestion        : {'✓' if status['scene_set_ingested'] else '✗'}"
        + (f"  ({status['n_images']} images)" if status["n_images"] else ""),
        f"  Representations  : {'✓' if status['representations_done'] else '✗'}",
    ]
    for spec_id, done in status["decompositions"].items():
        lines.append(f"  Decomposition [{spec_id}] : {'✓' if done else '✗'}")
    for spec_id, done in status["reports"].items():
        lines.append(f"  Report        [{spec_id}] : {'✓' if done else '✗'}")
    return "\n".join(lines)


def next_pipeline_step(status: dict) -> str | None:
    """Return the name of the next incomplete pipeline step, or None if done."""
    if not status["scene_set_ingested"]:
        return "ingest"
    if not status["representations_done"]:
        return "representations"
    if not all(status["decompositions"].values()):
        return "decompositions"
    if not all(status["reports"].values()):
        return "reports"
    return None
