from __future__ import annotations

from mousehash.schema.queries import (
    DEFAULT_DECOMP_SPECS,
    DEFAULT_RULE_ID,
    DEFAULT_SPEC_ID,
    format_status,
    next_pipeline_step,
    pipeline_status,
)


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


def plot_cell_natural_scenes_dff(
    manifest_path: str | None,
    cell_specimen_id: int,
    model_name: str = "google/vit-base-patch16-224",
    batch_size: int = 16,
    device: str = "cpu",
    threshold_max_class_idx: int = 397,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
    animate_color: str = "#1f77b4",
    inanimate_color: str = "#ff7f0e",
    context_color: str = "#b0b7c3",
    highlight_mode: str = "both",
) -> str:
    """Plot one cell's dF/F trace with animate/inanimate natural-scene overlays.

    Looks up a natural-scenes ophys experiment for the given cell, loads its
    dF/F trace, classifies the Allen natural-scene images on the
    animate/inanimate axis, aligns those labels to stimulus presentation
    intervals, and writes an interactive HTML plot.

    Args:
        manifest_path: Absolute path to the Allen BrainObservatoryCache manifest.
            If omitted, uses the configured ALLEN_DATA path.
        cell_specimen_id: Allen cell specimen identifier to fetch.
        model_name: HuggingFace image-classification model used for scene labels.
        batch_size: Batch size forwarded to the scene classifier helper.
        device: Torch device for scene classification, e.g. cpu or cuda.
        threshold_max_class_idx: Top-1 ImageNet index cutoff for animate vs inanimate.
        time_start_s: Optional start time in seconds for clipping the plotted trace.
        time_end_s: Optional end time in seconds for clipping the plotted trace.
        animate_color: Color for animate timepoints.
        inanimate_color: Color for inanimate timepoints.
        context_color: Color for non-natural-scene/background timepoints.
        highlight_mode: Which semantic class to color: both, animate, or inanimate.
    """
    from pathlib import Path

    from mousehash.config import ALLEN_MANIFEST_PATH
    from mousehash.tools.allen.cell_activity import (
        analyze_cell_dff_against_animate_inanimate,
    )

    resolved_manifest_path = (
        ALLEN_MANIFEST_PATH if manifest_path is None or not str(manifest_path).strip() else Path(manifest_path)
    )

    summary = analyze_cell_dff_against_animate_inanimate(
        manifest_path=resolved_manifest_path,
        cell_specimen_id=int(cell_specimen_id),
        model_name=model_name,
        batch_size=int(batch_size),
        device=device,
        threshold_max_class_idx=int(threshold_max_class_idx),
        time_start_s=None if time_start_s is None else float(time_start_s),
        time_end_s=None if time_end_s is None else float(time_end_s),
        animate_color=animate_color,
        inanimate_color=inanimate_color,
        context_color=context_color,
        highlight_mode=highlight_mode,
    )
    return (
        f"Cell plot complete. cell_specimen_id={summary['cell_specimen_id']}, "
        f"experiment_id={summary['experiment_id']}, "
        f"time_start_s={summary['time_start_s']}, "
        f"time_end_s={summary['time_end_s']}, "
        f"highlight_mode={summary['highlight_mode']}, "
        f"animate_color={summary['animate_color']}, "
        f"inanimate_color={summary['inanimate_color']}, "
        f"animate_timepoints={summary['n_animate_timepoints']}, "
        f"inanimate_timepoints={summary['n_inanimate_timepoints']}, "
        f"plot={summary['plot_path']}, "
        f"plot_png={summary['plot_png_path']}"
    )


def plot_cell_natural_scenes_dff_vanilla(
    manifest_path: str | None,
    cell_specimen_id: int,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
    line_color: str = "#2f4858",
) -> str:
    """Plot one cell's dF/F trace without animate/inanimate overlay labels.

    Args:
        manifest_path: Absolute path to the Allen BrainObservatoryCache manifest.
            If omitted, uses the configured ALLEN_DATA path.
        cell_specimen_id: Allen cell specimen identifier to fetch.
        time_start_s: Optional start time in seconds for clipping the plotted trace.
        time_end_s: Optional end time in seconds for clipping the plotted trace.
        line_color: Color for the vanilla dF/F trace.
    """
    from pathlib import Path

    from mousehash.config import ALLEN_MANIFEST_PATH
    from mousehash.tools.allen.cell_activity import analyze_cell_dff_vanilla

    resolved_manifest_path = (
        ALLEN_MANIFEST_PATH if manifest_path is None or not str(manifest_path).strip() else Path(manifest_path)
    )

    summary = analyze_cell_dff_vanilla(
        manifest_path=resolved_manifest_path,
        cell_specimen_id=int(cell_specimen_id),
        time_start_s=None if time_start_s is None else float(time_start_s),
        time_end_s=None if time_end_s is None else float(time_end_s),
        line_color=line_color,
    )
    return (
        f"Vanilla cell plot complete. cell_specimen_id={summary['cell_specimen_id']}, "
        f"experiment_id={summary['experiment_id']}, "
        f"time_start_s={summary['time_start_s']}, "
        f"time_end_s={summary['time_end_s']}, "
        f"line_color={summary['line_color']}, "
        f"n_timepoints={summary['n_timepoints']}, "
        f"plot={summary['plot_path']}, "
        f"plot_png={summary['plot_png_path']}"
    )


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


def run_nmf_at_temperature(
    scene_set_id: str,
    temperature: float,
    n_components: int = 10,
    representation_spec_id: str = DEFAULT_SPEC_ID,
    rule_id: str = DEFAULT_RULE_ID,
) -> str:
    """Run NMF on the stored softmax probabilities at a chosen temperature.

    Use this to compare how concentration of the semantic distribution affects
    the recovered NMF themes. Temperature is applied to the stored
    probabilities as ``p_T proportional to p^(1/T)`` (mathematically equivalent
    to ``softmax(logits/T)``); ``T<1`` sharpens, ``T>1`` smooths, ``T=1`` is a
    no-op. The ViT does NOT need to be re-run — do not invent new
    ``representation_spec_id`` values for this; only the existing
    representation is used.

    Synthesises (or reuses) a ``DecompositionSpec`` with id
    ``nmf_probs_<n_components>_T<temperature>``, runs the decomposition,
    builds the HTML report, and returns the spec id, reconstruction error,
    and report path.

    Args:
        scene_set_id:           Scene set whose representation to decompose.
        temperature:            Temperature for the input probabilities. Must be > 0.
        n_components:           Number of NMF components (default 10).
        representation_spec_id: Existing RepresentationSpec id; leave at default.
        rule_id:                Existing AnimateInanimateRule id; leave at default.
    """
    from mousehash.schema.decompositions import DecompositionSpec
    from mousehash.tools.decompositions.compute import compute_stimulus_decomposition
    from mousehash.tools.reports.build import build_decomposition_report

    if temperature <= 0:
        return f"Error: temperature must be > 0, got {temperature}"

    temp_label = f"{temperature:g}".replace(".", "p")
    decomposition_spec_id = f"nmf_probs_{n_components}_T{temp_label}"

    DecompositionSpec.insert1(
        dict(
            decomposition_spec_id=decomposition_spec_id,
            method="nmf",
            input_kind="probabilities",
            n_components=n_components,
            normalize_input=False,
            mode="agent",
            nmf_temperature=temperature,
        ),
        skip_duplicates=True,
    )

    decomp_summary = compute_stimulus_decomposition(
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        rule_id=rule_id,
        decomposition_spec_id=decomposition_spec_id,
    )
    report_summary = build_decomposition_report(
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        rule_id=rule_id,
        decomposition_spec_id=decomposition_spec_id,
    )
    return (
        f"NMF at T={temperature} done. "
        f"spec={decomposition_spec_id}, "
        f"reconstruction_err={decomp_summary['reconstruction_err']:.4f}, "
        f"report={report_summary['report_path']}"
    )


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

    from mousehash.artifacts.paths import decompositions_root, representations_root

    lines: list[str] = []

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


__all__ = [
    "check_pipeline_status",
    "get_analysis_summary",
    "run_ingestion",
    "run_representations",
    "run_decompositions",
    "run_nmf_at_temperature",
    "run_reports",
    "blahml_list_tools",
    "blahml_start_dialogue",
    "blahml_submit_answer",
    "blahml_run",
]


# ---------------------------------------------------------------------------
# BlahML dialogue tools
# ---------------------------------------------------------------------------

# Per-process dialogue state. Each entry tracks one in-progress conversation:
# the manifest id being filled, the cumulative answers, and the question
# trace (one record per asked-and-answered slot). v0 keeps this in memory;
# durability lives on ToolRunSpec once the dialogue is finalized.
_DIALOGUES: dict[str, dict] = {}


def _format_question_payload(dialogue_id: str, pending) -> str:
    import json as _json

    p = pending.parameter
    payload = dict(
        dialogue_id=dialogue_id,
        parameter_name=p.name,
        section=pending.section,
        type=p.type,
        required=p.required,
        ask=p.question.ask,
        explain=p.question.explain.strip(),
    )
    if p.choices is not None:
        payload["choices"] = p.choices
    if p.default is not None:
        payload["default"] = p.default
    if p.question.default_explanation:
        payload["default_explanation"] = p.question.default_explanation.strip()
    if p.range is not None:
        payload["range"] = dict(min=p.range.min, max=p.range.max)
    return _json.dumps(payload, default=str)


def blahml_list_tools() -> str:
    """List the BlahML tools that have manifests installed.

    Returns a JSON string of objects with ``id``, ``display_name``,
    ``workflow_family``, and ``priority`` so the agent can decide which
    manifest to start a dialogue against.
    """
    import json as _json

    from mousehash.blahml.registry import ManifestRegistry

    reg = ManifestRegistry()
    out = []
    for tool_id in reg.all_ids():
        m = reg.by_id(tool_id).manifest
        out.append(
            dict(
                id=m.id,
                display_name=m.display_name,
                workflow_family=m.workflow_family,
                priority=m.priority,
                description=m.description.get("short", ""),
            )
        )
    return _json.dumps(out)


def blahml_start_dialogue(
    tool_id: str,
    scene_set_id: str,
    representation_spec_id: str,
    rule_id: str,
) -> str:
    """Begin a manifest-driven parameter dialogue for one BlahML tool.

    Pre-fills the input_artifact slot from the three identifiers the user
    has already given (or from MouseHash defaults the agent supplied), then
    returns the first manifest-defined question that still needs an answer.

    Args:
        tool_id: BlahML manifest id, e.g. 'run_pca' or 'run_nmf'.
        scene_set_id: Allen scene set, typically 'allen_natural_scenes_v1'.
        representation_spec_id: RepresentationSpec primary key.
        rule_id: AnimateInanimateRule primary key.

    Returns:
        JSON string. If a question is pending, the payload contains
        ``dialogue_id``, ``parameter_name``, ``ask``, ``explain``, and
        optional ``choices``, ``default``, ``default_explanation``, ``range``.
        If the dialogue is already fully resolved, returns a payload with
        ``status="READY"`` and ``dialogue_id``.
    """
    import json as _json
    import uuid

    from mousehash.blahml.question_engine import next_question
    from mousehash.blahml.registry import ManifestRegistry

    reg = ManifestRegistry()
    manifest = reg.by_id(tool_id).manifest

    answers = {
        "input_artifact": dict(
            scene_set_id=scene_set_id,
            representation_spec_id=representation_spec_id,
            rule_id=rule_id,
        )
    }
    dialogue_id = uuid.uuid4().hex[:12]
    _DIALOGUES[dialogue_id] = dict(
        tool_id=tool_id,
        answers=answers,
        question_trace=[],
    )

    pending = next_question(manifest, answers)
    if pending is None:
        return _json.dumps(dict(status="READY", dialogue_id=dialogue_id))
    return _format_question_payload(dialogue_id, pending)


def blahml_submit_answer(
    dialogue_id: str, parameter_name: str, value: str
) -> str:
    """Record one user answer and return the next manifest-defined question.

    Args:
        dialogue_id: The id returned by blahml_start_dialogue.
        parameter_name: The slot being filled (matches ``parameter_name``
            from the previous question payload).
        value: The user's answer as a string. The spec builder coerces it
            to the manifest-declared type.

    Returns:
        Either the next question payload (same shape as
        blahml_start_dialogue) or, if the dialogue is fully resolved, a
        JSON object with ``status="READY"`` and ``dialogue_id``.
    """
    import json as _json

    from mousehash.blahml.question_engine import next_question
    from mousehash.blahml.registry import ManifestRegistry

    if dialogue_id not in _DIALOGUES:
        return _json.dumps(dict(status="ERROR", message=f"unknown dialogue_id {dialogue_id!r}"))

    state = _DIALOGUES[dialogue_id]
    state["answers"][parameter_name] = value
    state["question_trace"].append(
        dict(parameter_name=parameter_name, value=value, source="user")
    )

    reg = ManifestRegistry()
    manifest = reg.by_id(state["tool_id"]).manifest
    pending = next_question(manifest, state["answers"])
    if pending is None:
        return _json.dumps(dict(status="READY", dialogue_id=dialogue_id))
    return _format_question_payload(dialogue_id, pending)


def blahml_run(dialogue_id: str) -> str:
    """Finalize the dialogue, write the ToolRunSpec audit row, and dispatch
    the deterministic tool.

    Args:
        dialogue_id: The id returned by blahml_start_dialogue.

    Returns:
        Human-readable summary including the ``tool_run_spec_id``, the
        derived ``decomposition_spec_id``, any safety warnings, and a
        compact form of the deterministic-tool summary.
    """
    import json as _json

    from mousehash.blahml.executor import run_from_resolved_spec
    from mousehash.blahml.registry import ManifestRegistry
    from mousehash.blahml.spec_builder import build_resolved_spec

    if dialogue_id not in _DIALOGUES:
        return f"ERROR: unknown dialogue_id {dialogue_id!r}"

    state = _DIALOGUES[dialogue_id]
    reg = ManifestRegistry()
    registered = reg.by_id(state["tool_id"])

    resolved = build_resolved_spec(
        registered.manifest,
        state["answers"],
        manifest_sha256=registered.sha256,
        question_trace=state["question_trace"],
        created_by="agent_dialogue",
    )

    result = run_from_resolved_spec(resolved, registry=reg)

    lines = [
        f"BlahML dispatched {result['tool_id']}.",
        f"  tool_run_spec_id     = {result['tool_run_spec_id']}",
        f"  decomposition_spec_id = {result['decomposition_spec_id']}",
    ]
    for w in result.get("warnings", []):
        lines.append(f"  WARNING [{w['check']}]: {w['message']}")
    lines.append(f"  summary = {_json.dumps(result['summary'], default=str)}")
    return "\n".join(lines)
