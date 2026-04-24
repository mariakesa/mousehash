from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_stimulus_decomposition(
    scene_set_id: str,
    representation_spec_id: str,
    rule_id: str,
    decomposition_spec_id: str,
) -> dict:
    """Run PCA or NMF on stored representation arrays and register in DataJoint.

    Loads the appropriate array (logits for PCA, probabilities for NMF) from
    disk using paths stored in ``StimulusRepresentation``, runs the chosen
    decomposition, saves outputs, and inserts a ``StimulusDecomposition`` row.

    Idempotent: returns immediately if the DataJoint record already exists.

    Args:
        scene_set_id:           Primary key of an ``AllenNaturalSceneSet``.
        representation_spec_id: Primary key of a ``RepresentationSpec``.
        rule_id:                Primary key of an ``AnimateInanimateRule``.
        decomposition_spec_id:  Primary key of a ``DecompositionSpec``.

    Returns:
        Summary dict written to ``summary.json`` on disk.
    """
    from mousehash.artifacts.io import load_json, load_npy, save_json, save_npy
    from mousehash.artifacts.paths import decompositions_root
    from mousehash.schema.decompositions import DecompositionSpec, StimulusDecomposition
    from mousehash.schema.representations import StimulusRepresentation

    rep_key = dict(
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        rule_id=rule_id,
    )
    decomp_key = dict(**rep_key, decomposition_spec_id=decomposition_spec_id)

    if StimulusDecomposition & decomp_key:
        logger.info("StimulusDecomposition already exists, skipping.")
        summary_path = (StimulusDecomposition & decomp_key).fetch1("summary_path")
        return load_json(Path(summary_path))

    # --- load spec and representation paths ---
    spec = (DecompositionSpec & {"decomposition_spec_id": decomposition_spec_id}).fetch1()
    rep = (StimulusRepresentation & rep_key).fetch1()

    method = spec["method"]          # "pca" or "nmf"
    input_kind = spec["input_kind"]  # "logits" or "probabilities"
    n_components = spec["n_components"]
    normalize_input = bool(spec["normalize_input"])

    # --- load the correct input array ---
    if input_kind == "logits":
        X = load_npy(Path(rep["logits_path"]))
    elif input_kind == "probabilities":
        X = load_npy(Path(rep["probabilities_path"]))
    else:
        raise ValueError(f"Unknown input_kind: {input_kind!r}")

    logger.info(
        "Running %s on %s (%s), shape=%s",
        method.upper(), input_kind, decomposition_spec_id, X.shape,
    )

    # --- run decomposition ---
    if method == "pca":
        from mousehash.tools.decompositions.pca import run_pca
        result = run_pca(X, n_components=n_components, normalize_input=normalize_input)
    elif method == "nmf":
        from mousehash.tools.decompositions.nmf import run_nmf
        result = run_nmf(X, n_components=n_components)
    else:
        raise ValueError(f"Unknown decomposition method: {method!r}")

    # --- save arrays ---
    out_dir = (
        decompositions_root()
        / scene_set_id
        / representation_spec_id
        / decomposition_spec_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_path = out_dir / "scores.npy"
    components_path = out_dir / "components.npy"
    stats_path = out_dir / "component_stats.json"
    summary_path = out_dir / "summary.json"

    save_npy(scores_path, result["scores"])
    save_npy(components_path, result["components"])

    # per-component stats (method-specific extras stored here)
    if method == "pca":
        component_stats = dict(
            explained_variance_ratio=result["explained_variance_ratio"].tolist(),
            singular_values=result["singular_values"].tolist(),
            cumulative_variance=np.cumsum(result["explained_variance_ratio"]).tolist(),
        )
    else:
        component_stats = dict(
            reconstruction_err=result["reconstruction_err"],
            n_iter=result["n_iter"],
        )
    save_json(stats_path, component_stats)

    summary = dict(
        scene_set_id=scene_set_id,
        representation_spec_id=representation_spec_id,
        rule_id=rule_id,
        decomposition_spec_id=decomposition_spec_id,
        method=method,
        input_kind=input_kind,
        n_components=n_components,
        n_images=int(result["scores"].shape[0]),
        **({
            "explained_variance_ratio_total": float(
                np.sum(result["explained_variance_ratio"])
            )
        } if method == "pca" else {
            "reconstruction_err": result["reconstruction_err"]
        }),
    )
    save_json(summary_path, summary)

    # --- register in DataJoint ---
    StimulusDecomposition.insert1(
        dict(
            **decomp_key,
            scores_path=str(scores_path),
            components_path=str(components_path),
            component_stats_path=str(stats_path),
            summary_path=str(summary_path),
        ),
        skip_duplicates=True,
    )
    logger.info("Registered StimulusDecomposition: %s", decomposition_spec_id)
    return summary
