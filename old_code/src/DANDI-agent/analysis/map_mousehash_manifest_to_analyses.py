#!/usr/bin/env python3
"""
MouseHash analysis mapper

Reads a MouseHash NWB parser manifest and maps inferred roles to possible
AnalysisMoves from the MouseHash 100-tool taxonomy.

This script does NOT run scientific analyses. It compiles a ranked menu:
    RoleBundle -> candidate AnalysisMoves -> required views -> transformations -> tool

Usage:
    python map_mousehash_manifest_to_analyses.py path/to/file.mousehash_manifest.yaml
    python map_mousehash_manifest_to_analyses.py path/to/file.mousehash_manifest.json --out analyses.yaml

Dependencies:
    pip install pyyaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None


@dataclass(frozen=True)
class AnalysisMove:
    id: str
    tool_name: str
    family: str
    priority: str
    requires_any: Tuple[Tuple[str, ...], ...]
    requires_all: Tuple[str, ...]
    optional_roles: Tuple[str, ...]
    requires_view: str
    default_transformations: Tuple[str, ...]
    artifacts: Tuple[str, ...]
    question: str
    assumptions: Tuple[str, ...]
    failure_modes: Tuple[str, ...]
    validation_checks: Tuple[str, ...]


@dataclass
class MappedAnalysis:
    id: str
    tool_name: str
    family: str
    priority: str
    status: str
    score: float
    question: str
    satisfied_roles: List[str]
    missing_roles: List[str]
    optional_roles_present: List[str]
    requires_view: str
    default_transformations: List[str]
    artifacts: List[str]
    assumptions: List[str]
    failure_modes: List[str]
    validation_checks: List[str]
    evidence_summary: Dict[str, Any]


ANALYSIS_MOVES: List[AnalysisMove] = [
    AnalysisMove(
        id="raster_by_trial",
        tool_name="Generate Raster Plot",
        family="visualization",
        priority="mvp",
        requires_all=("neural_data", "time_organization"),
        requires_any=(),
        optional_roles=("conditions", "behavior", "metadata", "stimuli"),
        requires_view="trials x spike/event times",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_spikes_to_trial_events",
            "segmentation.extract_trial_windows",
            "tensorization.make_trial_spike_event_table",
        ),
        artifacts=("raster_plot", "unit_trial_spike_table", "html_report_section"),
        question="When do ALM units fire relative to sample, delay, go cue, lick, or photostim events?",
        assumptions=("Spike times and trial/event timestamps share a clock.",),
        failure_modes=("Misaligned clocks can create false task timing structure.",),
        validation_checks=("Plot event timestamps over trials.", "Check units have non-empty spike_times."),
    ),
    AnalysisMove(
        id="psth_by_task_phase_or_condition",
        tool_name="Generate PSTH Plot",
        family="visualization",
        priority="mvp",
        requires_all=("neural_data", "time_organization"),
        requires_any=(),
        optional_roles=("conditions", "behavior", "stimuli", "metadata"),
        requires_view="condition/trial x time bins -> firing rate",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_to_go_cue_or_sample_or_delay",
            "segmentation.extract_event_locked_windows",
            "binning.bin_spikes",
            "smoothing.gaussian_smooth_psth",
            "aggregation.average_by_condition",
        ),
        artifacts=("psth_plot", "condition_mean_rate_table", "html_report_section"),
        question="How does firing rate evolve across task epochs and perturbation conditions?",
        assumptions=("Task events identify comparable alignment points across trials.",),
        failure_modes=("Averaging across incompatible trial types can hide selectivity.",),
        validation_checks=("Bootstrap confidence intervals over trials.", "Plot trial counts per condition."),
    ),
    AnalysisMove(
        id="choice_decoder",
        tool_name="Fit Logistic Decoder",
        family="encoding_decoding",
        priority="mvp",
        requires_all=("neural_data", "time_organization"),
        requires_any=(("conditions", "behavior"),),
        optional_roles=("metadata", "stimuli"),
        requires_view="observations x neurons -> binary label",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_to_go_cue_or_delay",
            "segmentation.extract_response_or_delay_window",
            "binning.bin_spikes",
            "aggregation.mean_response_in_window",
            "labeling.assign_choice_or_trial_instruction_labels",
            "splitting.k_fold_by_trial_or_session",
            "normalization.zscore_train_only",
        ),
        artifacts=("decoder_model", "cv_accuracy", "confusion_matrix", "html_report_section"),
        question="Can ALM population activity decode the instructed or chosen lick direction?",
        assumptions=("Choice/trial labels are correctly mapped to left/right or anterior/posterior.",),
        failure_modes=("Early lick trials or outcome labels can leak behavioral timing into neural labels.",),
        validation_checks=("Cross-validated decoding.", "Permutation label shuffle.", "Confusion matrix."),
    ),
    AnalysisMove(
        id="perturbation_decoder_or_classifier",
        tool_name="Fit Multiclass Decoder",
        family="encoding_decoding",
        priority="soon",
        requires_all=("neural_data", "conditions", "time_organization"),
        requires_any=(),
        optional_roles=("stimuli", "behavior", "metadata"),
        requires_view="observations x neurons -> perturbation class label",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_to_photostim_or_go_cue",
            "segmentation.extract_epoch_windows",
            "binning.bin_spikes",
            "aggregation.mean_response_in_window",
            "labeling.assign_perturbation_labels",
            "splitting.stratified_k_fold_by_trial",
            "normalization.zscore_train_only",
        ),
        artifacts=("classifier_model", "cv_accuracy", "confusion_matrix", "permutation_null"),
        question="Can neural activity reveal which photoinhibition condition occurred?",
        assumptions=("Perturbation labels are explicit enough to distinguish control/left/right/bilateral.",),
        failure_modes=("If labels are only in metadata but not per trial, this is not runnable yet.",),
        validation_checks=("Permutation test.", "Class balance report.", "Confusion matrix."),
    ),
    AnalysisMove(
        id="poisson_glm_behavior_or_task",
        tool_name="Fit Poisson GLM Encoding Model",
        family="encoding_decoding",
        priority="soon",
        requires_all=("neural_data", "time_organization"),
        requires_any=(("behavior", "conditions", "stimuli"),),
        optional_roles=("metadata",),
        requires_view="time/trial bins x predictors -> spike count",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_behavior_and_spikes",
            "binning.bin_spikes",
            "feature_extraction.extract_behavioral_covariates",
            "labeling.encode_trial_phase_and_perturbation_labels",
            "tensorization.make_design_matrix",
            "splitting.k_fold_by_trial",
            "normalization.train_only_scaling",
        ),
        artifacts=("glm_coefficients", "cv_log_likelihood", "unit_significance_table", "report"),
        question="Which task, behavior, or perturbation variables explain spike counts?",
        assumptions=("Predictors are aligned to spike-count bins.",),
        failure_modes=("Unmodeled time/phase effects can masquerade as choice or perturbation coding.",),
        validation_checks=("Held-out likelihood.", "Baseline/null model comparison.", "Permutation controls."),
    ),
    AnalysisMove(
        id="trial_averaged_pca",
        tool_name="Run Trial-Averaged PCA",
        family="factor_modeling",
        priority="mvp",
        requires_all=("neural_data", "conditions", "time_organization"),
        requires_any=(),
        optional_roles=("behavior", "stimuli", "metadata"),
        requires_view="condition x time x neurons or condition x neurons",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_to_task_event",
            "segmentation.extract_task_epoch_windows",
            "binning.bin_spikes",
            "aggregation.average_trials_by_condition",
            "normalization.zscore_neurons",
            "tensorization.make_condition_by_time_by_neuron_tensor",
        ),
        artifacts=("pca_scores", "pca_loadings", "explained_variance", "trajectory_plot"),
        question="What low-dimensional ALM population structure appears across task phases and conditions?",
        assumptions=("There are enough trials per condition to average meaningfully.",),
        failure_modes=("Only a small number of units makes PCA fragile.",),
        validation_checks=("Bootstrap over trials.", "Report explained variance and trial counts."),
    ),
    AnalysisMove(
        id="dpca_task_choice_time",
        tool_name="Run demixed PCA",
        family="factor_modeling",
        priority="experimental",
        requires_all=("neural_data", "conditions", "time_organization"),
        requires_any=(),
        optional_roles=("behavior", "stimuli", "metadata"),
        requires_view="condition/factor x time x neurons",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_to_task_epoch",
            "binning.bin_spikes",
            "labeling.assign_choice_task_phase_perturbation_factors",
            "aggregation.average_trials_by_factor_combination",
            "tensorization.make_demixed_factor_tensor",
            "normalization.zscore_neurons",
        ),
        artifacts=("dpca_components", "factor_explained_variance", "component_plots"),
        question="Can population variance be separated into time, choice, task instruction, and perturbation factors?",
        assumptions=("Factor labels are explicit and sufficiently crossed.",),
        failure_modes=("Sparse trial combinations or few units make demixing unstable.",),
        validation_checks=("Cross-validation over trials.", "Shuffle controls for labels."),
    ),
    AnalysisMove(
        id="neural_trajectories",
        tool_name="Compute Neural Trajectories",
        family="dynamics",
        priority="mvp",
        requires_all=("neural_data", "time_organization"),
        requires_any=(),
        optional_roles=("conditions", "behavior", "metadata", "stimuli"),
        requires_view="trial/condition x time x latent dimensions",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_to_sample_delay_go",
            "segmentation.extract_trial_time_series",
            "binning.bin_spikes",
            "normalization.zscore_neurons",
            "dimensional_transformation.pca_transform",
            "tensorization.make_trial_by_time_by_latent_tensor",
        ),
        artifacts=("latent_trajectories", "trajectory_plot", "speed_table"),
        question="How does ALM population state evolve from sample to delay to response?",
        assumptions=("Binned spike trains are dense enough to form trajectories.",),
        failure_modes=("Too few units or spikes can make trajectories noisy.",),
        validation_checks=("Bootstrap trajectories.", "Compare against shuffled trial labels."),
    ),
    AnalysisMove(
        id="compare_dynamics_across_perturbations",
        tool_name="Compare Dynamics Across Conditions",
        family="dynamics",
        priority="soon",
        requires_all=("neural_data", "conditions", "time_organization"),
        requires_any=(),
        optional_roles=("behavior", "metadata", "stimuli"),
        requires_view="condition x trial x time x neural/latent",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_to_photostim_or_go_cue",
            "segmentation.extract_task_epoch_windows",
            "binning.bin_spikes",
            "labeling.assign_perturbation_labels",
            "dimensional_transformation.pca_transform",
            "tensorization.make_condition_trial_time_latent_tensor",
        ),
        artifacts=("condition_trajectories", "trajectory_distance_metrics", "permutation_test"),
        question="How do ALM trajectories differ under control, ipsilateral, contralateral, or bilateral photoinhibition?",
        assumptions=("Per-trial perturbation labels exist or can be derived.",),
        failure_modes=("Metadata-only optogenetic site labels may not tell which trials were perturbed.",),
        validation_checks=("Permutation test across condition labels.", "Trial-count balance report."),
    ),
    AnalysisMove(
        id="noise_correlations",
        tool_name="Compute Noise Correlations",
        family="connectivity",
        priority="soon",
        requires_all=("neural_data", "conditions", "time_organization"),
        requires_any=(),
        optional_roles=("stimuli", "behavior", "metadata"),
        requires_view="stimulus/condition x trial x neuron",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_trials",
            "segmentation.extract_response_window",
            "binning.bin_spikes",
            "tensorization.make_condition_by_trial_by_neuron_tensor",
            "residualization.subtract_condition_mean",
            "distance.compute_pairwise_correlations",
        ),
        artifacts=("noise_correlation_matrix", "qc_table", "correlation_plot"),
        question="Do ALM units covary from trial to trial after removing condition means?",
        assumptions=("Repeated trials exist for the same condition labels.",),
        failure_modes=("Few units/trials make correlation estimates unstable.",),
        validation_checks=("Bootstrap confidence intervals.", "Compare against trial shuffle."),
    ),
    AnalysisMove(
        id="cross_correlograms",
        tool_name="Compute Cross-Correlograms",
        family="connectivity",
        priority="mvp",
        requires_all=("neural_data", "time_organization"),
        requires_any=(),
        optional_roles=("conditions", "metadata"),
        requires_view="unit_pair x lag histogram",
        default_transformations=(
            "quality_control.select_good_units",
            "selection.select_unit_pairs",
            "segmentation.optionally_select_task_epochs",
            "tensorization.make_unit_pair_spike_times",
            "binning.compute_lag_histograms",
        ),
        artifacts=("cross_correlogram_table", "cross_correlogram_plot"),
        question="Are there short-lag temporal relationships between simultaneously recorded units?",
        assumptions=("Spike times are precise and units were recorded simultaneously.",),
        failure_modes=("Common task modulation can inflate apparent pairwise coupling.",),
        validation_checks=("Jitter/shuffle control.", "Epoch-stratified correlograms."),
    ),
    AnalysisMove(
        id="rsa_condition_geometry",
        tool_name="Run Representational Similarity Analysis",
        family="geometry",
        priority="soon",
        requires_all=("neural_data", "time_organization"),
        requires_any=(("conditions", "behavior", "stimuli"),),
        optional_roles=("metadata",),
        requires_view="RSM_neural compared to RSM_target",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_to_task_event",
            "segmentation.extract_response_window",
            "binning.bin_spikes",
            "aggregation.average_by_condition",
            "normalization.zscore_neurons",
            "distance.compute_neural_rdm",
            "distance.compute_condition_or_behavior_rdm",
        ),
        artifacts=("neural_rdm", "target_rdm", "rsa_statistics", "permutation_p_values"),
        question="Does ALM population geometry reflect choice, trial instruction, task phase, or perturbation structure?",
        assumptions=("Target labels define a meaningful representational geometry.",),
        failure_modes=("Small number of units/conditions can produce noisy RDMs.",),
        validation_checks=("Permutation test.", "Bootstrap over trials or units."),
    ),
    AnalysisMove(
        id="class_separability",
        tool_name="Compute Class Separability in Neural Space",
        family="geometry",
        priority="mvp",
        requires_all=("neural_data", "conditions"),
        requires_any=(),
        optional_roles=("time_organization", "behavior", "stimuli", "metadata"),
        requires_view="observations x neurons + labels",
        default_transformations=(
            "quality_control.select_good_units",
            "alignment.align_to_task_event",
            "segmentation.extract_epoch_window",
            "binning.bin_spikes",
            "aggregation.mean_response_in_window",
            "labeling.assign_condition_labels",
            "normalization.zscore_train_only",
        ),
        artifacts=("separability_metrics", "margin_distribution", "projection_plot"),
        question="How separable are trial instruction, choice, or perturbation labels in ALM neural space?",
        assumptions=("Condition labels exist at the observation/trial level.",),
        failure_modes=("Separability without cross-validation can overstate structure.",),
        validation_checks=("Train/test split.", "Permutation labels.", "Class balance report."),
    ),
    AnalysisMove(
        id="state_change_detection",
        tool_name="Detect Neural State Changes",
        family="dynamics",
        priority="experimental",
        requires_all=("neural_data", "time_organization"),
        requires_any=(),
        optional_roles=("conditions", "behavior", "metadata"),
        requires_view="time x neural features -> changepoints",
        default_transformations=(
            "quality_control.select_good_units",
            "binning.bin_spikes",
            "smoothing.smooth_spike_counts",
            "tensorization.make_population_state_matrix",
            "event_detection.detect_changepoints",
        ),
        artifacts=("changepoint_table", "state_change_plot"),
        question="Are there abrupt ALM population-state transitions around sample, delay, go cue, lick, or perturbation?",
        assumptions=("Temporal resolution is sufficient for state-change detection.",),
        failure_modes=("Task-locked events can be rediscovered trivially unless compared to event times.",),
        validation_checks=("Circular shift control.", "Compare changepoints to known task events."),
    ),
    AnalysisMove(
        id="bootstrap_confidence_intervals",
        tool_name="Run Bootstrap Confidence Intervals",
        family="statistics_validation",
        priority="mvp",
        requires_all=("metadata",),
        requires_any=(("neural_data", "conditions", "behavior", "time_organization"),),
        optional_roles=("stimuli",),
        requires_view="resampled observations/trials/sessions",
        default_transformations=(
            "splitting.bootstrap_resample_trials_or_units",
            "artifact_packaging.save_metric_table",
        ),
        artifacts=("bootstrap_distribution", "confidence_interval_table"),
        question="How uncertain are decoded accuracies, firing-rate differences, trajectories, or correlations?",
        assumptions=("A scientifically valid resampling unit is defined.",),
        failure_modes=("Bootstrapping spikes as independent can break trial/session structure.",),
        validation_checks=("Declare resampling unit.", "Stratify by condition when needed."),
    ),
    AnalysisMove(
        id="permutation_test",
        tool_name="Run Permutation Test",
        family="statistics_validation",
        priority="mvp",
        requires_all=("time_organization",),
        requires_any=(("conditions", "behavior", "stimuli"),),
        optional_roles=("neural_data", "metadata"),
        requires_view="observed metric + shuffled null metrics",
        default_transformations=(
            "splitting.permutation_shuffle_condition_labels",
            "artifact_packaging.save_metric_table",
        ),
        artifacts=("null_distribution", "p_value_table", "shuffle_report"),
        question="Does observed neural structure exceed a label-shuffled null?",
        assumptions=("The shuffle respects trial/session/temporal exchangeability.",),
        failure_modes=("Naive shuffles can destroy temporal structure or leak labels.",),
        validation_checks=("Block shuffling by session/trial where appropriate.",),
    ),
    AnalysisMove(
        id="structure_discovery_report",
        tool_name="Generate Structure Discovery Report",
        family="reporting",
        priority="mvp",
        requires_all=("metadata",),
        requires_any=(),
        optional_roles=("neural_data", "conditions", "behavior", "stimuli", "time_organization"),
        requires_view="tool artifacts + role manifests -> report",
        default_transformations=(
            "artifact_packaging.collect_manifest",
            "artifact_packaging.collect_analysis_artifacts",
            "artifact_packaging.save_html_report",
        ),
        artifacts=("html_report", "analysis_index_json"),
        question="What did MouseHash infer, what can it analyze, and what evidence/provenance supports each move?",
        assumptions=("Each artifact has provenance and parameter records.",),
        failure_modes=("A report without lineage becomes a pretty notebook, not a scientific object.",),
        validation_checks=("Check every figure/metric links to source manifest and transformation spec.",),
    ),
]

ROLE_NAMES = {"neural_data", "stimuli", "behavior", "conditions", "time_organization", "metadata"}


def load_manifest(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML input: pip install pyyaml")
        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is not None:
        try:
            return yaml.safe_load(text)
        except Exception:
            pass
    return json.loads(text)


def role_present(manifest: Dict[str, Any], role: str) -> bool:
    roles = manifest.get("mousehash_roles", {})
    value = roles.get(role)
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        for _, v in value.items():
            if isinstance(v, dict):
                if v.get("inferred") or v.get("sources"):
                    return True
            elif isinstance(v, (list, tuple, set)) and v:
                return True
            elif bool(v):
                return True
        return False
    return bool(value)


def present_roles(manifest: Dict[str, Any]) -> Set[str]:
    return {role for role in ROLE_NAMES if role_present(manifest, role)}


def get_role_sources(manifest: Dict[str, Any], role: str) -> Dict[str, Any]:
    return manifest.get("role_sources", {}).get(role, {})


def infer_manifest_capabilities(manifest: Dict[str, Any]) -> Dict[str, Any]:
    n_units = manifest.get("nwb_summary", {}).get("units", {}).get("n_units")
    n_spikes = manifest.get("nwb_summary", {}).get("units", {}).get("n_spikes_total")
    n_trials = manifest.get("file_summary", {}).get("n_trials")
    trial_columns = set(manifest.get("nwb_summary", {}).get("trial_columns", []) or [])
    conditions_sources = manifest.get("role_sources", {}).get("conditions", {})
    behavior_sources = manifest.get("role_sources", {}).get("behavior", {})
    stimuli = manifest.get("mousehash_roles", {}).get("stimuli", {})
    return {
        "n_units": n_units,
        "n_spikes_total": n_spikes,
        "n_trials": n_trials,
        "trial_columns": sorted(trial_columns),
        "has_units": bool(n_units and n_units > 0),
        "has_enough_units_for_population_methods": bool(n_units and n_units >= 10),
        "has_many_units_for_population_methods": bool(n_units and n_units >= 30),
        "has_trials": bool(n_trials and n_trials > 0),
        "has_trial_instruction": "trial_instruction" in trial_columns,
        "has_outcome": "outcome" in trial_columns,
        "has_early_lick": "early_lick" in trial_columns,
        "has_task_protocol": "task_protocol" in trial_columns,
        "has_behavioral_lick_trace": "choices" in behavior_sources,
        "has_perturbation_metadata": "perturbation_labels" in conditions_sources,
        "has_optogenetic_intervention": bool(isinstance(stimuli, dict) and stimuli.get("interventions"))
        or "optogenetic_intervention" in manifest.get("role_sources", {}).get("stimuli", {}),
        "has_sensory_stimulus_inferred": bool(isinstance(stimuli, dict) and stimuli.get("sensory")),
    }


def requirement_satisfied(move: AnalysisMove, roles: Set[str]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for r in move.requires_all:
        if r not in roles:
            missing.append(r)
    for any_group in move.requires_any:
        if not any(r in roles for r in any_group):
            missing.append(" OR ".join(any_group))
    return len(missing) == 0, missing


def score_move(move: AnalysisMove, manifest: Dict[str, Any]) -> float:
    roles = present_roles(manifest)
    ok, missing = requirement_satisfied(move, roles)
    caps = infer_manifest_capabilities(manifest)
    score = 0.0
    score += 60.0 if ok else -20.0 * len(missing)
    score += 4.0 * len([r for r in move.optional_roles if r in roles])
    score += {"mvp": 20.0, "soon": 10.0, "experimental": 0.0}.get(move.priority, 0.0)

    if move.id in {"raster_by_trial", "psth_by_task_phase_or_condition", "cross_correlograms"}:
        score += 10.0
    if move.id in {"trial_averaged_pca", "neural_trajectories", "dpca_task_choice_time"}:
        if caps["has_many_units_for_population_methods"]:
            score += 10.0
        elif caps["has_enough_units_for_population_methods"]:
            score += 2.0
        else:
            score -= 15.0
    if move.id in {"perturbation_decoder_or_classifier", "compare_dynamics_across_perturbations"}:
        if caps["has_perturbation_metadata"]:
            score += 8.0
        score -= 5.0
    if move.id in {"choice_decoder", "class_separability"} and (caps["has_trial_instruction"] or caps["has_behavioral_lick_trace"]):
        score += 8.0
    if move.id == "poisson_glm_behavior_or_task" and caps["has_behavioral_lick_trace"]:
        score += 5.0
    if move.id in {"rsa_condition_geometry", "noise_correlations"} and caps["has_trials"]:
        score += 5.0
    return round(score, 3)


def map_analyses(manifest: Dict[str, Any], include_not_runnable: bool = True) -> List[MappedAnalysis]:
    roles = present_roles(manifest)
    caps = infer_manifest_capabilities(manifest)
    results: List[MappedAnalysis] = []
    for move in ANALYSIS_MOVES:
        ok, missing = requirement_satisfied(move, roles)
        status = "runnable"
        if not ok:
            status = "missing_required_roles"
        elif move.id in {"perturbation_decoder_or_classifier", "compare_dynamics_across_perturbations"}:
            if caps["has_perturbation_metadata"]:
                status = "needs_label_disambiguation"
        elif move.id == "dpca_task_choice_time":
            status = "possible_but_needs_factor_check"
        if not include_not_runnable and status == "missing_required_roles":
            continue
        results.append(
            MappedAnalysis(
                id=move.id,
                tool_name=move.tool_name,
                family=move.family,
                priority=move.priority,
                status=status,
                score=score_move(move, manifest),
                question=move.question,
                satisfied_roles=sorted([r for r in ROLE_NAMES if r in roles]),
                missing_roles=missing,
                optional_roles_present=sorted([r for r in move.optional_roles if r in roles]),
                requires_view=move.requires_view,
                default_transformations=list(move.default_transformations),
                artifacts=list(move.artifacts),
                assumptions=list(move.assumptions),
                failure_modes=list(move.failure_modes),
                validation_checks=list(move.validation_checks),
                evidence_summary={
                    "present_roles": sorted(roles),
                    "capabilities": caps,
                    "role_sources": {role: get_role_sources(manifest, role) for role in sorted(roles)},
                },
            )
        )
    results.sort(key=lambda x: x.score, reverse=True)
    return results


def summarize(results: List[MappedAnalysis], top_k: int) -> Dict[str, Any]:
    return {
        "n_candidates": len(results),
        "n_runnable": sum(r.status == "runnable" for r in results),
        "n_needs_label_disambiguation": sum(r.status == "needs_label_disambiguation" for r in results),
        "top_candidates": [
            {
                "id": r.id,
                "tool_name": r.tool_name,
                "family": r.family,
                "priority": r.priority,
                "status": r.status,
                "score": r.score,
                "question": r.question,
                "requires_view": r.requires_view,
                "default_transformations": r.default_transformations,
                "artifacts": r.artifacts,
                "missing_roles": r.missing_roles,
                "validation_checks": r.validation_checks,
            }
            for r in results[:top_k]
        ],
    }


def write_output(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML output: pip install pyyaml")
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    else:
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Map a MouseHash manifest to possible AnalysisMoves.")
    parser.add_argument("manifest", type=Path, help="Path to parser-generated MouseHash manifest YAML/JSON.")
    parser.add_argument("--out", type=Path, default=None, help="Output path, .yaml/.yml or .json.")
    parser.add_argument("--top-k", type=int, default=15, help="Number of top analyses to show in summary.")
    parser.add_argument("--runnable-only", action="store_true", help="Only include analyses with required roles satisfied.")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    results = map_analyses(manifest, include_not_runnable=not args.runnable_only)
    payload = {
        "source_manifest": str(args.manifest),
        "present_roles": sorted(present_roles(manifest)),
        "manifest_capabilities": infer_manifest_capabilities(manifest),
        "summary": summarize(results, args.top_k),
        "analyses": [asdict(r) for r in results],
    }
    out_path = args.out
    if out_path is None:
        suffix = ".analysis_map.yaml" if yaml is not None else ".analysis_map.json"
        out_path = args.manifest.with_suffix(suffix)
    write_output(payload, out_path)
    print(f"[ok] wrote analysis map: {out_path}")
    print("\nTop candidates:")
    for row in payload["summary"]["top_candidates"]:
        print(f"  {row['score']:>6}  {row['status']:<28} {row['id']:<36} {row['tool_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
