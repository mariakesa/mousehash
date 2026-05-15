# MouseHash Transformations Specification

**Purpose:** Define the complete transformation layer needed for MouseHash to convert filled role manifests into analysis-ready views that can be consumed by tools.

This document is based on the MouseHash transformation path design. It expands that design into a parser-facing spec: once the parser fills the manifest, this document tells MouseHash what transformations are available, what roles they require, what views they produce, and how they connect to analysis suggestions.

---

## 1. Core design pattern

MouseHash should not treat tools as functions that secretly consume raw dataset roles.

A dataset may contain:

```text
neural_data
stimuli
behavior
conditions
time_organization
metadata
```

But scientific tools usually need a clean analysis-ready object such as:

```text
observations x neurons matrix
trials x time x neurons tensor
stimulus x trial x neuron tensor
design matrix
representational dissimilarity matrix
latent trajectory
functional connectivity graph
```

Therefore the canonical MouseHash pattern is:

```text
Roles → Transformations → AnalysisView → Tool → Artifact
```

Roles describe what exists. Transformations describe how raw roles become analyzable. Views are the actual typed objects tools consume. Tools perform scientific operations. Artifacts remember the result and provenance.

Design slogan:

```text
Roles tell MouseHash what exists.
Transformations tell MouseHash how it becomes analyzable.
Tools tell MouseHash what scientific question to ask.
```

---

## 2. Three-level analysis object model

| Layer | Meaning | Examples |
|---|---|---|
| Roles | Scientific material present in the dataset | Neural data, stimuli, behavior, conditions, time organization, metadata |
| Views | Analysis-ready objects materialized from roles by transformations | Trial x neuron matrix, stimulus x trial x neuron tensor, design matrix, RDM, latent trajectory |
| Tools | Scientific operations applied to views | PCA, GLM, decoder, RSA, HMM, noise correlations, report generator |

---

## 3. Parser-facing transformation decision logic

The parser should not directly suggest a tool merely because a role exists.

Bad:

```text
spikes present → suggest PSTH
```

Better:

```text
spikes present
+ events/trials present
+ alignment rule available
→ suggest transformation path: align spikes to event, bin spikes, make trial x time x neuron tensor
→ then suggest PSTH
```

Pseudocode:

```text
FUNCTION suggest_analysis_moves(role_manifest, tool_catalog, transformation_catalog):

    candidate_moves = []

    FOR each tool IN tool_catalog:

        role_status = check_required_roles(tool.requires_roles, role_manifest)

        IF role_status has missing non-derivable roles:
            candidate_moves.add(blocked_tool_report(tool, role_status))
            CONTINUE

        possible_views = find_views_for_tool(tool.requires_view, transformation_catalog)

        FOR each view_recipe IN possible_views:

            transform_status = check_transform_requirements(view_recipe, role_manifest)

            IF transform_status == satisfiable:
                candidate_moves.add(
                    AnalysisMove(
                        role_signature = tool.requires_roles,
                        transformation_plan = view_recipe.transformations,
                        analysis_view = view_recipe.output_view,
                        tool = tool,
                        validation_plan = default_validation_for_tool(tool),
                        artifact_plan = default_artifacts_for_tool(tool),
                        readiness = ready_or_ready_after_transformation
                    )
                )

            ELSE:
                candidate_moves.add(
                    blocked_view_report(tool, view_recipe, transform_status)
                )

    RETURN rank_candidate_moves(candidate_moves)
```

---

## 4. Core transformation object

Each transformation should be represented as a first-class object.

```yaml
TransformationSpec:
  name:
  family:
  purpose:
  requires_roles:
  requires_inputs:
  produces_view_or_role:
  parameters:
  assumptions:
  failure_modes:
  validation_checks:
  provenance_required:
  leakage_risk:
  default_for_exploration:
  default_for_evaluation:
```

Example:

```yaml
TransformationSpec:
  name: bin_spikes
  family: binning_resampling
  purpose: Convert spike times into spike counts on a common time grid.
  requires_roles:
    - neural_data.spikes
    - time_organization.continuous_time
  requires_inputs:
    - spike_times_by_unit
    - bin_edges_or_bin_size
  produces_view_or_role:
    - time_x_neuron_count_matrix
    - trial_x_time_x_neuron_tensor
  parameters:
    bin_size_ms: required
    alignment_reference: optional
    smoothing_kernel: optional
  assumptions:
    - Spike timestamps are in the same clock as the target events or have a known clock map.
  failure_modes:
    - Wrong clock alignment creates false temporal structure.
    - Too-large bins blur event-locked effects.
    - Too-small bins produce sparse unstable estimates.
  validation_checks:
    - Check spike count distributions.
    - Check event timestamps lie inside recording intervals.
    - Check unit-level firing rates.
  provenance_required:
    - source_nwb_asset
    - units_table_path
    - bin_size_ms
    - event_reference
    - code_version
  leakage_risk: low unless binning depends on train/test labels
```

---

## 5. Canonical transformation families

The transformation catalog has 20 canonical families.

```yaml
transformation_families:
  1_selection_subsetting:
  2_synchronization_clock_correction:
  3_alignment:
  4_segmentation_epoching:
  5_binning_resampling:
  6_filtering_smoothing:
  7_normalization_scaling:
  8_quality_control_artifact_rejection:
  9_feature_extraction:
  10_labeling_annotation:
  11_aggregation_summarization:
  12_residualization_nuisance_removal:
  13_splitting_evaluation_resampling:
  14_tensorization_view_construction:
  15_dimensional_transformation:
  16_distance_similarity_construction:
  17_event_detection:
  18_coordinate_unit_conversion:
  19_missing_data_handling:
  20_artifact_packaging:
```

---

## 6. Family 1: Selection / subsetting

### Purpose

Choose the slice of data to analyze.

### Examples

```text
select_sessions
select_brain_regions
select_neurons
select_units
select_channels
select_trials
exclude_bad_trials
select_time_window
select_subjects
select_genotypes
select_recording_modality
```

### Requires roles

```yaml
required:
  - metadata OR neural_data OR conditions OR time_organization
optional:
  - behavior
  - stimuli
```

### Produces

```text
subset_role_bundle
filtered_metadata_table
selected_units_table
selected_trials_table
selected_time_interval
```

### Parser suggestion triggers

```text
IF metadata.brain_area present:
    suggest select_brain_regions for area-specific analyses

IF metadata.session present:
    suggest select_sessions for cross-session or single-session workflows

IF conditions.trial_labels present:
    suggest select_trials for condition-specific analyses

IF neural_data.spikes/calcium present AND metadata.unit quality exists:
    suggest select_good_units
```

### Failure modes

```text
- Selection can introduce sampling bias.
- Selecting trials after inspecting outcomes can leak labels.
- Excluding units without documented QC criteria reduces reproducibility.
```

---

## 7. Family 2: Synchronization / clock correction

### Purpose

Make clocks and timestamps comparable across devices.

### Examples

```text
sync_device_clocks
correct_timestamp_drift
map_frame_index_to_time
map_stimulus_frame_to_neural_clock
align_behavior_clock_to_neural_clock
resolve_trial_event_order
```

### Requires roles

```yaml
required:
  - time_organization.continuous_time
optional:
  - neural_data
  - stimuli
  - behavior
  - metadata.acquisition_device
```

### Produces

```text
clock_map
synced_event_table
synced_time_series
frame_to_time_table
```

### Parser suggestion triggers

```text
IF multiple time-varying streams exist:
    suggest synchronization check

IF frames present AND continuous neural time present:
    suggest map_frame_index_to_time

IF behavior timestamps and neural timestamps appear in different objects:
    suggest align_behavior_clock_to_neural_clock
```

### Failure modes

```text
- Clock drift creates fake lags.
- Frame index mistaken for seconds creates invalid alignment.
- Missing dropped-frame correction corrupts stimulus-response analysis.
```

---

## 8. Family 3: Alignment

### Purpose

Map streams onto a shared scientific reference event.

### Examples

```text
align_to_stimulus_onset
align_to_choice
align_to_go_cue
align_to_reward
align_to_lick
align_to_perturbation_onset
align_behavior_to_neural_time
align_sessions
```

### Requires roles

```yaml
required:
  - time_organization.events OR time_organization.trials
  - time_organization.alignment_rules
optional:
  - neural_data
  - stimuli
  - behavior
  - conditions
```

### Produces

```text
event_locked_time_base
aligned_trial_table
aligned_neural_segments
aligned_behavior_segments
```

### Parser suggestion triggers

```text
IF spikes/calcium present AND stimulus onset events present:
    suggest align_to_stimulus_onset

IF choices present AND response times present:
    suggest align_to_choice

IF optogenetic intervention present AND stimulation onset present:
    suggest align_to_perturbation_onset
```

### Failure modes

```text
- Wrong event anchor changes scientific interpretation.
- Alignment to post-outcome variables can leak future information.
- Trial times may be in different clocks than neural data.
```

---

## 9. Family 4: Segmentation / epoching

### Purpose

Cut continuous recordings into scientifically meaningful chunks.

### Examples

```text
make_trials
make_epochs
make_sliding_windows
make_baseline_windows
make_event_locked_segments
make_sample_delay_response_epochs
make_sleep_wake_epochs
```

### Requires roles

```yaml
required:
  - time_organization.continuous_time OR time_organization.events OR time_organization.trials
optional:
  - conditions.session_phases
  - neural_data
  - behavior
```

### Produces

```text
trial_segments
epoch_table
window_table
event_locked_segments
```

### Parser suggestion triggers

```text
IF continuous_time present AND events present:
    suggest make_event_locked_segments

IF session_phases present:
    suggest make_phase_epochs

IF trials unknown BUT repeated event structure exists:
    suggest make_trials as derived_possible
```

### Failure modes

```text
- Overlapping windows can violate independence assumptions.
- Epoch definitions may depend on experimental interpretation.
- Baseline windows may overlap preceding trial effects.
```

---

## 10. Family 5: Binning / resampling

### Purpose

Change temporal resolution or sample grid.

### Examples

```text
bin_spikes
downsample_lfp
resample_behavior
aggregate_calcium_frames
interpolate_missing_samples
resample_to_common_time_grid
```

### Requires roles

```yaml
required:
  - time_organization.continuous_time
  - one time-varying data role
optional:
  - alignment_rules
```

### Produces

```text
time_x_neuron_matrix
trial_x_time_x_neuron_tensor
resampled_behavior_matrix
resampled_lfp_matrix
```

### Parser suggestion triggers

```text
IF spikes present AND PSTH/raster/GLM candidate:
    suggest bin_spikes

IF LFP present AND behavior/stimulus alignment candidate:
    suggest downsample_lfp or resample_to_common_time_grid

IF behavior stream and neural stream have different rates:
    suggest resample_behavior
```

### Failure modes

```text
- Bad bin size can erase dynamics or create sparsity.
- Resampling can introduce interpolation artifacts.
- Downsampling without filtering can alias signals.
```

---

## 11. Family 6: Filtering / smoothing

### Purpose

Remove or emphasize temporal/spatial components.

### Examples

```text
lowpass_filter
highpass_filter
bandpass_filter
notch_filter
smooth_spike_counts
gaussian_smooth_psth
filter_lfp_band
extract_band_limited_signal
```

### Requires roles

```yaml
required:
  - neural_data OR behavior
  - time_organization.continuous_time
optional:
  - metadata.acquisition_device
```

### Produces

```text
filtered_time_series
smoothed_rate_matrix
band_limited_lfp
```

### Parser suggestion triggers

```text
IF lfp/eeg present:
    suggest bandpass_filter, notch_filter, extract_lfp_bandpower

IF spikes present AND rate-based analysis requested:
    suggest smooth_spike_counts

IF calcium traces present:
    suggest smoothing only if compatible with event/transient analysis
```

### Failure modes

```text
- Filtering can shift phase if not handled carefully.
- Smoothing can inflate apparent temporal correlation.
- Filter choices can change scientific conclusions.
```

---

## 12. Family 7: Normalization / scaling

### Purpose

Make values comparable while avoiding leakage.

### Examples

```text
zscore_neurons
baseline_subtract
center_features
whiten_features
normalize_embeddings
train_only_scaling
minmax_scale_behavior
normalize_by_session
```

### Requires roles

```yaml
required:
  - analysis_view OR neural_data OR stimuli OR behavior
optional:
  - conditions
  - metadata
  - split_identity
```

### Produces

```text
normalized_matrix
baseline_corrected_tensor
train_fitted_scaler
test_transformed_matrix
```

### Parser suggestion triggers

```text
IF decoder or encoding model candidate:
    suggest train_only_scaling

IF neural population matrix candidate:
    suggest zscore_neurons

IF trial-aligned response candidate:
    suggest baseline_subtract if baseline window exists
```

### Failure modes

```text
- Fitting scaler on all data leaks test information.
- Baseline subtraction can remove meaningful sustained activity.
- Whitening can distort interpretability of original units.
```

---

## 13. Family 8: Quality control / artifact rejection

### Purpose

Detect and remove unreliable units, trials, channels, sessions, or samples.

### Examples

```text
detect_bad_channels
detect_bad_units
detect_motion_artifacts
detect_dropped_frames
detect_timing_mismatches
exclude_low_firing_units
exclude_low_snr_rois
exclude_bad_trials
```

### Requires roles

```yaml
required:
  - neural_data OR behavior OR stimuli
optional:
  - metadata
  - time_organization
```

### Produces

```text
qc_table
quality_mask
filtered_role_bundle
artifact_report
```

### Parser suggestion triggers

```text
IF units table contains quality metrics:
    suggest select_good_units

IF imaging data present:
    suggest detect_motion_artifacts and exclude_bad_rois

IF video/frames present:
    suggest detect_dropped_frames
```

### Failure modes

```text
- QC thresholds may remove biologically interesting units.
- Artifact detection can depend on modality-specific assumptions.
- Hidden QC makes results irreproducible.
```

---

## 14. Family 9: Feature extraction

### Purpose

Turn raw stimuli, behavior, or neural streams into usable features.

### Examples

```text
extract_vit_embeddings
extract_clip_embeddings
extract_image_pixels
extract_pose_features
extract_lfp_bandpower
extract_spike_count_features
extract_calcium_event_counts
extract_pupil_features
extract_running_speed
extract_whisker_angle
```

### Requires roles

```yaml
required:
  - stimuli OR behavior OR neural_data
  - time_organization when time-varying
optional:
  - metadata
```

### Produces

```text
feature_matrix
stimulus_embedding_table
behavior_feature_table
neural_feature_matrix
```

### Parser suggestion triggers

```text
IF visual images/movies present:
    suggest extract_vit_embeddings or extract_clip_embeddings

IF pose present:
    suggest extract_pose_features and kinematics

IF lfp present:
    suggest extract_lfp_bandpower

IF spikes/calcium present:
    suggest extract_spike_count_features or extract_event_counts
```

### Failure modes

```text
- Model-derived features may import external semantic assumptions.
- Feature extraction must be versioned.
- Time-varying features must be aligned to neural clocks.
```

---

## 15. Family 10: Labeling / annotation

### Purpose

Create categorical, semantic, anatomical, or task variables.

### Examples

```text
assign_condition_labels
assign_stimulus_labels
assign_animate_inanimate_labels
assign_choice_labels
assign_brain_region_labels
assign_cell_type_labels
assign_trial_outcome_labels
assign_perturbation_labels
assign_task_epoch_labels
```

### Requires roles

```yaml
required:
  - conditions OR stimuli OR behavior OR metadata OR time_organization
optional:
  - neural_data
```

### Produces

```text
label_vector
condition_table
trial_annotation_table
unit_annotation_table
```

### Parser suggestion triggers

```text
IF stimuli visual images and semantic model features exist:
    suggest assign_animate_inanimate_labels

IF optogenetic stimulation times and trials exist:
    suggest assign_perturbation_labels

IF brain area metadata exists:
    suggest assign_brain_region_labels
```

### Failure modes

```text
- Label definitions can be arbitrary or lab-specific.
- Labels derived from model outputs must be marked as model-derived, not ground truth.
- Trial labels must not be computed using future outcome data unless appropriate.
```

---

## 16. Family 11: Aggregation / summarization

### Purpose

Reduce observations into response summaries or descriptive statistics.

### Examples

```text
average_trials
average_by_stimulus
compute_psth
compute_tuning_curve
compute_response_window_mean
compute_condition_mean_response
compute_session_summary
```

### Requires roles

```yaml
required:
  - neural_data OR behavior
optional:
  - conditions
  - stimuli
  - time_organization
```

### Produces

```text
condition_x_neuron_matrix
stimulus_x_neuron_matrix
condition_x_time_x_neuron_tensor
summary_table
```

### Parser suggestion triggers

```text
IF repeated trials and labels exist:
    suggest average_by_condition

IF spikes and event alignment exist:
    suggest compute_psth

IF stimulus labels and response windows exist:
    suggest compute_tuning_curve
```

### Failure modes

```text
- Averaging hides trial-to-trial variability.
- Unequal trial counts can bias condition means.
- Aggregation must preserve grouping metadata.
```

---

## 17. Family 12: Residualization / nuisance removal

### Purpose

Remove known explanatory factors before another analysis.

### Examples

```text
subtract_stimulus_mean
regress_out_running_speed
regress_out_session_effects
remove_global_signal
remove_batch_effects
regress_out_pupil
regress_out_trial_history
```

### Requires roles

```yaml
required:
  - neural_data
  - nuisance_role OR conditions OR behavior OR metadata
optional:
  - time_organization
```

### Produces

```text
residual_matrix
residual_tensor
nuisance_model_artifact
```

### Parser suggestion triggers

```text
IF noise correlations requested and repeated stimuli exist:
    suggest subtract_stimulus_mean

IF running present and neural activity present:
    suggest regress_out_running_speed for controlled analyses

IF multi-session analysis and session metadata present:
    suggest regress_out_session_effects
```

### Failure modes

```text
- Residualization can remove meaningful signal.
- Nuisance model fitted on all data can leak information.
- Confounded variables may make interpretation unstable.
```

---

## 18. Family 13: Splitting / evaluation resampling

### Purpose

Create train/test/validation logic and null resamples.

### Examples

```text
train_test_split
k_fold_split
leave_one_session_out
time_blocked_split
bootstrap_resample
permutation_shuffle
trial_shuffle
stimulus_shuffle
circular_shift
```

### Requires roles

```yaml
required:
  - analysis_view
optional:
  - conditions
  - time_organization
  - metadata
```

### Produces

```text
split_identity
train_indices
test_indices
resample_indices
null_sample_spec
```

### Parser suggestion triggers

```text
IF predictive model candidate:
    suggest train_test_split or k_fold_split

IF time series model candidate:
    suggest time_blocked_split

IF cross-session analysis candidate:
    suggest leave_one_session_out

IF statistical test candidate:
    suggest permutation_shuffle or bootstrap_resample
```

### Failure modes

```text
- Random split can leak across trials or repeated stimuli.
- Time series requires blocked splits.
- Scaling and feature selection must be fitted only on training data.
```

---

## 19. Family 14: Tensorization / view construction

### Purpose

Materialize the object consumed by a tool.

### Examples

```text
make_trial_by_time_by_neuron_tensor
make_stimulus_by_trial_by_neuron_tensor
make_design_matrix
make_lagged_design_matrix
make_population_state_matrix
make_condition_by_time_by_neuron_tensor
make_unit_by_lag_correlogram_tensor
```

### Requires roles

```yaml
required:
  - transformed_inputs
  - time_organization when temporally structured
optional:
  - conditions
  - metadata
```

### Produces

```text
AnalysisView
matrix
tensor
design_matrix
lagged_design_matrix
population_state_matrix
```

### Parser suggestion triggers

```text
IF tool requires observations x neurons:
    suggest make_population_state_matrix

IF tool requires trials x time x neurons:
    suggest make_trial_by_time_by_neuron_tensor

IF tool requires X predictors and Y neural responses:
    suggest make_design_matrix
```

### Failure modes

```text
- Wrong axis semantics cause invalid tool use.
- Trial averaging before splitting can leak test data.
- Missing metadata makes downstream interpretation weak.
```

---

## 20. Family 15: Dimensional transformation

### Purpose

Change representation space before another tool.

### Examples

```text
pca_transform
nmf_transform
cebra_transform
autoencoder_encode
project_onto_semantic_axis
umap_transform
factor_analysis_transform
```

### Requires roles

```yaml
required:
  - analysis_view
optional:
  - conditions
  - behavior
  - metadata
```

### Produces

```text
latent_matrix
embedding_table
component_scores
projection_values
```

### Parser suggestion triggers

```text
IF high-dimensional population matrix exists:
    suggest PCA or factor model transforms

IF semantic labels exist:
    suggest project_onto_semantic_axis

IF behavior/time labels exist:
    suggest contrastive embeddings
```

### Failure modes

```text
- Fitting dimensionality reduction on all data can leak information.
- Nonlinear embeddings can distort global distances.
- Components require provenance and interpretation metadata.
```

---

## 21. Family 16: Distance / similarity construction

### Purpose

Turn objects into pairwise structure.

### Examples

```text
compute_rdm
compute_kernel_matrix
compute_graph_adjacency
compute_correlation_distance
compute_cosine_similarity
compute_population_distance_matrix
```

### Requires roles

```yaml
required:
  - analysis_view
optional:
  - conditions
  - stimuli
  - behavior
  - metadata
```

### Produces

```text
representational_dissimilarity_matrix
kernel_matrix
similarity_matrix
graph_adjacency_matrix
```

### Parser suggestion triggers

```text
IF RSA or CKA candidate:
    suggest compute_rdm or compute_kernel_matrix

IF connectivity/community detection candidate:
    suggest compute_graph_adjacency

IF stimulus and neural features are both present:
    suggest compute matched distance matrices
```

### Failure modes

```text
- Distance metric choice changes interpretation.
- Pairwise matrices must preserve item identity.
- Unequal trial counts can bias condition-level distances.
```

---

## 22. Family 17: Event detection

### Purpose

Create discrete events from continuous signals.

### Examples

```text
detect_spikes
detect_calcium_events
detect_saccades
detect_licks
detect_movement_onsets
detect_ripples
detect_stimulus_onsets
detect_wheel_movement_onsets
```

### Requires roles

```yaml
required:
  - continuous_time_series
optional:
  - behavior
  - neural_data
  - stimuli
```

### Produces

```text
event_table
event_times
event_labels
```

### Parser suggestion triggers

```text
IF calcium traces present but calcium events absent:
    suggest detect_calcium_events

IF behavior video/lick sensor present:
    suggest detect_licks or movement onsets

IF LFP present and ripple analysis requested:
    suggest detect_ripples
```

### Failure modes

```text
- Event thresholds can create false positives.
- Detection algorithm must be modality-specific.
- Detected events should be marked as derived, not raw.
```

---

## 23. Family 18: Coordinate / unit conversion

### Purpose

Convert measurements into shared units or reference systems.

### Examples

```text
samples_to_seconds
frames_to_seconds
pixels_to_degrees_visual_angle
probe_coordinates_to_brain_regions
volts_to_microvolts
wheel_position_to_speed
image_pixels_to_visual_coordinates
```

### Requires roles

```yaml
required:
  - metadata OR time_organization OR raw measurement
optional:
  - acquisition_device
```

### Produces

```text
converted_table
unit_standardized_view
brain_region_labels
visual_angle_coordinates
```

### Parser suggestion triggers

```text
IF frame indices but no timestamps:
    suggest frames_to_seconds

IF probe coordinates present but brain_area missing:
    suggest probe_coordinates_to_brain_regions

IF image dimensions and screen geometry present:
    suggest pixels_to_degrees_visual_angle
```

### Failure modes

```text
- Missing device calibration prevents accurate conversion.
- Coordinate systems may differ across atlases.
- Unit mistakes can silently corrupt analysis.
```

---

## 24. Family 19: Missing-data handling

### Purpose

Represent, remove, or impute incomplete observations.

### Examples

```text
drop_missing
mask_missing
interpolate_missing_behavior
impute_neural_features
align_only_complete_trials
mark_missing_as_unknown
```

### Requires roles

```yaml
required:
  - analysis_view OR raw role table
optional:
  - time_organization
  - metadata
```

### Produces

```text
missingness_mask
complete_case_view
imputed_view
missingness_report
```

### Parser suggestion triggers

```text
IF multiple streams are aligned and some timestamps are missing:
    suggest align_only_complete_trials or mask_missing

IF behavior stream has gaps:
    suggest interpolate_missing_behavior

IF imputation would affect modeling:
    require human confirmation
```

### Failure modes

```text
- Imputation can invent structure.
- Dropping missing trials can bias conditions.
- Missingness itself may be behaviorally meaningful.
```

---

## 25. Family 20: Artifact packaging

### Purpose

Save durable outputs and register them for provenance.

### Examples

```text
save_model_artifact
save_tensor_artifact
save_metric_table
save_html_report
register_artifact_in_datajoint
save_manifest
save_analysis_move_report
```

### Requires roles

```yaml
required:
  - analysis_result OR analysis_view OR manifest
optional:
  - metadata
  - provenance
```

### Produces

```text
artifact_record
artifact_path
report_bundle
provenance_entry
```

### Parser suggestion triggers

```text
Always suggest artifact packaging for every completed transformation, view, tool run, and report.
```

### Failure modes

```text
- Result without provenance cannot be audited.
- Large arrays should not be stored in metadata tables.
- Artifact paths must remain stable or content-addressed.
```

---

## 26. AnalysisView schema

The parser should recommend tools only when it knows how to produce the required view.

```yaml
AnalysisView:
  view_id:
  view_type:
  source_roles:
  transformation_lineage:
  axis_schema:
  shape:
  units:
  coordinate_system:
  observation_identity:
  condition_identity:
  split_identity:
  artifact_path:
  provenance:
```

Common view types:

```yaml
view_types:
  observations_x_neurons:
    axes:
      rows: observations/trials/time bins
      columns: units/neurons/ROIs/channels

  trials_x_time_x_neurons:
    axes:
      dim0: trial
      dim1: aligned_time
      dim2: unit/neuron/ROI/channel

  conditions_x_time_x_neurons:
    axes:
      dim0: condition
      dim1: aligned_time
      dim2: unit/neuron/ROI/channel

  stimulus_x_trial_x_neuron:
    axes:
      dim0: stimulus_id
      dim1: repeat/trial
      dim2: unit/neuron/ROI/channel

  design_matrix:
    axes:
      rows: observations
      columns: predictors/features

  lagged_design_matrix:
    axes:
      rows: observations/time bins
      columns: feature_lag combinations

  rdm:
    axes:
      rows: items
      columns: items

  latent_trajectory:
    axes:
      dim0: trial/condition/session
      dim1: time
      dim2: latent_dimension

  graph_adjacency:
    axes:
      rows: nodes
      columns: nodes
```

---

## 27. Transformation-aware tool contract

Each tool should declare not only required roles, but also required views and allowed transformation recipes.

```yaml
ToolContract:
  name:
  requires_roles:
  requires_view:
  allowed_transformations:
  default_recipe:
  assumptions:
  failure_modes:
  validation_checks:
  provenance_required:
```

Interpretation:

| Field | Meaning |
|---|---|
| `requires_roles` | Canonical roles that must exist: neural_data, stimuli, behavior, conditions, time_organization, metadata. |
| `requires_view` | The exact analysis-ready object consumed by the tool. |
| `allowed_transformations` | Recipes that can safely produce the view. |
| `default_recipe` | Conservative exploratory default, not a hidden scientific decision. |
| `assumptions` | What must be true for the result to be meaningful. |
| `failure_modes` | How the transformation or tool can mislead, break, leak information, or invent structure. |
| `validation_checks` | Controls and uncertainty procedures that should accompany the analysis. |
| `provenance_required` | Parameters, versions, hashes, source paths, split identity, and artifact paths required for reproducibility. |

---

## 28. Example transformation path: stimulus-aligned ridge encoding

```yaml
analysis_move: stimulus_aligned_ridge_encoding
roles:
  required:
    - neural_data
    - stimuli
    - time_organization
  optional:
    - behavior
    - metadata
transformations:
  - quality_control.select_good_units
  - synchronization.map_stimulus_frames_to_neural_clock
  - alignment.align_to_stimulus_onset
  - segmentation.extract_response_window
  - aggregation.mean_response_in_window
  - feature_extraction.extract_vit_embeddings
  - tensorization.make_design_matrix
  - splitting.k_fold_by_stimulus
  - normalization.zscore_train_only
tool:
  name: fit_ridge_encoding_model
  consumes_view:
    X: observations x stimulus_features
    Y: observations x neurons
artifacts:
  - model_weights
  - cross_validated_scores
  - prediction_plots
  - structure_discovery_report
```

---

## 29. Example transformation path: trial-averaged PCA by stimulus

```yaml
analysis_move: trial_averaged_pca_by_stimulus
roles:
  required:
    - neural_data
    - conditions OR stimuli
    - time_organization
transformations:
  - selection.select_trials
  - alignment.align_to_stimulus_onset
  - segmentation.extract_response_window
  - aggregation.group_by_stimulus_identity
  - aggregation.average_trials
  - normalization.zscore_neurons
  - tensorization.make_condition_x_neuron_matrix
tool:
  name: run_trial_averaged_pca
  consumes_view:
    Y: condition x neuron matrix
artifacts:
  - pca_scores
  - pca_loadings
  - explained_variance
  - visualization
```

---

## 30. Example transformation path: noise correlations

```yaml
analysis_move: noise_correlations
roles:
  required:
    - neural_data
    - conditions OR stimuli
    - time_organization
transformations:
  - alignment.align_trials
  - tensorization.make_stimulus_x_trial_x_neuron_tensor
  - residualization.subtract_stimulus_mean
  - tensorization.compute_trial_residuals
tool:
  name: compute_noise_correlations
  consumes_view:
    residuals: stimulus x trial x neuron tensor
artifacts:
  - noise_correlation_matrix
  - qc_table
  - figure
```

---

## 31. Example transformation path: temporal decoder

```yaml
analysis_move: temporal_decoder
roles:
  required:
    - neural_data
    - behavior OR conditions
    - time_organization
transformations:
  - synchronization.align_neural_and_behavior_clocks
  - tensorization.make_lagged_time_windows
  - splitting.train_test_split_or_time_blocked_split
  - normalization.standardize_train_only
tool:
  name: fit_temporal_decoder
  consumes_view:
    X: time x lagged_neurons
    y: time x target
artifacts:
  - decoder_model
  - held_out_performance
  - confusion_or_trajectory_plot
```

---

## 32. Example transformation path: representational similarity analysis

```yaml
analysis_move: representational_similarity_analysis
roles:
  required:
    - neural_data
    - stimuli OR conditions OR behavior
  optional:
    - time_organization
    - metadata
transformations:
  - tensorization.make_response_matrix
  - aggregation.aggregate_by_stimulus_or_condition
  - normalization.normalize_response_vectors
  - distance_similarity.compute_neural_rdm
  - distance_similarity.compute_stimulus_or_model_rdm
  - splitting.permutation_control
tool:
  name: run_representational_similarity_analysis
  consumes_view:
    neural_rdm: items x items
    target_rdm: items x items
artifacts:
  - neural_rdm
  - target_rdm
  - correlation_statistics
  - permutation_p_values
  - report
```

---

## 33. DataJoint-backed implementation spine

MouseHash should record transformations as scientific computation objects, not incidental preprocessing code.

```yaml
RoleBundle:
  role: Canonical six-role mapping
  type: imported/manual

TransformationSpec:
  role: Parameters and assumptions for a transformation
  type: manual/lookup

TransformationRun:
  role: Execution record including code version, input hashes, output paths, status
  type: computed

AnalysisView:
  role: Analysis-ready object produced by transformations
  type: computed artifact

ToolSpec:
  role: Tool contract with required roles, view, assumptions, failure modes, validation plan
  type: manual/lookup

ToolRun:
  role: Bounded scientific operation applied to an AnalysisView
  type: computed

AnalysisArtifact:
  role: Model, metric table, plot, latent factors, graph, statistical result, or report bundle
  type: computed artifact
```

---

## 34. Reproducibility rules

```text
1. No tool may hide alignment, binning, filtering, normalization, splitting, aggregation, or tensorization inside an opaque implementation.

2. Every transformation has a spec object containing parameters, source roles, output view type, assumptions, and failure modes.

3. Every evaluative workflow must distinguish train-fitted transformations from test-applied transformations to avoid leakage.

4. Every analysis artifact stores the transformation lineage, code version, parameter spec, source data identity, split identity, and artifact paths.

5. Exploratory views and split-safe evaluative views should be different artifacts, even if they use similar code.

6. The same algorithm applied to different views is a different scientific move.
```

---

## 35. Parser-to-transformation matching rules

```text
FUNCTION match_manifest_to_transformations(manifest, transformation_catalog):

    matches = []

    FOR each transformation IN transformation_catalog:

        required_status = check_required_roles(transformation.requires_roles, manifest)

        IF all required roles present:
            matches.add(transformation, status="available")

        ELSE IF some required roles derived_possible:
            matches.add(transformation, status="available_after_derivation")

        ELSE IF some required roles likely_present:
            matches.add(transformation, status="needs_confirmation")

        ELSE:
            matches.add(transformation, status="blocked")

    RETURN matches
```

---

## 36. Readiness status for transformations

```yaml
transformation_readiness:
  available:
    meaning: All required roles and inputs are directly present.
  available_after_derivation:
    meaning: Required roles can be created by earlier transformations.
  needs_confirmation:
    meaning: Some roles are likely but ambiguous.
  blocked:
    meaning: Required roles are missing.
  unsafe_without_human_review:
    meaning: The transformation can materially change interpretation or leak information.
```

---

## 37. Practical MouseHash rule

The unit of analysis should be an **AnalysisMove**, not merely a tool.

```text
AnalysisMove = RoleSignature
             + TransformationPlan
             + AnalysisView
             + Tool
             + ValidationPlan
             + Artifact
```

This turns MouseHash from a bag of neuroscience algorithms into a typed, queryable, reproducible workflow system. It gives agents a stable target:

```text
inspect roles
→ choose/build required view
→ run bounded tool
→ package result into auditable artifact
```
