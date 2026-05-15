# MouseHash FastMCP Architecture

**Purpose:** define a library architecture for MouseHash as a FastMCP-powered neuroscience workflow system that can connect to **Allen Brain Observatory**, **IBL**, and **DANDI**, while sharing transformations, statistical models, validation logic, reports, and provenance across targets.

This document assumes the MouseHash design rule:

> **Targets provide data. Roles describe what exists. Transformations create analysis-ready views. Tools perform bounded scientific operations. Artifacts remember what happened. FastMCP exposes the system to agents.**

---

## 1. The architectural thesis

MouseHash should not become three separate projects:

- `mousehash_allen`
- `mousehash_ibl`
- `mousehash_dandi`

That would duplicate models, statistics, transformations, reports, validation logic, and agent-facing tool contracts.

Instead, MouseHash should have:

1. **Target-specific adapters**
   - Allen adapter
   - IBL adapter
   - DANDI/NWB adapter

2. **A shared MouseHash scientific core**
   - roles
   - manifests
   - transformations
   - analysis views
   - statistical tools
   - validation controls
   - artifact packaging
   - provenance

3. **A FastMCP server layer**
   - exposes target-specific functions
   - exposes shared scientific functions
   - exposes query/readiness/reporting functions
   - keeps LLMs away from raw filesystem/database chaos

The shape is:

```text
Claude / ChatGPT / Cursor / local agent
             |
             v
      MouseHash FastMCP Server
             |
             v
  -------------------------------------------------
  | Target MCP Tools | Shared Scientific MCP Tools |
  -------------------------------------------------
             |
             v
      MouseHash Python Library Core
             |
             v
  -------------------------------------------------
  | Allen Adapter | IBL Adapter | DANDI/NWB Adapter |
  -------------------------------------------------
             |
             v
  DataJoint provenance + disk/object-store artifacts
```

FastMCP is the agent-facing door. The Python library is the real engine. DataJoint/artifacts are the lab notebook and freezer.

---

## 2. Design principles

### 2.1 Agents decide, tools do, artifacts remember

The agent should not improvise Python code against AllenSDK, ONE, or pynwb. The agent should call bounded tools.

Examples:

```text
Good:
  "scan_dandiset_roles(dandiset_id='000XXX')"

Bad:
  "Write arbitrary code that opens a random NWB URL and guesses what to do."
```

### 2.2 Target adapters are ingestion and access layers, not analysis systems

Allen, IBL, and DANDI each need special handling. But once data is mapped into MouseHash roles, downstream tools should not care where the data came from.

A ridge encoding model should consume:

```text
X: observations x stimulus_features
Y: observations x neurons
```

not:

```text
Allen-specific observatory session object
```

or:

```text
IBL-specific ephys session object
```

or:

```text
DANDI-specific NWB file handle
```

### 2.3 Transformations are first-class scientific objects

No tool is allowed to secretly align, bin, z-score, split, smooth, filter, or aggregate data.

Those are explicit transformations:

```text
raw target data
  -> RoleBundle
  -> TransformationPlan
  -> AnalysisView
  -> ToolRun
  -> Artifact
```

This is the anti-haunted-notebook rule.

### 2.4 The unit of execution is an AnalysisMove

A tool alone is too underspecified.

`PCA` is not a scientific analysis until we know:

- what roles were required
- what transformations produced the matrix
- whether the matrix was trial-averaged or continuous
- whether normalization was fit globally or train-only
- what validation controls were attached
- where the artifact was saved

So the central object should be:

```text
AnalysisMove =
    Target
  + RoleSignature
  + TransformationPlan
  + AnalysisView
  + Tool
  + ValidationPlan
  + Artifact
```

---

## 3. Package layout

Recommended repository layout:

```text
mousehash/
  pyproject.toml
  README.md
  docs/
    architecture/
      fastmcp_architecture.md
      role_bundle.md
      transformation_paths.md
      tool_contracts.md
    examples/
      allen_natural_scenes.md
      ibl_choice_decoding.md
      dandi_nwb_scan.md

  configs/
    targets/
      allen.yaml
      ibl.yaml
      dandi.yaml
    tools/
      tool_registry.yaml
      analysis_moves.yaml
    mcp/
      server.yaml

  scripts/
    mousehash-server
    mousehash-scan-dandi
    mousehash-scan-allen
    mousehash-scan-ibl
    mousehash-run-analysis

  src/
    mousehash/
      __init__.py

      core/
        ids.py
        types.py
        errors.py
        registry.py
        contracts.py
        manifests.py
        role_bundle.py
        analysis_move.py
        analysis_view.py
        validation.py
        provenance.py

      targets/
        base.py
        registry.py

        allen/
          __init__.py
          client.py
          manifest.py
          roles.py
          resources.py
          loaders.py
          transforms.py

        ibl/
          __init__.py
          client.py
          manifest.py
          roles.py
          resources.py
          loaders.py
          transforms.py

        dandi/
          __init__.py
          client.py
          manifest.py
          roles.py
          nwb_inspector.py
          resources.py
          loaders.py
          transforms.py

      transformations/
        base.py
        specs.py
        registry.py
        selection.py
        synchronization.py
        alignment.py
        segmentation.py
        binning.py
        filtering.py
        normalization.py
        quality_control.py
        feature_extraction.py
        labeling.py
        aggregation.py
        residualization.py
        splitting.py
        tensorization.py
        dimensional.py
        similarity.py
        event_detection.py
        coordinate_conversion.py
        missing_data.py
        artifact_packaging.py

      views/
        base.py
        matrix.py
        tensor.py
        design_matrix.py
        rdm.py
        latent.py
        graph.py
        report_bundle.py

      tools/
        base.py
        registry.py
        contracts.py

        encoding/
          linear.py
          ridge.py
          lasso.py
          glm.py
          temporal.py

        decoding/
          logistic.py
          multiclass.py
          ridge_decoder.py
          temporal_decoder.py

        factor_models/
          pca.py
          nmf.py
          fa.py
          gpfa.py
          cebra.py
          autoencoder.py

        geometry/
          rdm.py
          rsa.py
          cka.py
          cca.py
          procrustes.py
          subspaces.py
          semantic_axes.py

        dynamics/
          lds.py
          hmm.py
          trajectories.py
          jpca.py
          dmd.py
          timescales.py

        connectivity/
          correlations.py
          functional_connectivity.py
          graphical_lasso.py
          cross_correlograms.py
          coupling_glm.py
          assemblies.py

        statistics/
          permutation.py
          bootstrap.py
          shuffles.py
          multiple_comparisons.py

        visualization/
          raster.py
          psth.py
          latent_trajectory.py

        reports/
          structure_discovery.py
          html.py

      artifacts/
        base.py
        paths.py
        io.py
        hashes.py
        model_artifact.py
        table_artifact.py
        figure_artifact.py
        html_artifact.py
        bundle.py

      schema/
        __init__.py
        connection.py
        role_schema.py
        transformation_schema.py
        view_schema.py
        tool_schema.py
        artifact_schema.py
        target_schema.py

      mcp/
        __init__.py
        server.py
        context.py
        auth.py
        resources.py
        prompts.py

        tools/
          target_tools.py
          manifest_tools.py
          readiness_tools.py
          transformation_tools.py
          analysis_tools.py
          artifact_tools.py
          report_tools.py

      pipelines/
        base.py
        planner.py
        executor.py
        recipes.py
        allen_natural_scenes.py
        ibl_choice_decoding.py
        dandi_structure_scan.py

      testing/
        fixtures.py
        synthetic_nwb.py
        golden_manifests.py
        contract_tests.py

  tests/
    unit/
    integration/
    mcp/
    fixtures/
```

The goblin rule: target-specific code lives in `targets/`; reusable scientific code lives in `transformations/`, `views/`, `tools/`, `artifacts/`, and `schema/`.

---

## 4. Main runtime layers

## 4.1 Target layer

The target layer knows how to speak to each external ecosystem.

### Shared target interface

```python
class TargetAdapter(Protocol):
    target_name: str

    def list_datasets(self, query: DatasetQuery) -> list[DatasetRef]:
        ...

    def get_dataset_metadata(self, dataset_ref: DatasetRef) -> DatasetMetadata:
        ...

    def build_manifest(self, dataset_ref: DatasetRef) -> RoleManifest:
        ...

    def materialize_resource(self, resource_ref: ResourceRef) -> LocalResource:
        ...

    def load_role_bundle(self, manifest_id: str) -> RoleBundle:
        ...
```

The adapter has one job: translate target-specific reality into MouseHash concepts.

### Allen adapter

Responsibilities:

- connect to AllenSDK / Allen Brain Observatory resources
- list experiments, sessions, stimuli, brain regions, cells/units
- map Allen objects into MouseHash roles
- expose target-specific functions for natural scenes, natural movies, calcium traces, Neuropixels, running, pupil, metadata
- produce Allen-specific manifests

Example Allen target functions:

```text
allen_list_experiments()
allen_get_session_metadata(session_id)
allen_list_stimuli(session_id)
allen_load_natural_scenes(session_id)
allen_load_calcium_traces(session_id)
allen_load_neuropixels_units(session_id)
allen_build_role_manifest(session_id)
allen_materialize_role_bundle(session_id)
```

Allen-specific outputs become standard roles:

```text
Allen natural scenes          -> stimuli
Allen calcium traces          -> neural_data
Allen running speed           -> behavior
Allen stimulus presentations  -> time_organization + conditions
Allen cell metadata           -> metadata
```

### IBL adapter

Responsibilities:

- connect through ONE / Alyx
- list subjects, sessions, probes, trials, task variables
- load spikes, clusters, LFP where available
- map task trials, choices, feedback, reaction time, wheel movement
- produce IBL-specific manifests

Example IBL target functions:

```text
ibl_search_sessions(subject=None, task_protocol=None, brain_region=None)
ibl_get_session_metadata(eid)
ibl_load_trials(eid)
ibl_load_spikes(eid, probe)
ibl_load_clusters(eid, probe)
ibl_load_behavior(eid)
ibl_build_role_manifest(eid)
ibl_materialize_role_bundle(eid)
```

IBL-specific outputs become standard roles:

```text
IBL spikes/clusters      -> neural_data + metadata
IBL trials               -> conditions + time_organization
IBL choices/reaction     -> behavior
IBL probes/brain regions -> metadata
```

### DANDI adapter

Responsibilities:

- query DANDI metadata
- inspect dandisets and assets
- inspect NWB structure with `pynwb`/`h5py`
- infer role presence from metadata and NWB paths
- support remote and local NWB reading
- produce role manifests for dandisets and individual assets

Example DANDI target functions:

```text
dandi_search_dandisets(query)
dandi_get_dandiset_metadata(dandiset_id)
dandi_list_assets(dandiset_id)
dandi_inspect_nwb_asset(asset_id)
dandi_build_role_manifest(dandiset_id, asset_strategy="representative")
dandi_materialize_role_bundle(dandiset_id, asset_id=None)
```

DANDI-specific outputs become standard roles:

```text
NWB units/spike_times             -> neural_data
NWB acquisition timeseries        -> neural_data or behavior depending on type
NWB trials table                  -> conditions + time_organization
NWB stimulus presentation tables  -> stimuli + time_organization
NWB processing/behavior modules   -> behavior
NWB subject/electrodes/optogenetics metadata -> metadata
```

---

## 5. Role layer

MouseHash should use a canonical six-role bundle:

```python
@dataclass
class RoleBundle:
    conditions: ConditionsRole | None
    stimuli: StimuliRole | None
    behavior: BehaviorRole | None
    neural_data: NeuralDataRole | None
    time_organization: TimeOrganizationRole | None
    metadata: MetadataRole | None
```

Each role should contain:

```python
@dataclass
class RoleEvidence:
    status: Literal["present", "likely", "absent", "unknown", "derivable"]
    confidence: Literal["high", "medium", "low"]
    source: str
    path: str | None
    notes: str | None
```

The manifest is not only a checklist. It is evidence-backed.

Example:

```yaml
dataset:
  target: dandi
  dandiset_id: "000XXX"

roles:
  neural_data:
    status: present
    confidence: high
    evidence:
      - source: nwb
        path: /units/spike_times
        notes: spike times detected

  conditions:
    status: present
    confidence: high
    evidence:
      - source: nwb
        path: /intervals/trials
        notes: trials table contains stimulus_name and condition columns

  stimuli:
    status: likely
    confidence: medium
    evidence:
      - source: nwb
        path: /stimulus/presentation
        notes: stimulus timestamps present, raw stimulus features may require extraction

  time_organization:
    status: present
    confidence: high
    evidence:
      - source: nwb
        path: /intervals/trials
        notes: start_time and stop_time available
```

The manifest is the bridge between target-specific data and generic MouseHash tools.

---

## 6. Transformation layer

Transformations are shared across all targets.

Allen, IBL, and DANDI may load data differently, but all of them eventually need things like:

- selecting neurons
- selecting trials
- aligning to events
- binning spikes
- extracting response windows
- z-scoring train-only
- building design matrices
- computing RDMs
- creating train/test splits
- packaging artifacts

Recommended transformation interface:

```python
class Transformation(Protocol):
    name: str
    family: str

    def validate_inputs(self, inputs: RoleBundle | AnalysisView) -> ValidationResult:
        ...

    def run(self, inputs: RoleBundle | AnalysisView, spec: TransformationSpec) -> AnalysisView:
        ...

    def provenance(self) -> TransformationProvenance:
        ...
```

### Transformation families

MouseHash should implement these families as first-class modules:

| Family | Purpose |
|---|---|
| selection | choose sessions, neurons, trials, areas, time windows |
| synchronization | map device clocks and frame indices |
| alignment | align streams to stimulus onset, choice, reward, movement |
| segmentation | create trials, epochs, windows, baseline periods |
| binning | bin spikes, resample behavior, aggregate calcium frames |
| filtering | smooth spike counts, filter LFP, remove line noise |
| normalization | z-score, baseline subtract, whiten, train-only scaling |
| quality control | remove bad units, channels, trials, sessions |
| feature extraction | ViT/CLIP embeddings, pose features, LFP bandpower |
| labeling | stimulus labels, choice labels, semantic labels, brain regions |
| aggregation | PSTHs, tuning curves, trial averages, response windows |
| residualization | remove stimulus means, running speed, session effects |
| splitting | train/test, k-fold, blocked time splits, permutation, bootstrap |
| tensorization | design matrices, trial x time x neuron tensors, population matrices |
| dimensional | PCA/NMF/CEBRA transforms, semantic-axis projection |
| similarity | RDMs, kernels, graph adjacency, cosine/correlation distance |
| event detection | spikes, calcium events, licks, saccades, ripples |
| coordinate conversion | samples to seconds, frames to seconds, probe coordinate mapping |
| missing data | masks, interpolation, complete-trial filtering |
| artifact packaging | save arrays, models, figures, reports, DataJoint records |

Target-specific transforms may exist, but they should be thin wrappers around shared transformations whenever possible.

Example:

```text
allen.transforms.map_stimulus_presentations_to_time()
  -> uses shared synchronization + alignment concepts

ibl.transforms.align_spikes_to_go_cue()
  -> uses shared alignment + binning + tensorization

dandi.transforms.nwb_trials_to_events()
  -> uses shared segmentation + labeling
```

---

## 7. AnalysisView layer

Tools should consume explicit `AnalysisView` objects, not raw target data.

Suggested view types:

```python
class AnalysisViewKind(StrEnum):
    OBSERVATION_BY_NEURON = "observation_by_neuron"
    OBSERVATION_BY_FEATURE = "observation_by_feature"
    TRIAL_TIME_NEURON = "trial_time_neuron"
    CONDITION_TIME_NEURON = "condition_time_neuron"
    DESIGN_MATRIX = "design_matrix"
    LAGGED_DESIGN_MATRIX = "lagged_design_matrix"
    RDM = "rdm"
    LATENT_TRAJECTORY = "latent_trajectory"
    FUNCTIONAL_GRAPH = "functional_graph"
    METRIC_TABLE = "metric_table"
    REPORT_BUNDLE = "report_bundle"
```

Example:

```yaml
view:
  view_id: "view_abc123"
  kind: observation_by_neuron
  shape: [118, 240]
  axes:
    observations: stimulus_presentations
    features: neurons
  source_roles:
    - neural_data
    - stimuli
    - time_organization
  transformation_lineage:
    - select_good_units
    - align_to_stimulus_onset
    - extract_response_window
    - mean_response_in_window
    - zscore_neurons
```

The view object is what lets MouseHash say:

> This PCA was run on a stimulus-averaged neural response matrix, not on raw time series, not on trial tensor, not on calcium traces before alignment.

That distinction matters scientifically.

---

## 8. Tool layer

Tools are shared scientific operations.

A tool contract should declare:

```yaml
tool:
  name: fit_ridge_encoding_model
  family: encoding
  requires_roles:
    required:
      - stimuli
      - neural_data
      - time_organization
    optional:
      - behavior
      - conditions
      - metadata
  consumes_view:
    X: observation_by_feature
    Y: observation_by_neuron
  allowed_transformations:
    - select_good_units
    - align_to_stimulus_onset
    - extract_response_window
    - mean_response_in_window
    - extract_vit_embeddings
    - make_design_matrix
    - k_fold_by_stimulus
    - zscore_train_only
  produces:
    - model_artifact
    - metric_table
    - prediction_plot
    - report_bundle
  assumptions:
    - observations in X and Y are aligned
    - train/test split prevents stimulus leakage
    - normalization is fit only on train folds
  failure_modes:
    - leakage through global scaling
    - repeated stimulus contamination across folds
    - confounding by running or arousal
  validation_checks:
    - cross_validated_score
    - permutation_test
    - baseline_model_comparison
```

This is the contract that FastMCP exposes to agents.

---

## 9. FastMCP layer

FastMCP should expose MouseHash as a collection of bounded MCP tools, resources, and prompts.

FastMCP should not be the architecture itself. It is the protocol/server layer.

### 9.1 MCP tools

Tool groups:

```text
Target tools
  - connect/search/list/fetch target-specific data

Manifest tools
  - build role manifests
  - inspect role evidence
  - compare manifests

Readiness tools
  - list runnable tools for a dataset
  - explain missing roles
  - suggest transformations needed for a tool

Transformation tools
  - create transformation plans
  - materialize analysis views
  - validate view lineage

Analysis tools
  - run bounded statistical/scientific tools
  - run validation controls
  - compare artifacts

Artifact tools
  - list artifacts
  - inspect artifact metadata
  - load artifact summary
  - generate reports

Planning tools
  - parse research question into required roles
  - match datasets to hypotheses
  - propose AnalysisMoves
```

### 9.2 Example FastMCP server skeleton

```python
from fastmcp import FastMCP

mcp = FastMCP("MouseHash")

@mcp.tool
def dandi_build_role_manifest(dandiset_id: str, asset_strategy: str = "representative") -> dict:
    """Inspect a DANDI dandiset and return an evidence-backed MouseHash role manifest."""
    ...

@mcp.tool
def allen_build_role_manifest(session_id: int) -> dict:
    """Build a MouseHash role manifest for an Allen Brain Observatory session."""
    ...

@mcp.tool
def ibl_build_role_manifest(eid: str) -> dict:
    """Build a MouseHash role manifest for an IBL session."""
    ...

@mcp.tool
def list_runnable_tools(manifest_id: str) -> dict:
    """Return MouseHash tools whose role signatures are satisfied by a manifest."""
    ...

@mcp.tool
def propose_analysis_moves(manifest_id: str, scientific_goal: str) -> dict:
    """Suggest valid AnalysisMoves from available roles, transformations, and tools."""
    ...

@mcp.tool
def materialize_analysis_view(manifest_id: str, transformation_plan: dict) -> dict:
    """Run explicit transformations and save an AnalysisView artifact."""
    ...

@mcp.tool
def run_tool(tool_name: str, view_ids: list[str], spec: dict) -> dict:
    """Run a bounded MouseHash tool on existing AnalysisView objects."""
    ...

@mcp.tool
def generate_structure_discovery_report(artifact_ids: list[str]) -> dict:
    """Generate an auditable report from existing artifacts and provenance."""
    ...
```

### 9.3 MCP resources

Resources expose read-only state.

Examples:

```text
mousehash://targets
mousehash://targets/allen/sessions/{session_id}/manifest
mousehash://targets/ibl/sessions/{eid}/manifest
mousehash://targets/dandi/dandisets/{dandiset_id}/manifest
mousehash://manifests/{manifest_id}
mousehash://views/{view_id}/summary
mousehash://tools/{tool_name}/contract
mousehash://artifacts/{artifact_id}/summary
mousehash://reports/{report_id}
```

Resources should be safe to read. Tools can mutate or compute.

### 9.4 MCP prompts

Prompts are useful for repeated scientific interactions.

Examples:

```text
prompt: "explain_dataset_readiness"
prompt: "design_analysis_plan"
prompt: "debug_failed_analysis_move"
prompt: "write_structure_discovery_summary"
prompt: "compare_datasets_for_hypothesis"
```

Example prompt:

```text
Given a MouseHash RoleManifest and a scientific question, identify:
1. which roles are available,
2. which required roles are missing,
3. which transformations are needed,
4. which tools are valid,
5. which validation controls should accompany the analysis.
```

---

## 10. Target-specific MCP functions

The key design decision: MouseHash exposes both **target-specific access tools** and **shared analysis tools**.

### 10.1 Allen MCP tools

```text
allen_search_sessions
allen_get_session_metadata
allen_list_stimuli
allen_list_brain_regions
allen_load_natural_scenes
allen_load_stimulus_presentations
allen_load_calcium_traces
allen_load_neuropixels_units
allen_build_role_manifest
allen_materialize_role_bundle
allen_create_natural_scene_response_view
```

### 10.2 IBL MCP tools

```text
ibl_search_sessions
ibl_get_session_metadata
ibl_list_probes
ibl_load_trials
ibl_load_spikes
ibl_load_clusters
ibl_load_wheel
ibl_load_behavior
ibl_build_role_manifest
ibl_materialize_role_bundle
ibl_create_choice_aligned_spike_view
```

### 10.3 DANDI MCP tools

```text
dandi_search_dandisets
dandi_get_dandiset_metadata
dandi_list_assets
dandi_inspect_asset
dandi_inspect_nwb_structure
dandi_build_role_manifest
dandi_materialize_role_bundle
dandi_create_nwb_trial_response_view
```

These are allowed to be target-specific because target data access is target-specific.

But after a standard `AnalysisView` exists, use shared tools.

---

## 11. Shared MCP functions

### 11.1 Manifest and readiness

```text
get_role_manifest
compare_role_manifests
list_available_roles
list_runnable_tools
explain_tool_readiness
explain_missing_roles
suggest_derivable_roles
```

Example:

```text
User: Can I run RSA on this DANDI dataset?

MCP:
  get_role_manifest(dandiset_id)
  explain_tool_readiness(tool_name="rsa", manifest_id)

Returns:
  runnable: true/false
  required_roles:
    neural_data: present
    stimuli_or_conditions_or_behavior: present
  required_views:
    neural_rdm
    target_rdm
  needed_transformations:
    make_response_matrix
    aggregate_by_condition
    compute_rdm
    permutation_test
```

### 11.2 Transformation planning

```text
suggest_transformation_plan
validate_transformation_plan
materialize_analysis_view
list_views_for_manifest
inspect_view_lineage
```

### 11.3 Shared scientific tools

```text
run_pca
run_nmf
fit_ridge_encoding_model
fit_logistic_decoder
compute_rdm
run_rsa
compute_cka
compute_noise_correlations
compute_functional_connectivity
run_permutation_test
run_bootstrap_ci
generate_raster_plot
generate_psth_plot
generate_structure_discovery_report
```

These should accept view IDs, not raw data handles.

```python
@mcp.tool
def run_pca(view_id: str, n_components: int, spec: dict | None = None) -> dict:
    ...
```

---

## 12. DataJoint/provenance schema

Even with FastMCP, MouseHash still needs a structured internal memory.

Recommended tables:

```text
Target
Dataset
DatasetVersion
Resource
RoleManifest
RoleEvidence

TransformationSpec
TransformationRun
AnalysisView

ToolSpec
ToolRun
ValidationRun

Artifact
ArtifactFile
Report
```

### 12.1 Minimal schema concepts

```python
Target:
  target_name
  adapter_version
  config_hash

Dataset:
  target_name
  dataset_id
  dataset_version
  metadata_hash

RoleManifest:
  manifest_id
  dataset_id
  created_at
  parser_version
  manifest_json_path
  summary_json

RoleEvidence:
  manifest_id
  role_name
  status
  confidence
  source
  source_path
  notes

TransformationSpec:
  transformation_name
  spec_hash
  spec_json

TransformationRun:
  transformation_run_id
  manifest_id
  input_artifact_ids
  spec_hash
  code_version
  status
  output_view_id

AnalysisView:
  view_id
  view_kind
  shape_json
  axes_json
  artifact_path
  lineage_hash

ToolRun:
  tool_run_id
  tool_name
  view_ids
  spec_hash
  code_version
  status

Artifact:
  artifact_id
  artifact_kind
  tool_run_id
  path
  summary_json
  hash
```

FastMCP calls functions; DataJoint remembers what those functions did.

---

## 13. Workflow examples

## 13.1 DANDI role scan to runnable tools

```text
User:
  "What analyses can MouseHash run on DANDI dataset 000XXX?"

FastMCP:
  dandi_get_dandiset_metadata
  dandi_list_assets
  dandi_inspect_nwb_structure
  dandi_build_role_manifest
  list_runnable_tools
  explain_tool_readiness

MouseHash result:
  - PSTH: runnable
  - raster: runnable
  - trial-averaged PCA: runnable
  - RSA: runnable if stimulus/condition labels exist
  - ridge encoding: needs stimulus features or behavior covariates
  - noise correlations: runnable if repeated conditions exist
```

## 13.2 Allen natural scenes semantic analysis

```text
User:
  "Run the MouseHash v0 semantic workflow on Allen natural scenes."

FastMCP:
  allen_load_natural_scenes
  allen_build_role_manifest
  materialize_analysis_view:
    - cache images
    - extract ViT logits/probabilities
    - assign animate/inanimate labels
  run_pca
  run_nmf
  generate_structure_discovery_report
```

Artifacts:

```text
stimulus image cache
ViT logits
softmax probabilities
top-1 labels
animate/inanimate vector
PCA scores/loadings
NMF components/weights
HTML report
```

## 13.3 IBL choice decoding

```text
User:
  "Find IBL sessions where we can decode choice from neural activity."

FastMCP:
  ibl_search_sessions
  ibl_build_role_manifest for candidates
  list_runnable_tools(tool_family="decoding")
  suggest_transformation_plan(goal="choice decoding")
```

Transformation path:

```text
load spikes
load trials
select good units
align spikes to go cue or stimulus onset
bin spikes into trial window
create trial x neuron matrix
assign choice labels
split train/test by trial block
z-score train-only
fit logistic decoder
run permutation test
generate report
```

Shared tools used:

```text
fit_logistic_decoder
run_cross_validated_decoding_evaluation
run_permutation_test
generate_structure_discovery_report
```

Only the loading and manifest construction are IBL-specific.

---

## 14. Tool registry design

The tool registry should be declarative.

Example `configs/tools/tool_registry.yaml`:

```yaml
tools:
  fit_ridge_encoding_model:
    family: encoding
    mcp_name: run_ridge_encoding
    python_callable: mousehash.tools.encoding.ridge.fit_ridge_encoding_model
    requires_roles:
      required:
        - stimuli
        - neural_data
        - time_organization
      optional:
        - behavior
        - conditions
        - metadata
    consumes_views:
      X: observation_by_feature
      Y: observation_by_neuron
    produces:
      - model_artifact
      - metric_table
      - report_bundle
    default_validation:
      - cross_validation
      - permutation_test
      - baseline_model
    priority: mvp

  run_rsa:
    family: geometry
    mcp_name: run_rsa
    python_callable: mousehash.tools.geometry.rsa.run_rsa
    requires_roles:
      required:
        - neural_data
      any_of:
        - stimuli
        - conditions
        - behavior
      optional:
        - time_organization
        - metadata
    consumes_views:
      neural_rdm: rdm
      target_rdm: rdm
    produces:
      - metric_table
      - figure_artifact
      - report_bundle
    default_validation:
      - permutation_test
    priority: mvp
```

This lets the readiness engine operate without hardcoding every rule in Python.

---

## 15. AnalysisMove registry

Analysis moves combine roles, transformations, views, tools, and validation.

Example `configs/tools/analysis_moves.yaml`:

```yaml
analysis_moves:
  stimulus_aligned_ridge_encoding:
    target_scope:
      - allen
      - dandi
    required_roles:
      - neural_data
      - stimuli
      - time_organization
    transformation_plan:
      - quality_control.select_good_units
      - synchronization.map_stimulus_frames_to_neural_clock
      - alignment.align_to_stimulus_onset
      - segmentation.extract_response_window
      - aggregation.mean_response_in_window
      - feature_extraction.extract_vit_embeddings
      - tensorization.make_design_matrix
      - splitting.k_fold_by_stimulus
      - normalization.zscore_train_only
    views:
      X: observation_by_feature
      Y: observation_by_neuron
    tool:
      name: fit_ridge_encoding_model
    validation:
      - permutation_test
      - baseline_model_comparison
    artifacts:
      - model_weights
      - cv_scores
      - prediction_plots
      - structure_discovery_report

  choice_decoder:
    target_scope:
      - ibl
      - dandi
    required_roles:
      - neural_data
      - behavior
      - time_organization
    transformation_plan:
      - quality_control.select_good_units
      - alignment.align_to_choice_or_stimulus
      - binning.bin_spikes
      - tensorization.make_trial_by_neuron_matrix
      - labeling.assign_choice_labels
      - splitting.time_blocked_train_test_split
      - normalization.zscore_train_only
    views:
      X: observation_by_neuron
      y: condition_or_behavior_label
    tool:
      name: fit_logistic_decoder
    validation:
      - cross_validated_decoding_evaluation
      - permutation_test
    artifacts:
      - decoder_model
      - confusion_matrix
      - decoding_report
```

This is where MouseHash becomes a compiler.

---

## 16. FastMCP deployment shapes

### 16.1 Local researcher mode

```text
Claude Desktop / Cursor / local agent
        |
        v
MouseHash FastMCP over stdio
        |
        v
local data cache + DataJoint + artifacts
```

Best for:

- vibe coding
- local DANDI/NWB inspection
- Allen/IBL experiments
- demos to collaborators

### 16.2 Lab server mode

```text
Lab users / agents
        |
        v
MouseHash FastMCP over HTTP
        |
        v
shared lab server / HPC / NAS
        |
        v
DataJoint + artifact store
```

Best for:

- one lab’s data
- private datasets
- shared provenance
- heavier analyses

### 16.3 Public discovery mode

```text
Web UI / chatbot / agent
        |
        v
MouseHash FastMCP API
        |
        v
indexed DANDI/Allen/IBL manifests
        |
        v
readiness + recommendation engine
```

Best for:

- dataset search
- “what analysis can I run?”
- “find data for my hypothesis”
- demoing MouseHash as an agentic neuroscience discovery layer

---

## 17. Recommended MVP sequence

### Phase 1: Library spine

Build:

```text
core/
targets/base.py
role_bundle.py
manifests.py
contracts.py
analysis_view.py
artifacts/
schema/
```

Goal:

```text
Can represent a dataset, its roles, a view, a tool contract, and an artifact.
```

### Phase 2: DANDI scanner

Build:

```text
targets/dandi/client.py
targets/dandi/nwb_inspector.py
targets/dandi/manifest.py
mcp/tools/target_tools.py
```

Goal:

```text
Given a dandiset_id, return a role manifest and runnable tool list.
```

This is a strong MVP because DANDI has heterogeneous NWB files and demonstrates the parser/compiler idea.

### Phase 3: Allen v0 workflow

Build:

```text
targets/allen/
feature_extraction.extract_vit_embeddings
tools.factor_models.pca
tools.factor_models.nmf
reports.structure_discovery
```

Goal:

```text
Rebuild the Allen natural scenes semantic workflow behind FastMCP.
```

### Phase 4: Shared transformations

Build the top transformations:

```text
select_good_units
align_to_stimulus_onset
extract_response_window
mean_response_in_window
bin_spikes
make_design_matrix
make_trial_by_neuron_matrix
zscore_train_only
k_fold_split
compute_rdm
```

Goal:

```text
Tools consume views, not raw data.
```

### Phase 5: First real shared scientific tools

Build:

```text
run_pca
run_nmf
fit_ridge_encoding_model
fit_logistic_decoder
compute_rdm
run_rsa
run_permutation_test
generate_structure_discovery_report
```

Goal:

```text
Same tools run on views from Allen, IBL, or DANDI.
```

### Phase 6: IBL choice-decoding vertical slice

Build:

```text
ibl_search_sessions
ibl_load_trials
ibl_load_spikes
ibl_build_role_manifest
ibl_create_choice_aligned_spike_view
fit_logistic_decoder
```

Goal:

```text
Show target-specific ingestion + shared decoding stack.
```

---

## 18. Testing strategy

MouseHash needs ruthless tests because agents will happily call nonsense if the contracts are squishy.

### 18.1 Unit tests

```text
test_role_manifest_schema.py
test_role_evidence_merge.py
test_tool_signature_matching.py
test_transformation_spec_hashing.py
test_analysis_view_lineage.py
```

### 18.2 Golden manifest fixtures

Create fixtures:

```text
fixtures/manifests/
  allen_natural_scenes_manifest.json
  ibl_choice_session_manifest.json
  dandi_spikes_trials_manifest.json
  dandi_calcium_no_trials_manifest.json
```

Test expected readiness:

```text
Allen natural scenes:
  PCA: yes
  NMF: yes
  ridge encoding: yes after neural responses exist
  choice decoder: no

IBL choice task:
  logistic decoder: yes
  PSTH: yes
  RSA: maybe
  ViT encoding: no unless stimuli/features exist

DANDI spikes + trials:
  raster: yes
  PSTH: yes
  trial PCA: yes
  RSA: yes if conditions/stimuli exist
```

### 18.3 Contract tests

Every tool must pass:

```text
- rejects wrong view kind
- rejects missing roles
- records provenance
- writes artifact summary
- does not mutate input views
- does not do hidden transformations
```

### 18.4 MCP tests

Test every MCP tool with JSON inputs/outputs.

```text
test_mcp_dandi_build_role_manifest.py
test_mcp_list_runnable_tools.py
test_mcp_materialize_analysis_view.py
test_mcp_run_pca.py
```

MCP responses should be boring, structured, and stable. Boring is beautiful here.

---

## 19. Security and safety boundaries

Because MouseHash can run code over local data, the FastMCP server should be conservative.

Rules:

```text
- no arbitrary Python execution through MCP
- no arbitrary filesystem paths unless whitelisted
- no deleting artifacts through agent tools at first
- no overwriting artifacts without versioning
- no silent remote downloads without explicit target config
- no credentials in prompts or reports
- all target access configured through config files or environment variables
```

For local mode, allow:

```text
MOUSEHASH_DATA_DIR
MOUSEHASH_ARTIFACT_DIR
MOUSEHASH_DJ_CONFIG
MOUSEHASH_ALLOWED_PATHS
```

For server mode, add:

```text
auth
rate limits
audit logs
job queue
resource quotas
```

---

## 20. What this architecture gives MouseHash

This architecture gives you three important superpowers.

### 20.1 One agent interface over many neuroscience ecosystems

Allen, IBL, and DANDI stay different at the edges, but become comparable inside MouseHash.

```text
Allen session -> RoleBundle -> AnalysisView -> Tool -> Artifact
IBL session   -> RoleBundle -> AnalysisView -> Tool -> Artifact
DANDI NWB     -> RoleBundle -> AnalysisView -> Tool -> Artifact
```

### 20.2 A real compiler for neuroscience analysis

MouseHash can compile:

```text
"I want to test whether neural geometry tracks stimulus geometry"
```

into:

```text
required roles:
  neural_data + stimuli/conditions

transformations:
  response matrix
  aggregation
  normalization
  neural RDM
  stimulus/model RDM

tools:
  RSA
  permutation test
  report
```

### 20.3 A marketplace-compatible tool system

Each tool can be packaged with:

```text
- Python implementation
- role signature
- required view contract
- allowed transformations
- validation checks
- artifact schema
- FastMCP wrapper
- tests
```

That is very close to a Hugging Face-style ecosystem, but for neuroscience analysis moves.

---

## 21. The simplest useful version

The smallest serious version of this architecture is:

```text
src/mousehash/
  targets/
    dandi/
    allen/
  core/
    role_bundle.py
    contracts.py
    analysis_view.py
  transformations/
    tensorization.py
    normalization.py
    splitting.py
    similarity.py
  tools/
    factor_models/pca.py
    factor_models/nmf.py
    encoding/ridge.py
    geometry/rdm.py
    geometry/rsa.py
    statistics/permutation.py
    reports/structure_discovery.py
  artifacts/
    io.py
    bundle.py
  mcp/
    server.py
```

And the first MCP tools:

```text
dandi_build_role_manifest
allen_build_role_manifest
list_runnable_tools
explain_tool_readiness
materialize_analysis_view
run_pca
run_nmf
compute_rdm
run_rsa
fit_ridge_encoding_model
generate_structure_discovery_report
```

That is already MouseHash.

Not the giant final cathedral. The first working goblin cathedral brick.

---

## 22. Final recommendation

Build MouseHash as:

```text
FastMCP server
  over
MouseHash core library
  over
target adapters
  over
DataJoint + artifact store
```

Do **not** make FastMCP wrappers call raw Allen/IBL/DANDI code directly forever. That would be quick at first and cursed later.

Instead:

1. target adapter loads data,
2. manifest parser maps it into roles,
3. readiness engine checks tool signatures,
4. transformation planner creates explicit views,
5. shared tools run on views,
6. artifacts and provenance are stored,
7. FastMCP exposes the whole thing as callable scientific infrastructure.

This preserves the MouseHash identity:

> **MouseHash is not a chatbot. MouseHash is a typed, reproducible, agent-callable structure-finding system for neuroscience data.**
