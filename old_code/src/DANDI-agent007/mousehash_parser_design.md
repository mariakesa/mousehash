# MouseHash Parser Design

**Purpose:** Design an evidence-backed parser that reads DANDI metadata, individual NWB files, and optional scientific papers/abstracts, then fills the MouseHash role manifest and computes which analyses are valid, blocked, or derivable.

This document is intentionally written as a design/specification document rather than runnable code. Pseudocode is used to define behavior without prescribing implementation details.

---

## 1. Core idea

The MouseHash parser is the **front-end of the MouseHash scientific compiler**.

It does not merely say:

```yaml
spikes: true
visual_stimuli: true
trials: true
```

It should say:

```yaml
mousehash_roles:
  neural_data:
    spikes:
      status: present
      confidence: 0.98
      evidence:
        - source: nwb
          path: /units/spike_times
          reason: NWB Units table contains spike_times
        - source: dandiset_metadata
          field: assetsSummary.variableMeasured
          value: Units
      contradictions: []
      needs_human_review: false
```

The parser turns messy scientific data into a typed, inspectable **EvidenceBackedRoleManifest**. That manifest is then consumed by the **tool-readiness layer**, which checks which MouseHash analyses can be responsibly suggested.

Pipeline:

```text
DANDI metadata + NWB structure + papers/abstracts
        ↓
Evidence store
        ↓
Evidence-backed MouseHash role manifest
        ↓
Derived-variable inference
        ↓
Tool-readiness layer
        ↓
Suggested transformations and valid analysis moves
```

---

## 2. Input sources

The parser can consume several evidence sources.

### 2.1 DANDI metadata

DANDI metadata provides broad, dataset-level information. It is useful but not always complete enough for analysis decisions.

Useful fields include:

```text
name
description
keywords
assetsSummary.variableMeasured
assetsSummary.measurementTechnique
assetsSummary.approach
assetsSummary.species
assetsSummary.numberOfSubjects
wasGeneratedBy
relatedResource
contributor
license
citation
```

Typical role signal:

```text
variableMeasured contains Units
    → neural_data.spikes likely_present

measurementTechnique contains current clamp technique
    → neural_data.current_clamp likely_present

approach contains optogenetic approach
    → stimuli.interventions.optogenetic likely_present
```

Metadata is good for a coarse first pass. It should not be treated as final truth.

---

### 2.2 NWB file structure

NWB is the strongest evidence source because it describes actual data objects, tables, timestamps, devices, trials, electrodes, units, imaging planes, and processing modules.

The parser should inspect a representative set of NWB assets and create an intermediate summary:

```yaml
NWBStructureSummary:
  file_path_or_asset_id:
  nwb_identifier:
  session_start_time:
  subject_fields:
  devices:
  acquisition_keys:
  stimulus_keys:
  processing_modules:
  intervals:
  trials_columns:
  units_columns:
  electrodes_columns:
  imaging_planes:
  neurodata_objects:
    - path:
      neurodata_type:
      name:
      description:
      comments:
      unit:
      rate:
      has_timestamps:
      shape:
```

The role extractor should operate over this summary, not directly over raw NWB internals.

---

### 2.3 Papers, DOIs, and abstracts

Papers and abstracts provide semantic context. They are especially useful for detecting things like task structure, delay periods, perturbation epochs, or behavioral meaning.

Examples:

```text
"mice performed a delayed-response task"
    → conditions.session_phases.task_epoch likely_present

"photoinhibition during the delay period"
    → stimuli.interventions.optogenetic likely_present
    → conditions.perturbation_labels likely_present
    → conditions.session_phases.task_epoch.delay likely_present

"two-photon calcium imaging"
    → neural_data.calcium likely_present
```

However, papers are lower-confidence evidence than NWB structural paths. A paper may describe the broader experiment while the shared DANDI asset contains only a subset of variables.

Paper evidence should therefore be capped in confidence unless confirmed by metadata or NWB structure.

---

## 3. Output object: EvidenceBackedRoleManifest

Each role should have a status, confidence, evidence list, contradiction list, and optional derivation recipe.

### 3.1 Status vocabulary

```text
present
likely_present
absent
unknown
ambiguous
derived_possible
```

Meaning:

| Status | Meaning |
|---|---|
| `present` | Strong direct evidence exists, usually from NWB paths or tables. |
| `likely_present` | Metadata, paper text, or indirect structural evidence suggests the role. |
| `absent` | The parser has enough evidence to say the role is probably not present. |
| `unknown` | Not enough evidence either way. |
| `ambiguous` | Evidence exists but maps to multiple possible roles. |
| `derived_possible` | The role is not directly stored but can likely be computed from other variables. |

Example:

```yaml
behavior:
  reaction_times:
    status: derived_possible
    confidence: 0.82
    derivation_recipe:
      name: compute_reaction_time
      formula: response_time - stimulus_onset_time
      required_fields:
        - trials.stimulus_onset_time
        - trials.response_time
    evidence:
      - source: nwb
        path: /intervals/trials/stimulus_onset_time
      - source: nwb
        path: /intervals/trials/response_time
```

---

## 4. MouseHash role taxonomy

The parser fills the following canonical role bundle.

```yaml
mousehash_roles:
  neural_data:
    - spikes
    - lfp
    - eeg
    - calcium
    - photometry
    - images
    - current_clamp
    - voltage_clamp
    - patch_clamp

  stimuli:
    sensory:
      - visual
      - auditory
      - tactile
      - odor
    interventions:
      - optogenetic
      - electrical
      - pharmacological
      - anesthesia

  behavior:
    - choices
    - reaction_times
    - pose
    - locomotion
    - pupil
    - kinematics
    - behavioral_states
    - running

  conditions:
    - task_labels
    - trial_labels
    - experimental_groups
    - brain_states
    - session_phases
    - perturbation_labels

  time_organization:
    - continuous_time
    - trials
    - epochs
    - events
    - frames
    - alignment_rules

  metadata:
    - subject
    - species
    - genotype
    - session
    - brain_area
    - probe_electrode_imaging_plane
    - acquisition_device
    - preprocessing_info
```

Recommended extension for `session_phases`:

```yaml
session_phases:
  coarse:
    - baseline
    - stimulation
    - task
    - rest
    - sleep
    - wake
  task_epoch:
    - sample
    - delay
    - response
    - go_cue
    - reward
```

This distinction matters because `session_phases` can mean either broad recording states or task-epoch structure.

---

## 5. Main parser pseudocode

```text
FUNCTION parse_mousehash_roles(dandiset_id):

    evidence_store = empty EvidenceStore
    manifest = initialize_empty_role_manifest()

    dandiset_metadata = fetch_dandiset_metadata(dandiset_id)
    evidence_store.add(
        parse_dandiset_metadata(dandiset_metadata)
    )

    asset_list = fetch_asset_list(dandiset_id)

    representative_assets = select_representative_nwb_assets(asset_list)

    FOR each asset IN representative_assets:

        asset_metadata = fetch_asset_metadata(asset)
        evidence_store.add(
            parse_asset_metadata(asset_metadata)
        )

        nwb_summary = inspect_nwb_structure(asset)
        evidence_store.add(
            parse_nwb_structure(nwb_summary)
        )

    related_resources = extract_related_resources(dandiset_metadata)
    paper_records = resolve_publications(related_resources)

    FOR each paper_record IN paper_records:
        abstract = fetch_abstract_if_available(paper_record)

        IF abstract exists:
            evidence_store.add(
                parse_paper_abstract_with_llm(abstract)
            )

    evidence_store.add(
        infer_derived_roles_from_combinations(evidence_store)
    )

    manifest = merge_evidence_into_manifest(evidence_store)

    manifest = run_consistency_checks(manifest, evidence_store)

    manifest = attach_transformation_hints(manifest)

    manifest = attach_tool_readiness_hints(manifest)

    RETURN manifest
```

---

## 6. Evidence source priority

The parser should rank evidence sources by reliability.

```text
Highest confidence:
  NWB structural paths
  NWB neurodata types
  NWB table columns
  NWB processing module names
  NWB acquisition/stimulus/interval objects

Medium confidence:
  DANDI asset metadata
  DANDI assetsSummary
  Dandiset description
  Dandiset keywords
  Related resource titles

Lower confidence:
  Paper abstract language
  LLM inference from paper text
  Fuzzy string matches
  Filename hints
```

Suggested confidence caps:

```yaml
confidence_caps:
  nwb_direct_path: 0.99
  nwb_neurodata_type: 0.97
  nwb_table_column: 0.95
  dandi_assets_summary: 0.85
  dandi_description: 0.75
  paper_abstract_llm: 0.75
  filename_hint: 0.45
```

---

## 7. DANDI metadata parser

```text
FUNCTION parse_dandiset_metadata(metadata):

    evidence = []

    text_blob = concatenate_text_fields(
        metadata.name,
        metadata.description,
        metadata.keywords,
        metadata.assetsSummary.variableMeasured,
        metadata.assetsSummary.measurementTechnique,
        metadata.assetsSummary.approach
    )

    FOR each role IN mousehash_role_taxonomy:
        matches = rule_match_role_against_text(role, text_blob)

        FOR each match IN matches:
            evidence.append(
                Evidence(
                    role = role,
                    status = likely_present,
                    confidence = metadata_rule_confidence(match),
                    source = "dandiset_metadata",
                    field = match.field,
                    value = match.value,
                    reason = match.reason
                )
            )

    evidence += parse_species_subject_metadata(metadata)
    evidence += parse_related_resources(metadata)

    RETURN evidence
```

### 7.1 Metadata role rules

```text
IF variableMeasured contains "Units":
    neural_data.spikes = likely_present

IF variableMeasured contains "ElectricalSeries":
    neural_data.lfp OR neural_data.eeg OR raw_ephys = likely_present
    ambiguity = "Need NWB path/sampling rate/electrode metadata to distinguish LFP vs raw extracellular signal"

IF variableMeasured contains "LFP":
    neural_data.lfp = likely_present

IF variableMeasured contains "TwoPhotonSeries":
    neural_data.calcium = likely_present

IF variableMeasured contains "OnePhotonSeries":
    neural_data.calcium = likely_present

IF variableMeasured contains "RoiResponseSeries" OR "DfOverF" OR "Fluorescence":
    neural_data.calcium = likely_present

IF variableMeasured contains "PatchClampSeries":
    neural_data.patch_clamp = likely_present

IF measurementTechnique contains "current clamp":
    neural_data.current_clamp = likely_present
    neural_data.patch_clamp = likely_present

IF measurementTechnique contains "voltage clamp":
    neural_data.voltage_clamp = likely_present
    neural_data.patch_clamp = likely_present

IF approach contains "optogenetic":
    stimuli.interventions.optogenetic = likely_present

IF approach contains "behavioral":
    behavior = likely_present_but_unspecified

IF species exists:
    metadata.species = present
```

---

## 8. NWB structure parser

```text
FUNCTION parse_nwb_structure(nwb_summary):

    evidence = []

    evidence += detect_neural_data_roles(nwb_summary)
    evidence += detect_stimuli_roles(nwb_summary)
    evidence += detect_behavior_roles(nwb_summary)
    evidence += detect_condition_roles(nwb_summary)
    evidence += detect_time_organization_roles(nwb_summary)
    evidence += detect_metadata_roles(nwb_summary)

    RETURN evidence
```

---

## 9. Neural data role detection

```text
FUNCTION detect_neural_data_roles(summary):

    evidence = []

    IF "/units" exists AND "spike_times" in units_columns:
        evidence.add(present("neural_data.spikes", source_path="/units/spike_times"))

    IF any NeurodataType == "ElectricalSeries":
        evidence.add(likely_present("neural_data.lfp_or_raw_ephys"))

        IF path_or_description_contains(["lfp", "local field", "LFP"]):
            evidence.add(present("neural_data.lfp"))

        ELSE IF object under "/processing/ecephys/LFP":
            evidence.add(present("neural_data.lfp"))

        ELSE:
            evidence.add(ambiguous("neural_data.lfp", reason="ElectricalSeries present but not clearly LFP"))

    IF any NeurodataType in ["LFP", "DecompositionSeries", "Spectrum"]:
        evidence.add(present("neural_data.lfp"))

    IF any object/path/description contains ["EEG", "ECoG"]:
        evidence.add(present("neural_data.eeg"))

    IF any NeurodataType in ["TwoPhotonSeries", "OnePhotonSeries", "RoiResponseSeries", "DfOverF", "Fluorescence"]:
        evidence.add(present("neural_data.calcium"))

    IF any NeurodataType contains "FiberPhotometry" OR path/description contains ["photometry", "fiber"]:
        evidence.add(present("neural_data.photometry"))

    IF any NeurodataType in ["ImageSeries", "Images", "ImageSegmentation"]:
        evidence.add(present("neural_data.images"))

    IF any NeurodataType in ["CurrentClampSeries", "CurrentClampStimulusSeries"]:
        evidence.add(present("neural_data.current_clamp"))
        evidence.add(present("neural_data.patch_clamp"))

    IF any NeurodataType in ["VoltageClampSeries", "VoltageClampStimulusSeries"]:
        evidence.add(present("neural_data.voltage_clamp"))
        evidence.add(present("neural_data.patch_clamp"))

    IF electrodes table exists:
        evidence.add(present("metadata.probe_electrode_imaging_plane"))

    IF imaging planes exist:
        evidence.add(present("metadata.probe_electrode_imaging_plane"))

    RETURN evidence
```

---

## 10. Stimulus role detection

```text
FUNCTION detect_stimuli_roles(summary):

    evidence = []

    stimulus_text = text_from(
        stimulus_keys,
        trials_columns,
        intervals,
        time_series_descriptions,
        processing_module_names
    )

    IF stimulus group exists:
        evidence.add(present("stimuli"))

    IF stimulus_text contains ["visual", "movie", "grating", "drifting", "natural scene", "image", "flash", "bar", "screen"]:
        evidence.add(present("stimuli.sensory.visual"))

    IF stimulus_text contains ["auditory", "sound", "tone", "frequency", "click", "white noise"]:
        evidence.add(present("stimuli.sensory.auditory"))

    IF stimulus_text contains ["whisker", "touch", "tactile", "pole", "vibration", "air puff"]:
        evidence.add(present("stimuli.sensory.tactile"))

    IF stimulus_text contains ["odor", "olfactory", "valve", "go odor", "odorant"]:
        evidence.add(present("stimuli.sensory.odor"))

    IF stimulus_text contains ["opto", "laser", "photostim", "photoinhibition", "LED", "ChR2", "Arch", "NpHR"]:
        evidence.add(present("stimuli.interventions.optogenetic"))

    IF stimulus_text contains ["electrical stimulation", "stim electrode", "microstimulation"]:
        evidence.add(present("stimuli.interventions.electrical"))

    IF stimulus_text contains ["drug", "pharmacology", "injection", "saline", "muscimol", "ketamine", "APV", "CNQX"]:
        evidence.add(present("stimuli.interventions.pharmacological"))

    IF stimulus_text contains ["anesthesia", "isoflurane", "urethane", "ketamine/xylazine"]:
        evidence.add(present("stimuli.interventions.anesthesia"))

    RETURN evidence
```

---

## 11. Behavior role detection

```text
FUNCTION detect_behavior_roles(summary):

    evidence = []

    behavior_text = text_from(
        processing_modules,
        acquisition_keys,
        trials_columns,
        intervals,
        time_series_names,
        descriptions
    )

    IF trials_columns contains any ["choice", "response", "lick_side", "correct", "incorrect"]:
        evidence.add(present("behavior.choices"))

    IF trials_columns contains both event_start and response_time:
        evidence.add(derived_possible("behavior.reaction_times"))

    IF trials_columns contains ["reaction_time", "rt", "response_latency"]:
        evidence.add(present("behavior.reaction_times"))

    IF behavior_text contains ["pose", "DeepLabCut", "keypoints", "skeleton", "body part"]:
        evidence.add(present("behavior.pose"))
        evidence.add(present("behavior.kinematics"))

    IF behavior_text contains ["running", "running_speed", "wheel", "velocity", "locomotion"]:
        evidence.add(present("behavior.running"))
        evidence.add(present("behavior.locomotion"))

    IF behavior_text contains ["pupil", "eye tracking", "pupil_area", "pupil diameter"]:
        evidence.add(present("behavior.pupil"))

    IF behavior_text contains ["kinematic", "velocity", "acceleration", "paw", "whisker angle", "jaw", "tongue"]:
        evidence.add(present("behavior.kinematics"))

    IF behavior_text contains ["sleep", "wake", "arousal", "engaged", "quiescent", "behavioral state"]:
        evidence.add(present("behavior.behavioral_states"))

    RETURN evidence
```

---

## 12. Condition role detection

```text
FUNCTION detect_condition_roles(summary):

    evidence = []

    trial_cols = summary.trials_columns
    interval_names = summary.intervals
    text = text_from(trial_cols, interval_names, descriptions)

    IF trials table exists:
        evidence.add(present("time_organization.trials"))

    IF trial_cols contains ["stimulus", "stimulus_name", "image_id", "orientation", "frequency", "contrast"]:
        evidence.add(present("conditions.trial_labels"))
        evidence.add(present("conditions.task_labels OR conditions.stimulus_labels"))

    IF trial_cols contains ["choice", "response", "correct", "outcome", "rewarded"]:
        evidence.add(present("conditions.trial_labels"))

    IF text contains ["group", "genotype", "control", "treatment", "condition"]:
        evidence.add(likely_present("conditions.experimental_groups"))

    IF text contains ["awake", "sleep", "anesthetized", "REM", "NREM", "arousal"]:
        evidence.add(present("conditions.brain_states"))

    IF trial_cols or interval_names contain ["sample", "delay", "response", "go cue", "baseline"]:
        evidence.add(present("conditions.session_phases"))

    IF text contains ["perturbation", "opto", "laser", "drug", "stimulation", "inhibition"]:
        evidence.add(present("conditions.perturbation_labels"))

    RETURN evidence
```

---

## 13. Time organization detection

```text
FUNCTION detect_time_organization_roles(summary):

    evidence = []

    IF any TimeSeries has rate OR timestamps:
        evidence.add(present("time_organization.continuous_time"))

    IF trials table exists:
        evidence.add(present("time_organization.trials"))

    IF intervals table exists beyond trials:
        evidence.add(present("time_organization.epochs"))

    IF event-like columns exist:
        evidence.add(present("time_organization.events"))

    IF ImageSeries OR TwoPhotonSeries OR OnePhotonSeries exists:
        evidence.add(present("time_organization.frames"))

    IF trials table contains columns like stimulus_onset, go_cue_time, response_time:
        evidence.add(present("time_organization.alignment_rules"))

    IF timestamps exist across neural/stimulus/behavior streams:
        evidence.add(derived_possible("time_organization.alignment_rules"))

    RETURN evidence
```

Recommended richer object:

```yaml
alignment_rules:
  status: present
  candidate_events:
    - stimulus_onset
    - choice_time
    - reward_time
    - go_cue
  available_clocks:
    - neural_time
    - stimulus_time
    - behavior_time
  sync_status:
    value: likely_synced
    evidence:
      - source: nwb
        reason: all streams use NWB timestamps
```

---

## 14. Metadata role detection

```text
FUNCTION detect_metadata_roles(summary, dandiset_metadata):

    evidence = []

    IF nwb.subject exists:
        evidence.add(present("metadata.subject"))

    IF nwb.subject.species exists OR dandiset species exists:
        evidence.add(present("metadata.species"))

    IF nwb.subject.genotype exists:
        evidence.add(present("metadata.genotype"))

    IF nwb.session_id OR session_start_time exists:
        evidence.add(present("metadata.session"))

    IF electrodes.location OR imaging_plane.location exists:
        evidence.add(present("metadata.brain_area"))

    IF electrodes table OR electrode groups OR imaging planes exist:
        evidence.add(present("metadata.probe_electrode_imaging_plane"))

    IF devices exist:
        evidence.add(present("metadata.acquisition_device"))

    IF processing modules exist:
        evidence.add(likely_present("metadata.preprocessing_info"))

    RETURN evidence
```

---

## 15. DOI and paper enrichment

DANDI metadata may contain related resources. When DOIs or publication links exist, the parser can enrich the manifest using paper metadata and abstracts.

```text
FUNCTION resolve_publications(dandiset_metadata):

    related_resources = metadata.relatedResource

    paper_records = []

    FOR each resource IN related_resources:

        IF resource has identifier that looks like DOI:
            paper_records.add(
                PaperRecord(
                    doi = resource.identifier,
                    relation = resource.relation,
                    source = "dandi.relatedResource"
                )
            )

        ELSE IF resource.url contains "doi.org":
            doi = extract_doi_from_url(resource.url)
            paper_records.add(PaperRecord(doi=doi))

        ELSE IF resource.url contains "biorxiv.org":
            doi = extract_biorxiv_doi_if_possible(resource.url)
            paper_records.add(PaperRecord(doi=doi, server="biorxiv"))

        ELSE:
            paper_records.add(
                PaperRecord(
                    title_or_url = resource.name or resource.url,
                    needs_resolution = true
                )
            )

    FOR each paper_record:

        IF paper_record.doi starts with "10.1101":
            paper_record.abstract = fetch_biorxiv_details_by_doi(paper_record.doi)

        ELSE:
            paper_record.crossref_metadata = fetch_crossref_work_by_doi(paper_record.doi)

            IF crossref_metadata indicates preprint relation:
                paper_record.preprint = fetch_biorxiv_or_medrxiv_if_possible()

    RETURN paper_records
```

Rules:

```text
IF DOI exists:
    enrich from DOI

ELSE IF title exists:
    try title-based resolution

ELSE:
    skip literature enrichment
```

The parser must not depend on papers. Paper parsing is an enrichment layer, not a hard dependency.

---

## 16. LLM abstract parser

```text
FUNCTION parse_paper_abstract_with_llm(abstract):

    prompt = """
    Extract MouseHash roles from this neuroscience abstract.

    Return structured claims with:
      - role_path
      - status
      - confidence
      - quoted_evidence
      - reasoning
      - whether this claim describes recorded data, experimental design, analysis, or biological interpretation

    Do not infer that data are present unless the abstract says they were recorded,
    measured, stimulated, tracked, imaged, or analyzed.
    """

    llm_output = call_llm(prompt, abstract)

    evidence = convert_llm_output_to_evidence(llm_output)

    FOR each evidence_item:
        evidence_item.source = "paper_abstract"
        evidence_item.confidence = min(evidence_item.confidence, 0.75)

    RETURN evidence
```

---

## 17. LLM label normalizer

The LLM is especially useful for messy NWB field names and lab-specific conventions.

```text
FUNCTION normalize_labels_with_llm(raw_names):

    prompt = """
    Map these NWB/DANDI field names to MouseHash ontology paths.

    Return:
      - raw_name
      - candidate_role_paths
      - confidence
      - explanation
      - needs_human_review

    Field names:
      ...
    """

    candidates = call_llm(prompt)

    RETURN candidates
```

Examples:

```text
"lick_flag"
    → behavior.choices OR behavior.licking

"pole_in"
    → stimuli.sensory.tactile
    → time_organization.events

"photostim_on"
    → stimuli.interventions.optogenetic
    → conditions.perturbation_labels

"delay_period"
    → conditions.session_phases.task_epoch.delay

"wheel_speed"
    → behavior.running
```

Acceptance rules:

```text
IF llm_confidence high AND structural evidence exists:
    accept mapping

IF llm_confidence high BUT structural evidence weak:
    mark likely_present

IF llm_confidence medium:
    mark ambiguous

IF role unlocks expensive downstream analysis:
    request human confirmation
```

---

## 18. Derived role inference

```text
FUNCTION infer_derived_roles_from_combinations(evidence_store):

    derived_evidence = []

    IF trials has stimulus_onset AND trials has response_time:
        derived_evidence.add(
            derived_possible("behavior.reaction_times",
                recipe="response_time - stimulus_onset")
        )

    IF wheel_position exists AND timestamps exist:
        derived_evidence.add(
            derived_possible("behavior.running",
                recipe="differentiate wheel_position over time")
        )

    IF pose keypoints exist AND timestamps exist:
        derived_evidence.add(
            derived_possible("behavior.kinematics",
                recipe="differentiate pose trajectories")
        )

    IF events exist AND continuous neural time exists:
        derived_evidence.add(
            derived_possible("time_organization.alignment_rules",
                recipe="align neural data to event timestamps")
        )

    IF optogenetic stimulus exists AND trials table exists:
        derived_evidence.add(
            derived_possible("conditions.perturbation_labels",
                recipe="label trials by stimulation overlap")
        )

    RETURN derived_evidence
```

---

## 19. Evidence merger

```text
FUNCTION merge_evidence_into_manifest(evidence_store):

    manifest = initialize_all_roles_unknown()

    FOR each role_path IN taxonomy:

        role_evidence = evidence_store.get(role_path)

        IF any high_confidence_present_evidence(role_evidence):
            manifest[role_path].status = present

        ELSE IF multiple medium_confidence_sources_agree(role_evidence):
            manifest[role_path].status = likely_present

        ELSE IF only paper_or_metadata_suggests(role_evidence):
            manifest[role_path].status = likely_present
            manifest[role_path].needs_confirmation = true

        ELSE IF contradictory_evidence(role_evidence):
            manifest[role_path].status = ambiguous
            manifest[role_path].contradictions = collect_contradictions(role_evidence)

        ELSE IF role_can_be_derived_from_other_roles(role_path, manifest):
            manifest[role_path].status = derived_possible

        ELSE:
            manifest[role_path].status = unknown

        manifest[role_path].confidence = compute_confidence(role_evidence)
        manifest[role_path].evidence = top_k_evidence(role_evidence)
        manifest[role_path].source_coverage = summarize_sources(role_evidence)

    RETURN manifest
```

Example:

```text
metadata says: electrophysiology
NWB has: ElectricalSeries
NWB does not have: Units table

Result:
  neural_data.spikes = unknown or absent
  neural_data.lfp_or_raw_ephys = likely_present
  note = extracellular ephys exists, but spike sorting output was not found
```

---

## 20. Consistency checks

```text
FUNCTION run_consistency_checks(manifest, evidence_store):

    IF neural_data.spikes present AND time_organization.continuous_time unknown:
        add_warning("Spike times imply continuous time; check timestamp extraction.")

    IF calcium present AND time_organization.frames unknown:
        add_warning("Calcium imaging usually implies frames; inspect imaging objects.")

    IF behavior.reaction_times present BUT trials absent:
        add_warning("Reaction times without trial structure may not be interpretable.")

    IF stimuli.interventions.optogenetic present AND perturbation_labels unknown:
        add_hint("Perturbation labels may be derivable by intersecting stimulation times with trials.")

    IF conditions.trial_labels present AND time_organization.trials unknown:
        add_warning("Trial labels imply trials; inspect intervals/trials table.")

    RETURN manifest_with_warnings
```

---

## 21. Tool-readiness layer

The parser should compute which MouseHash tools are ready, blocked, or need transformation.

```text
FUNCTION compute_tool_readiness(manifest, tool_signature_library):

    readiness = []

    FOR each tool_signature IN tool_signature_library:

        required = tool_signature.required_roles
        optional = tool_signature.optional_roles

        satisfied = []
        missing = []
        uncertain = []
        derivable = []

        FOR each role IN required:
            status = manifest.get(role).status

            IF status == present:
                satisfied.add(role)

            ELSE IF status == likely_present:
                uncertain.add(role)

            ELSE IF status == derived_possible:
                derivable.add(role)

            ELSE:
                missing.add(role)

        IF missing is empty AND uncertain is empty AND derivable is empty:
            tool_status = ready

        ELSE IF missing is empty AND derivable not empty:
            tool_status = ready_after_transformation

        ELSE IF missing is empty AND uncertain not empty:
            tool_status = needs_confirmation

        ELSE:
            tool_status = blocked

        readiness.add(
            ToolReadiness(
                tool = tool_signature.name,
                status = tool_status,
                satisfied_roles = satisfied,
                uncertain_roles = uncertain,
                derivable_roles = derivable,
                missing_roles = missing,
                suggested_transformations = suggest_views(tool_signature, manifest)
            )
        )

    RETURN readiness
```

Status vocabulary:

```text
ready
ready_after_transformation
needs_confirmation
blocked
not_recommended
```

---

## 22. AnalysisMove object

A suggested analysis should not be just a tool name. It should be an **AnalysisMove**.

```yaml
AnalysisMove:
  name: stimulus_aligned_ridge_encoding
  role_signature:
    required:
      - neural_data
      - stimuli
      - time_organization
    optional:
      - behavior
      - metadata
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
  required_view:
    X: observations x stimulus_features
    Y: observations x neurons
  tool:
    name: fit_ridge_encoding_model
  validation_plan:
    - k_fold_cross_validation
    - permutation_control
    - train_only_normalization
  artifacts:
    - model_weights
    - cv_scores
    - prediction_plots
    - structure_discovery_report
```

Formula:

```text
AnalysisMove = RoleSignature
             + TransformationPlan
             + AnalysisView
             + Tool
             + ValidationPlan
             + Artifact
```

---

## 23. Example output

For a tactile delayed-response task with spikes and optogenetic perturbation:

```yaml
mousehash_roles:
  neural_data:
    spikes:
      status: present
    lfp:
      status: likely_present
    calcium:
      status: absent

  stimuli:
    sensory:
      tactile:
        status: present
    interventions:
      optogenetic:
        status: present

  behavior:
    choices:
      status: present
    reaction_times:
      status: derived_possible
    running:
      status: unknown
    pupil:
      status: unknown

  conditions:
    task_labels:
      status: present
    trial_labels:
      status: present
    perturbation_labels:
      status: present
    session_phases:
      task_epoch:
        sample:
          status: present
        delay:
          status: present
        response:
          status: present

  time_organization:
    continuous_time:
      status: present
    trials:
      status: present
    events:
      status: present
    alignment_rules:
      status: present

  metadata:
    subject:
      status: present
    species:
      status: present
    brain_area:
      status: present
    probe_electrode_imaging_plane:
      status: present
```

Tool-readiness:

```yaml
tool_readiness:
  ready:
    - generate_raster_plot
    - generate_psth_plot
    - fit_logistic_decoder
    - compute_neural_trajectories
    - compare_dynamics_across_conditions
    - compute_noise_correlations

  ready_after_transformation:
    - compute_reaction_time_decoder
    - perturbation_aligned_psth

  needs_confirmation:
    - lfp_bandpower_analysis

  blocked:
    - calcium_event_analysis
    - pupil_regression
```

---

## 24. MVP implementation sequence

```text
MVP 1:
  DANDI metadata parser
  NWB structural parser
  rule-based role extraction
  evidence-backed manifest

MVP 2:
  LLM label normalizer for weird NWB field names
  derived-variable detector
  tool readiness scoring

MVP 3:
  DOI / paper abstract enrichment
  bioRxiv / Crossref / Semantic Scholar resolver
  abstract-to-role extraction

MVP 4:
  human correction loop
  corrected labels become future examples
  parser improves over time
```

---

## 25. Design principle

Do not build a magic LLM that pretends to understand data.

Build a parser that says:

```text
Here is what I saw.
Here is where I saw it.
Here is how confident I am.
Here is what can be derived.
Here is what analysis becomes valid.
Here is what still needs human confirmation.
```

That is the MouseHash compiler front-end.
