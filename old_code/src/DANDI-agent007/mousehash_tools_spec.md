# MouseHash Tool Readiness and Analysis Suggestion Specification
**Purpose:** Define the complete tool-signature layer that MouseHash needs in order to suggest analyses from a filled role manifest. This is the parser-facing companion to the transformation specification. It tells the agent which tools require which roles, what analysis view each tool consumes, which transformations usually produce that view, and what artifacts the tool should produce.
---
## 1. Core idea
MouseHash tools are not free-floating functions. Each tool has a typed scientific target. The parser fills the role manifest; the transformation layer builds analysis-ready views; the tool-readiness layer decides which analyses are ready, blocked, or derivable.
```text
RoleManifest -> RoleSignature check -> TransformationPlan -> AnalysisView -> ToolRun -> Artifact
```
---
## 2. Core bundle roles
| Role | Meaning |
|---|---|
| Conditions | Task labels, trial labels, stimulus labels, experimental groups, brain states, session phases, task epochs. |
| Stimuli | Sensory stimuli, model-derived features, semantic features, interventions, stimulation, stimulus embeddings. |
| Behavior | Choices, reaction times, running, pupil, pose, kinematics, licking, reward, behavioral state. |
| Neural data | Spikes, calcium events/traces, LFP, EEG/ECoG, population matrices, latent states, fitted neural responses. |
| Time organization | Continuous time, trials, epochs, frames, events, lags, windows, alignments, trajectory segmentation. |
| Metadata | Subject/session, brain region, probe/electrode/unit/cell identity, acquisition, preprocessing, model/tool provenance. |
---
## 3. Tool contract schema
```yaml
ToolSpec:
  tool_id:
  name:
  workflow_family:
  requires_roles:
  optional_roles:
  target_subcategories:
  requires_view:
  default_transformation_path:
  assumptions:
  failure_modes:
  validation_checks:
  output_artifacts:
  parser_readiness_rules:
  mvp_priority:
```
Readiness statuses:

```text
ready
ready_after_transformation
needs_confirmation
blocked
not_recommended
```
---
## 4. Tool-readiness pseudocode
```text
FUNCTION compute_tool_readiness(manifest, tool_catalog):

    reports = []

    FOR each tool IN tool_catalog:
        satisfied = []
        uncertain = []
        derivable = []
        missing = []

        FOR each required_role IN tool.requires_roles:
            role_status = manifest.status(required_role)

            IF role_status == present:
                satisfied.add(required_role)
            ELSE IF role_status == likely_present OR role_status == ambiguous:
                uncertain.add(required_role)
            ELSE IF role_status == derived_possible:
                derivable.add(required_role)
            ELSE:
                missing.add(required_role)

        IF missing is empty AND uncertain is empty AND derivable is empty:
            status = ready
        ELSE IF missing is empty AND derivable not empty:
            status = ready_after_transformation
        ELSE IF missing is empty AND uncertain not empty:
            status = needs_confirmation
        ELSE:
            status = blocked

        reports.add(ToolReadinessReport(tool, status, satisfied, uncertain, derivable, missing))

    RETURN rank_reports(reports)
```
---
## 5. Family-level target summary
| Family | Dominant role tuple | Common targets |
|---|---|---|
| Encoding / decoding | Encoding: Stimuli or Behavior + Neural Data + Time. Decoding: Neural Data + Conditions or Behavior + Time. | Stimulus-response models, behavioral covariates, class labels, train/test-safe prediction. |
| Dimensionality reduction / factor modeling | Mostly Neural Data, often plus Conditions, Time, Metadata. | Population structure, latent factors, trajectories, session/area comparisons. |
| Geometry / representational structure | Neural Data plus comparison axis: Stimuli, Conditions, Behavior, or Metadata. | RSMs, semantic axes, subspaces, manifolds, alignment across models/sessions/areas. |
| Dynamics / state-space | Neural Data + Time, often plus Conditions/Behavior/Metadata. | Temporal evolution, hidden states, attractors, lagged influence, trajectories. |
| Connectivity / interaction | Neural Data + Time + Metadata, often plus Conditions. | Unit/area interactions, correlations, coupling, assemblies, functional graphs. |
| Statistics / validation / visualization | Target roles of tested/plotted tool, plus Time and Metadata for resampling/reporting. | Nulls, uncertainty, leakage controls, multiple comparisons, plots, auditable reports. |
---
## 6. Complete tool catalog

### Encoding / decoding
| # | Tool | Required roles | Optional roles | Typical materialized view | Default transformation path | Output artifacts |
|---:|---|---|---|---|---|---|
| 1 | Fit Linear Encoding Model | Stimuli or Behavior; Neural data; Time organization | Conditions; Metadata | `observations x features -> observations x neurons` | selection/QC -> alignment -> response window -> design matrix -> split -> train-only scaling | model weights; predictions; metrics; report |
| 2 | Fit Ridge Encoding Model | Stimuli; Neural data; Time organization | Behavior; Conditions; Metadata | `observations x high-d features -> responses` | QC -> stimulus alignment -> feature extraction -> design matrix -> CV split -> train-only scaling | ridge weights; CV scores; prediction plots |
| 3 | Fit Lasso Encoding Model | Stimuli or Behavior; Neural data; Time organization | Conditions; Metadata | `observations x features -> neuron response` | alignment -> response extraction -> sparse candidate predictors -> split -> scaling | sparse weights; selected features; CV metrics |
| 4 | Fit Elastic Net Encoding Model | Stimuli or Behavior; Neural data; Time organization | Conditions; Metadata | `observations x features -> responses` | alignment -> response extraction -> design matrix -> split -> scaling | weights; sparsity/stability summary; CV metrics |
| 5 | Fit Poisson GLM Encoding Model | Stimuli or Behavior; Neural data; Time organization | Conditions; Metadata | `time/trial bins x predictors -> count` | bin spikes/events -> make predictors -> lag design optional -> split -> scaling | GLM coefficients; likelihood; deviance; CV metrics |
| 6 | Fit Negative Binomial Encoding Model | Stimuli or Behavior; Neural data; Time organization | Conditions; Metadata | `bins x predictors -> overdispersed count` | bin counts -> check overdispersion -> design matrix -> split | NB coefficients; dispersion; likelihood metrics |
| 7 | Fit Zero-Inflated Encoding Model | Stimuli or Behavior; Neural data; Time organization | Conditions; Metadata | `observations x predictors -> zero-inflated response` | event/count extraction -> zero-heavy response view -> design matrix -> split | zero-inflation parameters; response model; CV likelihood |
| 8 | Fit Logistic Decoder | Neural data; Conditions; Time organization | Behavior; Stimuli; Metadata | `observations x neurons -> binary label` | align/bin/aggregate neural data -> label vector -> split -> train-only scaling | classifier; accuracy/AUC; confusion matrix |
| 9 | Fit Multiclass Decoder | Neural data; Conditions; Time organization | Stimuli; Behavior; Metadata | `observations x neurons -> class label` | population matrix -> multiclass labels -> stratified split -> scaling | classifier; accuracy; confusion matrix |
| 10 | Fit Ridge Decoder | Neural data; Stimuli or Behavior; Time organization | Conditions; Metadata | `observations x neurons -> continuous target` | population matrix -> continuous target features -> split -> scaling | decoder weights; R2/correlation; predictions |
| 11 | Fit Temporal Encoding Model | Stimuli or Behavior; Neural data; Time organization | Conditions; Metadata | `time x lagged_features -> neural response` | sync clocks -> lagged design matrix -> binned responses -> time-aware split | temporal kernels; CV scores; lag importance |
| 12 | Fit Temporal Decoder | Neural data; Behavior or Conditions; Time organization | Stimuli; Metadata | `time x lagged_neurons -> target` | lag neural history -> target alignment -> time-block split -> scaling | temporal decoder; held-out metrics |
| 13 | Fit Receptive Field Model | Stimuli; Neural data; Time organization | Conditions; Metadata | `space x time stimulus -> response` | stimulus grid extraction -> spike/event alignment -> stimulus history matrix -> split | RF map; temporal kernel; validation metrics |
| 14 | Fit Spike-Triggered Average | Stimuli; Neural data; Time organization | Conditions; Metadata | `events x pre-event stimulus window` | detect/collect spikes -> stimulus history windows -> average | STA; uncertainty; event counts |
| 15 | Fit Spike-Triggered Covariance | Stimuli; Neural data; Time organization | Conditions; Metadata | `events x stimulus-history vectors` | spike-triggered ensemble -> covariance comparison -> eigenspectrum | STC components; covariance stats |
| 16 | Run Cross-Validated Encoding Evaluation | Stimuli or Behavior; Neural data; Time organization | Conditions; Metadata | `train/test observation splits` | define leakage-safe split -> train transforms on train -> evaluate held-out | CV metrics; split manifest; leakage checks |
| 17 | Run Cross-Validated Decoding Evaluation | Neural data; Conditions or Behavior; Time organization | Stimuli; Metadata | `train/test observation splits` | define labels -> split -> train-only normalization -> held-out decoding | CV accuracy; AUC; confusion matrix |
| 18 | Compare Feature Spaces for Encoding | Stimuli; Neural data; Time organization | Behavior; Conditions; Metadata | `multiple feature matrices -> same responses` | extract feature spaces -> align common observations -> shared splits | feature-space comparison table; plots |
| 19 | Compute Encoding Model Significance | Neural data; Stimuli or Behavior; Time organization | Conditions; Metadata | `model metrics per neuron/population` | fit baseline/null models -> permutation/bootstrap -> multiple-comparison correction | p-values/q-values; effect sizes |
| 20 | Compute Decoder Confusion Matrix | Neural data; Conditions; Time organization | Stimuli; Behavior; Metadata | `true label x predicted label` | held-out predictions -> label alignment -> confusion table | confusion matrix; class metrics |

### Dimensionality reduction / factor modeling
| # | Tool | Required roles | Optional roles | Typical materialized view | Default transformation path | Output artifacts |
|---:|---|---|---|---|---|---|
| 21 | Run PCA on Neural Activity | Neural data | Conditions; Time organization; Metadata | `observations x neurons` | population matrix -> optional scaling -> PCA | scores; loadings; explained variance |
| 22 | Run Trial-Averaged PCA | Neural data; Conditions; Time organization | Stimuli; Behavior; Metadata | `condition x time x neurons or condition x neurons` | align trials -> group/average by condition -> scale -> PCA | condition scores; loadings; explained variance |
| 23 | Run demixed PCA | Neural data; Conditions; Time organization | Stimuli; Behavior; Metadata | `condition/factor x time x neurons` | factor labels -> trial tensor -> condition averages -> dPCA | demixed components; variance by factor |
| 24 | Run Factor Analysis | Neural data | Conditions; Time organization; Metadata | `observations x neurons` | population matrix -> scaling -> factor analysis | factor loadings; shared/private variance |
| 25 | Run Gaussian Process Factor Analysis | Neural data; Time organization | Conditions; Behavior; Metadata | `trials x time x neurons` | trial sequences -> bin spikes -> GPFA | smooth latent trajectories; likelihood |
| 26 | Run NMF on Neural Responses | Neural data | Conditions; Stimuli; Time organization; Metadata | `observations x neurons, nonnegative` | nonnegative response matrix -> optional normalization -> NMF | components; activations; reconstruction error |
| 27 | Run ICA on Neural Responses | Neural data | Conditions; Time organization; Metadata | `observations x neurons/signals` | population/signal matrix -> centering/whitening -> ICA | independent components; mixing matrix |
| 28 | Run Sparse PCA | Neural data | Conditions; Time organization; Metadata | `observations x neurons` | population matrix -> scaling -> sparse PCA | sparse loadings; scores |
| 29 | Run UMAP Embedding | Neural data | Conditions; Stimuli; Behavior; Time organization; Metadata | `observations x neural features -> 2D/3D` | population matrix -> optional scaling -> UMAP | embedding coordinates; visualization |
| 30 | Run t-SNE Embedding | Neural data | Conditions; Stimuli; Behavior; Time organization; Metadata | `observations x neural features -> 2D` | population matrix -> optional scaling -> t-SNE | embedding coordinates; visualization |
| 31 | Run Isomap Embedding | Neural data | Conditions; Behavior; Time organization; Metadata | `observations x neural features -> manifold coords` | population matrix -> neighborhood graph -> Isomap | manifold coordinates; geodesic graph |
| 32 | Run Diffusion Map Embedding | Neural data | Conditions; Behavior; Time organization; Metadata | `observations x neural features -> diffusion coords` | population matrix -> affinity/kernel -> diffusion map | diffusion components; eigenvalues |
| 33 | Run CEBRA Embedding | Neural data; Time organization | Behavior; Stimuli; Conditions; Metadata | `samples x neurons with contrastive labels` | population samples -> contrast labels/time positives -> CEBRA | embedding; contrastive loss; plots |
| 34 | Run LFADS-Style Latent Dynamics Model | Neural data; Time organization | Conditions; Behavior; Metadata | `trials x time x neurons` | spike-count sequences -> train/validation split -> latent dynamics model | latent trajectories; denoised rates |
| 35 | Run Autoencoder Embedding | Neural data | Conditions; Stimuli; Behavior; Time organization; Metadata | `observations x neurons -> latent` | population matrix -> train/test split -> autoencoder | latent codes; reconstruction metrics |
| 36 | Run Variational Autoencoder Embedding | Neural data | Conditions; Stimuli; Behavior; Time organization; Metadata | `observations x neurons -> probabilistic latent` | population matrix -> split -> VAE | latent posterior; reconstruction/KL metrics |
| 37 | Run Contrastive Embedding Model | Neural data; Conditions or Stimuli or Behavior; Time organization | Metadata | `paired/contrastive sample batches` | population samples -> positive/negative sampling rules -> contrastive training | embedding; retrieval/contrast metrics |
| 38 | Compute Latent Dimensionality | Neural data | Conditions; Time organization; Metadata | `observations x neurons or latent covariance` | population/latent covariance -> spectrum analysis | dimensionality estimate; spectrum |
| 39 | Compute Participation Ratio | Neural data | Conditions; Time organization; Metadata | `observations x neurons -> covariance spectrum` | population matrix -> covariance -> participation ratio | PR metric; subgroup table |
| 40 | Compare Latent Spaces Across Sessions | Neural data; Metadata | Conditions; Stimuli; Behavior; Time organization | `session x observations x latent dimensions` | per-session embeddings -> alignment anchors -> subspace comparison | alignment metrics; stability report |

### Geometry / representational structure
| # | Tool | Required roles | Optional roles | Typical materialized view | Default transformation path | Output artifacts |
|---:|---|---|---|---|---|---|
| 41 | Compute Representational Similarity Matrix | Neural data | Conditions; Stimuli; Behavior; Time organization; Metadata | `items x neurons -> items x items similarity` | aggregate response vectors -> distance/similarity matrix | RSM/RDM; item metadata |
| 42 | Run Representational Similarity Analysis | Neural data; Stimuli or Behavior or Conditions | Time organization; Metadata | `RSM_neural compared to RSM_target` | neural RDM -> target RDM -> permutation/statistic | RSA correlation; p-values; plots |
| 43 | Compute Centered Kernel Alignment | Neural data; Stimuli or Behavior or Neural data | Conditions; Time organization; Metadata | `same observations x features in two spaces` | align observations -> compute kernels -> CKA | CKA score; null distribution |
| 44 | Compute Canonical Correlation Analysis | Neural data; Neural data or Stimuli or Behavior | Conditions; Time organization; Metadata | `observations x features_A and observations x features_B` | align multivariate spaces -> split -> CCA | canonical correlations; weights |
| 45 | Compute Partial Least Squares Alignment | Neural data; Stimuli or Behavior or Neural data | Conditions; Time organization; Metadata | `X observations x features -> Y observations x features` | align source/target matrices -> split -> PLS | PLS components; prediction metrics |
| 46 | Run Procrustes Alignment | Neural data; Metadata | Conditions; Stimuli; Behavior; Time organization | `matched items x dimensions across spaces` | match anchors -> latent spaces -> Procrustes | aligned coordinates; residual error |
| 47 | Compute Subspace Angles | Neural data | Conditions; Stimuli; Behavior; Metadata | `basis_A and basis_B` | fit/load subspaces -> principal angles | angle spectrum; overlap summary |
| 48 | Compute Projection Overlap | Neural data | Conditions; Stimuli; Behavior; Metadata | `basis/projection matrices` | define subspaces -> compute projection overlap | overlap score; null comparison |
| 49 | Compute Neural Manifold Curvature | Neural data | Time organization; Conditions; Behavior; Metadata | `points on learned manifold` | embedding/manifold graph -> local curvature estimates | curvature metrics; manifold plot |
| 50 | Compute Geodesic Distances on Neural Manifold | Neural data | Time organization; Conditions; Behavior; Metadata | `manifold points -> pairwise geodesic matrix` | embedding graph -> geodesic shortest paths | geodesic distance matrix |
| 51 | Run Principal Geodesic Analysis | Neural data | Conditions; Time organization; Behavior; Metadata | `manifold points + metric` | manifold-valued points -> tangent/geodesic model | principal geodesics; variance |
| 52 | Run Contrastive PCA | Neural data; Conditions | Stimuli; Behavior; Time organization; Metadata | `foreground observations x neurons vs background observations x neurons` | foreground/background matrices -> covariance contrast | contrastive components; enrichment scores |
| 53 | Run Contrastive Principal Geodesic Analysis | Neural data; Conditions | Behavior; Time organization; Metadata | `foreground/background manifold-valued observations` | foreground/background manifolds -> contrastive PGA | contrastive geodesics; group differences |
| 54 | Find Semantic Axes in Neural Space | Neural data; Stimuli or Conditions; Time organization | Behavior; Metadata | `observations x neurons + concept labels/features` | aligned responses -> concept labels/features -> axis fitting | semantic axis vector; validation metrics |
| 55 | Project Neural Activity onto Semantic Axis | Neural data; Stimuli or Conditions; Time organization | Behavior; Metadata | `observations x neurons -> scalar projection` | population matrix -> semantic axis -> projection | projection values; condition plots |
| 56 | Compute Neural Geometry Stability | Neural data; Time organization | Conditions; Stimuli; Behavior; Metadata | `repeat/session RSMs or subspaces` | matched repeats/sessions -> geometry objects -> stability metric | stability scores; bootstrap CIs |
| 57 | Compute Population Distance Matrix | Neural data | Conditions; Stimuli; Behavior; Time organization; Metadata | `items x neurons -> pairwise distance` | response vectors -> distance metric | distance matrix; item annotations |
| 58 | Compute Class Separability in Neural Space | Neural data; Conditions | Stimuli; Behavior; Time organization; Metadata | `observations x neurons + labels` | population matrix + labels -> separability metric | separability score; margins |
| 59 | Compute Margin Distribution | Neural data; Conditions | Stimuli; Behavior; Time organization; Metadata | `observations x classifier scores/margins` | classifier scores -> margin distribution | margin table; plots |
| 60 | Map Stimulus Geometry to Neural Geometry | Stimuli; Neural data; Time organization | Conditions; Behavior; Metadata | `stimulus RSM -> neural RSM` | stimulus/model distances -> neural distances -> mapping/stat test | geometry mapping score; nulls |

### Dynamics / state-space
| # | Tool | Required roles | Optional roles | Typical materialized view | Default transformation path | Output artifacts |
|---:|---|---|---|---|---|---|
| 61 | Fit Linear Dynamical System | Neural data; Time organization | Behavior; Conditions; Metadata | `trials/time x neurons -> latent states` | ordered sequences -> split -> LDS | latent states; transition matrix |
| 62 | Fit Switching Linear Dynamical System | Neural data; Time organization | Conditions; Behavior; Metadata | `sequences x time x neurons -> regimes + latent states` | sequences -> SLDS model | regime sequence; latent states |
| 63 | Fit Hidden Markov Model | Neural data; Time organization | Behavior; Conditions; Metadata | `time x neural observations -> hidden states` | ordered observations -> HMM | hidden states; transition matrix |
| 64 | Fit Autoregressive Neural State Model | Neural data; Time organization | Behavior; Conditions; Metadata | `time x lagged state -> next state` | lagged state matrix -> AR model | AR coefficients; forecast metrics |
| 65 | Fit Recurrent Neural Network Dynamics Model | Neural data; Time organization | Stimuli; Behavior; Conditions; Metadata | `sequence batches: time x features` | sequence tensors -> split -> RNN dynamics model | trained model; prediction metrics |
| 66 | Compute Neural Trajectories | Neural data; Time organization | Conditions; Behavior; Metadata | `trial x time x latent dimensions` | trial tensor -> dimensional transform -> ordered latent paths | trajectory artifact; plots |
| 67 | Align Neural Trajectories Across Trials | Neural data; Time organization | Conditions; Behavior; Metadata | `trials x time x latent dimensions` | trajectory extraction -> event/time warping alignment | aligned trajectories; warping metrics |
| 68 | Compute Trajectory Speed | Neural data; Time organization | Conditions; Behavior; Metadata | `time-indexed trajectory -> speed(t)` | latent trajectory -> finite differences with dt | speed curve; condition summary |
| 69 | Compute Trajectory Curvature | Neural data; Time organization | Conditions; Behavior; Metadata | `time-indexed trajectory -> curvature(t)` | latent trajectory -> derivative/curvature estimator | curvature curve; summary metrics |
| 70 | Compute Fixed Points of Learned Dynamics | Neural data; Time organization | Conditions; Behavior; Metadata | `learned vector field / transition model` | fit dynamics model -> solve fixed points | fixed points; stability analysis |
| 71 | Analyze Attractor Structure | Neural data; Time organization | Conditions; Behavior; Metadata | `state-space trajectories + dynamics model` | trajectories/dynamics -> attractor diagnostics | attractor report; state plots |
| 72 | Compute Rotational Dynamics with jPCA | Neural data; Time organization | Conditions; Behavior; Metadata | `conditions x time x neurons` | condition-averaged trajectories -> jPCA | rotational components; phase plots |
| 73 | Compute State Transition Matrix | Neural data; Time organization | Conditions; Behavior; Metadata | `state_t -> state_t+1 counts/probabilities` | discrete states -> transition count/probability matrix | transition matrix; state diagram |
| 74 | Detect Neural State Changes | Neural data; Time organization | Conditions; Behavior; Metadata | `time x neural features -> changepoints` | time series/population matrix -> changepoint detection | changepoints; state segments |
| 75 | Detect Neural Events or Motifs | Neural data; Time organization | Conditions; Behavior; Metadata | `sliding windows x neural features` | windowed time series -> motif detection | motif templates; occurrence table |
| 76 | Compare Dynamics Across Conditions | Neural data; Time organization; Conditions | Behavior; Metadata | `condition x trial x time x neural/latent` | condition trajectories/models -> comparison statistic | condition differences; null tests |
| 77 | Estimate Neural Timescales | Neural data; Time organization | Conditions; Behavior; Metadata | `time series per neuron/population/latent` | ordered time series -> autocorrelation/decay fit | timescale estimates; confidence intervals |
| 78 | Compute Cross-Area Lagged Influence | Neural data; Time organization; Metadata | Conditions; Behavior | `time x features_area_A, time x features_area_B` | area grouping -> lagged predictors -> influence metric | lagged influence scores |
| 79 | Fit Granger Causality Model | Neural data; Time organization; Metadata | Conditions; Behavior | `time x neural variables with lagged predictors` | multivariate lag matrix -> Granger model | causality statistics; p-values |
| 80 | Fit Dynamic Mode Decomposition | Neural data; Time organization | Conditions; Behavior; Metadata | `state matrix X and shifted matrix Xprime` | ordered snapshots -> X/Xprime -> DMD | modes; eigenvalues; reconstructions |

### Connectivity / interaction
| # | Tool | Required roles | Optional roles | Typical materialized view | Default transformation path | Output artifacts |
|---:|---|---|---|---|---|---|
| 81 | Compute Noise Correlations | Neural data; Conditions; Time organization | Stimuli; Behavior; Metadata | `stimulus/condition x trial x neuron` | trial tensor -> subtract condition mean -> residual correlations | noise correlation matrix; QC |
| 82 | Compute Signal Correlations | Neural data; Conditions or Stimuli; Time organization | Metadata | `condition/stimulus x neuron tuning matrix` | condition means/tuning curves -> pairwise correlations | signal correlation matrix |
| 83 | Compute Functional Connectivity Matrix | Neural data; Time organization | Conditions; Behavior; Metadata | `time/observations x neural units -> unit x unit matrix` | simultaneous activity matrix -> dependency metric | connectivity matrix; graph |
| 84 | Compute Partial Correlation Network | Neural data; Time organization | Conditions; Behavior; Metadata | `observations x neural units -> precision/partial corr` | activity matrix -> covariance/precision -> partial corr | partial correlation network |
| 85 | Fit Graphical Lasso Network | Neural data | Conditions; Time organization; Metadata | `observations x neural units -> sparse precision` | activity matrix -> covariance -> graphical lasso | sparse precision matrix; graph |
| 86 | Compute Cross-Correlograms | Neural data; Time organization | Conditions; Metadata | `unit_pair x lag histogram` | spike/event times -> pairwise lag histograms | cross-correlograms; pair table |
| 87 | Estimate Coupling GLM | Neural data; Time organization | Stimuli; Behavior; Conditions; Metadata | `target count ~ stimulus + coupled neural history` | target spike counts -> population history predictors -> GLM | coupling filters; model metrics |
| 88 | Compute Population Coupling | Neural data; Time organization | Conditions; Metadata | `observations x neurons -> unit-population coupling` | unit activity + population activity -> coupling metric | coupling scores per unit |
| 89 | Detect Neural Assemblies | Neural data; Time organization | Conditions; Behavior; Metadata | `time/observations x neurons` | coactivation matrix -> assembly detection | assembly memberships; activation time courses |
| 90 | Run Community Detection on Functional Graph | Neural data; Metadata | Conditions; Time organization | `node x node connectivity matrix -> communities` | connectivity graph -> community detection | communities; graph metrics |

### Statistics / validation / visualization
| # | Tool | Required roles | Optional roles | Typical materialized view | Default transformation path | Output artifacts |
|---:|---|---|---|---|---|---|
| 91 | Run Permutation Test | Any target bundle roles used by tested tool | Conditions; Time organization; Metadata | `observed metric + shuffled null metrics` | define exchangeability units -> shuffle -> recompute metric | null distribution; p-value |
| 92 | Run Bootstrap Confidence Intervals | Any measured output or source bundle roles | Conditions; Time organization; Metadata | `resampled observations/trials/sessions` | define resampling unit -> bootstrap -> summarize interval | confidence intervals; bootstrap samples |
| 93 | Run Trial Shuffle Control | Neural data; Time organization | Conditions; Stimuli; Behavior; Metadata | `trial-shuffled observations` | preserve within-trial structure -> shuffle trial labels/alignment | shuffle null metric; control report |
| 94 | Run Stimulus Shuffle Control | Stimuli; Neural data; Time organization | Conditions; Metadata | `stimulus-response mapping with shuffled labels` | shuffle stimulus labels/features -> recompute mapping | null scores; p-values |
| 95 | Run Circular Shift Control | Neural data; Time organization | Stimuli; Behavior; Conditions; Metadata | `time series with circular shifts` | shift time series/covariates preserving autocorrelation | shift null distribution |
| 96 | Correct for Multiple Comparisons | Metadata | Neural data; Conditions; Stimuli; Behavior | `table of p-values/effects -> corrected q-values` | define test family -> FDR/FWER correction | q-values; corrected decisions |
| 97 | Generate Raster Plot | Neural data; Time organization | Conditions; Stimuli; Behavior; Metadata | `trials x spike/event times` | align events -> collect spike/event times -> plot by trial | raster figure; source table |
| 98 | Generate PSTH Plot | Neural data; Time organization | Conditions; Stimuli; Behavior; Metadata | `condition/trial x time bins -> firing rate` | align -> bin/smooth -> aggregate by condition | PSTH figure; rate table |
| 99 | Generate Latent Trajectory Plot | Neural data; Time organization | Conditions; Behavior; Metadata | `trial/condition x time x latent dimensions` | extract latent trajectories -> annotate by condition/behavior | trajectory plot; latent artifact |
| 100 | Generate Structure Discovery Report | Artifacts from any role-signed tools; Metadata | All six roles | `tool artifacts + role manifests -> report` | collect manifest/artifacts/provenance -> compose report | HTML/Markdown/PDF report; audit bundle |

---
## 7. Global assumptions by workflow family
### Encoding / decoding
- Observations must be aligned between predictors and responses.
- Train/test splits must be leakage-safe.
- Normalization and feature selection must be fitted on training data only for evaluative workflows.
- Labels must not be derived from future information unless the scientific question explicitly permits it.

### Dimensionality reduction / factor modeling
- The observation axis must be explicit.
- Scaling choices must be provenance-tracked.
- Exploratory embeddings and evaluative embeddings should be separate artifacts.
- Trial averaging changes the scientific object and must be recorded.

### Geometry / representational structure
- Matched item identities are mandatory when comparing geometries.
- Distance/similarity metric must be recorded.
- RDM/kernel construction is a transformation, not hidden inside RSA/CKA.

### Dynamics / state-space
- Time order and sampling intervals must be known.
- Sequence boundaries must be explicit.
- Time-blocked splits are preferred for predictive evaluation.

### Connectivity / interaction
- Simultaneity and shared clock assumptions must be checked.
- Common input and stimulus effects should be considered.
- Correlation is not causation; causal language requires stronger assumptions.

### Statistics / validation / visualization
- The resampling unit must be explicit: trial, stimulus, session, unit, subject, or time block.
- Null controls must preserve the relevant dependency structure.
- Visualizations should expose grouping, alignment event, window, and preprocessing choices.
---
## 8. Parser suggestion ranking
```text
FUNCTION rank_analysis_suggestions(tool_readiness_reports):

    score = 0

    IF status == ready:
        score += 100
    IF status == ready_after_transformation:
        score += 80
    IF status == needs_confirmation:
        score += 50
    IF status == blocked:
        score -= 100

    score += number_of_required_roles_present * 5
    score += number_of_optional_roles_present * 1

    IF tool is in MVP priority set:
        score += 20

    IF tool produces easy-to-inspect visualization/report:
        score += 10

    IF tool has high leakage risk and no validation plan:
        score -= 30

    RETURN sorted suggestions by score
```
---
## 9. MVP priority set
For a first MouseHash DANDI-agent, prioritize tools that are broadly useful, robust, and easy to inspect:

```yaml
mvp_tools:
  - Generate Raster Plot
  - Generate PSTH Plot
  - Run PCA on Neural Activity
  - Run Trial-Averaged PCA
  - Fit Logistic Decoder
  - Run Cross-Validated Decoding Evaluation
  - Fit Ridge Encoding Model
  - Run Cross-Validated Encoding Evaluation
  - Compute Representational Similarity Matrix
  - Run Representational Similarity Analysis
  - Compute Noise Correlations
  - Run Permutation Test
  - Generate Latent Trajectory Plot
  - Generate Structure Discovery Report
```
---
## 10. Parser output example
```yaml
tool_readiness:
  - tool: Generate PSTH Plot
    status: ready_after_transformation
    satisfied_roles:
      - neural_data.spikes
      - time_organization.trials
      - time_organization.events
    required_view: condition/trial x time bins -> firing rate
    suggested_transformations:
      - align_to_stimulus_onset
      - bin_spikes
      - gaussian_smooth_psth
      - aggregate_by_condition
    artifacts:
      - psth_figure
      - rate_table

  - tool: Fit Logistic Decoder
    status: needs_confirmation
    satisfied_roles:
      - neural_data.spikes
      - time_organization.trials
    uncertain_roles:
      - conditions.trial_labels
    required_view: observations x neurons -> binary label
    suggested_transformations:
      - make_population_state_matrix
      - assign_choice_or_condition_labels
      - train_test_split
      - train_only_scaling
```
---
## 11. Practical rule
The parser should never say only: "run PCA". It should say:

```text
Run Trial-Averaged PCA because neural_data, conditions, and trials are present.
Required view: condition x time x neuron tensor.
Transformation path: select trials -> align to event -> extract response window -> average by condition -> normalize -> PCA.
Readiness: ready_after_transformation.
Validation/reporting: explained variance, condition labels, artifact provenance.
```

This is what makes MouseHash an agent-compatible scientific workflow system rather than a pile of notebook snippets.
