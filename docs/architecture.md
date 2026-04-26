mousehash/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
│
├── docs/
│   ├── architecture.md
│   ├── prototype_v0.md
│   └── schema_notes.md
│
├── configs/
│   ├── default.yaml
│   └── prototype_natural_scenes.yaml
│
├── scripts/
│   ├── setup_schema.py
│   ├── ingest_natural_scenes.py
│   ├── compute_representations.py
│   ├── compute_decompositions.py
│   ├── build_reports.py
│   └── run_prototype.py
│
├── src/
│   └── mousehash/
│       ├── __init__.py
│       ├── config.py
│       ├── settings.py
│       │
│       ├── schema/
│       │   ├── __init__.py
│       │   ├── stimuli.py
│       │   ├── representations.py
│       │   ├── decompositions.py
│       │   └── reports.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── allen/
│       │   │   ├── __init__.py
│       │   │   ├── natural_scenes.py
│       │   │   └── stimulus_fetch.py
│       │   ├── representations/
│       │   │   ├── __init__.py
│       │   │   ├── vit_imagenet.py
│       │   │   └── animate_inanimate.py
│       │   ├── decompositions/
│       │   │   ├── __init__.py
│       │   │   ├── pca.py
│       │   │   └── nmf.py
│       │   └── reports/
│       │       ├── __init__.py
│       │       ├── pca_html.py
│       │       └── nmf_html.py
│       │
│       ├── artifacts/
│       │   ├── __init__.py
│       │   ├── paths.py
│       │   ├── io.py
│       │   └── manifests.py
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   └── prototype_natural_scenes.py
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   └── coordinator.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── hashing.py
│           ├── imagenet.py
│           ├── logging.py
│           └── serialization.py
│
├── tests/
│   ├── test_stimuli.py
│   ├── test_representations.py
│   ├── test_decompositions.py
│   └── test_reports.py
│
└── notebooks/
    └── exploration.ipynb

MouseHash is best understood as a small data-processing system with an AI control surface, not just a model wrapper. The core architecture is: configuration and environment in config.py, persistent metadata in DataJoint schemas like representations.py and decompositions.py, heavy numerical artifacts on disk via paths.py and io.py, pure-ish compute modules in tools/*, and an LLM coordinator in coordinator.py plus tools.py. One practical note: architecture.md looks more like an older intended tree than a faithful description of the current codebase.

At a structural level, the dominant pattern is a staged materialization pipeline. Ingestion writes stimuli and metadata, representation computes logits and labels, decomposition derives PCA/NMF factors, and reporting turns stored artifacts into HTML. Each stage reads prior outputs, writes new files, then registers the result in a schema table. You can see that most clearly in vit_imagenet.py, compute.py, and build.py. This is close to a pipeline pattern, but with an important twist: the database is not the payload store, it is the state index. DataJoint records mostly point at files on disk rather than embedding arrays directly. That makes the system feel like a hybrid of a workflow engine and a content-addressed artifact store.

A second important pattern is “spec-driven computation.” The system does not just run “the representation” or “the NMF.” It runs a representation keyed by representation_spec_id, a labeling rule keyed by rule_id, and a decomposition keyed by decomposition_spec_id. Those specs live in lookup tables such as representations.py and decompositions.py. In effect, specs are first-class strategy objects, but stored relationally instead of implemented as Python subclasses. That is a real design choice: configuration is elevated into durable, queryable identity. I’d call this pattern “Relational Strategy Keys.”

A third pattern is pervasive idempotent materialization. Almost every stage checks whether its output already exists in DataJoint and short-circuits if it does. Representation does this in vit_imagenet.py, decomposition does it in compute.py, and report generation does it in build.py. This is not just defensive coding. It defines the operating model: stages are replayable commands over a partially materialized graph. A useful name for that pattern is “Checkpoint-by-Existence.” The system decides what is complete by asking whether the node exists, not by replay logs or in-memory state.

The query layer in queries.py adds another design idea: a lightweight read model over the pipeline graph. pipeline_status, format_status, and next_pipeline_step turn low-level table existence into a higher-level state machine. That is similar to CQRS, but much smaller in scope. The write model is the actual stage execution; the read model is a compact projection of completion state. I’d call this “Projection-Oriented Orchestration.”

The AI layer is also architecturally distinct. The coordinator in coordinator.py does not directly implement pipeline logic. Instead, it defines policy in prompt text and delegates capability through tool wrappers in tools.py. That is effectively a “Policy Prompt + Deterministic Tool Substrate” pattern. The language model owns sequencing decisions and user interaction shape; the Python tools own side effects and correctness. This keeps the unsafe part narrow: the agent can choose among tools, but the tools still enforce the real boundaries.

There is also a subtle semantic pattern around labels and rules. The animate/inanimate split is not baked into model code. It is represented as a lookup-table rule in representations.py and applied by a tiny transformation in the representation stage. That separates “prediction production” from “semantic interpretation.” I’d call that “Late Semantic Binding”: raw model outputs are preserved, and domain meaning is attached afterward through explicit rules.

If I had to summarize the library in one sentence, I’d call it: a spec-keyed, idempotent, artifact-backed analysis pipeline with an LLM orchestration layer on top.

The non-GoF patterns I think are genuinely present here are:

“Relational Strategy Keys”: behavior is selected by durable spec rows, not subclass hierarchies.
“Checkpoint-by-Existence”: stage completion is encoded by the existence of registered artifacts.
“Projection-Oriented Orchestration”: orchestration reads a projected status model rather than raw execution flow.
“Artifact-Indexed Persistence”: the database stores identity and lineage, while arrays live on disk.
“Late Semantic Binding”: model outputs are produced first and interpreted later via explicit domain rules.
“Policy Prompt + Deterministic Tool Substrate”: the LLM governs sequencing, but only through bounded tool affordances.


Design Pattern View
This refactor would formalize a new pattern already latent in MouseHash:
“Canonical Stimulus Spine”
Dataset-specific sources are normalized into one canonical stimulus model, and all downstream analytics depend only on that canonical layer.

