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