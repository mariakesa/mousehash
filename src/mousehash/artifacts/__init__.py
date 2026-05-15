"""On-disk artifact storage: env-resolved path roots + JSON/NPY/HTML I/O."""

from mousehash.artifacts.hashes import sha1_file
from mousehash.artifacts.io import (
    load_json,
    load_npy,
    save_csv,
    save_html,
    save_json,
    save_npy,
)
from mousehash.artifacts.paths import (
    artifact_root,
    cache_root,
    data_root,
    decompositions_root,
    ensure_dir,
    manifests_root,
    representations_root,
    reports_root,
    stimuli_root,
)

__all__ = [
    "artifact_root",
    "cache_root",
    "data_root",
    "decompositions_root",
    "ensure_dir",
    "load_json",
    "load_npy",
    "manifests_root",
    "representations_root",
    "reports_root",
    "save_csv",
    "save_html",
    "save_json",
    "save_npy",
    "sha1_file",
    "stimuli_root",
]
