"""On-disk artifact storage: env-resolved path roots + JSON/NPY/HTML I/O + content-addressed cache."""

from mousehash.artifacts.cache import (
    ComputationSpec,
    cache_dir_for,
    cached_computation,
    find_cached_view,
    fingerprint_array,
    save_cached_view,
)
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
    "ComputationSpec",
    "artifact_root",
    "cache_dir_for",
    "cache_root",
    "cached_computation",
    "data_root",
    "decompositions_root",
    "ensure_dir",
    "find_cached_view",
    "fingerprint_array",
    "load_json",
    "load_npy",
    "manifests_root",
    "representations_root",
    "reports_root",
    "save_cached_view",
    "save_csv",
    "save_html",
    "save_json",
    "save_npy",
    "sha1_file",
    "stimuli_root",
]
