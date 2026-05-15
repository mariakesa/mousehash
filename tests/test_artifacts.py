"""Tests for artifacts/paths.py, io.py, hashes.py.

The `data_root_tmp` fixture isolates MOUSEHASH_DATA_ROOT per test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousehash.artifacts.hashes import sha1_file
from mousehash.artifacts.io import load_json, load_npy, save_csv, save_html, save_json, save_npy
from mousehash.artifacts.paths import (
    MissingEnvError,
    artifact_root,
    cache_root,
    data_root,
    decompositions_root,
    manifests_root,
    representations_root,
    reports_root,
    stimuli_root,
)


# ---------- paths ----------

class TestPaths:
    def test_data_root_requires_env_var(self):
        with pytest.raises(MissingEnvError):
            data_root()

    def test_data_root_returns_env_value(self, data_root_tmp: Path):
        assert data_root() == data_root_tmp

    @pytest.mark.parametrize("helper,subdir", [
        (stimuli_root, "stimuli"),
        (cache_root, "cache"),
        (reports_root, "reports"),
        (artifact_root, "artifacts"),
        (manifests_root, "manifests"),
    ])
    def test_default_subroot_under_data_root(self, helper, subdir, data_root_tmp: Path):
        assert helper() == data_root_tmp / subdir
        assert helper().exists()  # ensure_dir effect

    def test_representations_root_under_artifact_root(self, data_root_tmp: Path):
        assert representations_root() == data_root_tmp / "artifacts" / "representations"

    def test_decompositions_root_under_artifact_root(self, data_root_tmp: Path):
        assert decompositions_root() == data_root_tmp / "artifacts" / "decompositions"

    def test_subroot_override_env_var(self, data_root_tmp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        custom = tmp_path / "custom_stimuli_disk"
        custom.mkdir()
        monkeypatch.setenv("MOUSEHASH_STIMULI_ROOT", str(custom))
        from mousehash.artifacts import paths as _paths
        _paths._load_dotenv_once.cache_clear()
        assert stimuli_root() == custom

    def test_missing_env_error_subclasses_mousehash_error(self):
        from mousehash.core.errors import MouseHashError
        assert issubclass(MissingEnvError, MouseHashError)


# ---------- io ----------

class TestIO:
    def test_json_round_trip(self, tmp_path: Path):
        data = {"name": "alpha", "n": 42, "nested": {"k": [1, 2, 3]}}
        path = save_json(tmp_path / "x.json", data)
        assert path.exists()
        assert load_json(path) == data

    def test_json_handles_pathlike_via_default_str(self, tmp_path: Path):
        # Path is not JSON-serializable; save_json passes default=str.
        path = save_json(tmp_path / "x.json", {"p": Path("/tmp/a")})
        rebuilt = load_json(path)
        assert rebuilt["p"] == "/tmp/a"

    def test_save_json_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c" / "x.json"
        save_json(nested, {"ok": True})
        assert nested.exists()

    def test_npy_round_trip(self, tmp_path: Path):
        arr = np.arange(12).reshape(3, 4).astype(np.float32)
        path = save_npy(tmp_path / "arr.npy", arr)
        loaded = load_npy(path)
        np.testing.assert_array_equal(loaded, arr)
        assert loaded.dtype == arr.dtype

    def test_html_write(self, tmp_path: Path):
        body = "<html><body>hello</body></html>"
        path = save_html(tmp_path / "out.html", body)
        assert path.read_text(encoding="utf-8") == body

    def test_csv_with_rows(self, tmp_path: Path):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        path = save_csv(tmp_path / "out.csv", rows)
        text = path.read_text()
        assert "a,b" in text
        assert "1,x" in text
        assert "2,y" in text

    def test_csv_empty_rows(self, tmp_path: Path):
        path = save_csv(tmp_path / "empty.csv", [])
        assert path.read_text() == ""


# ---------- hashes ----------

class TestHashes:
    def test_sha1_deterministic(self, tmp_path: Path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"hello world")
        h1 = sha1_file(p)
        h2 = sha1_file(p)
        assert h1 == h2
        # Known SHA-1 of "hello world"
        assert h1 == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"

    def test_sha1_differs_per_content(self, tmp_path: Path):
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"alpha")
        p2.write_bytes(b"alpha\n")
        assert sha1_file(p1) != sha1_file(p2)

    def test_sha1_chunk_size_invariant(self, tmp_path: Path):
        p = tmp_path / "big.bin"
        p.write_bytes(b"x" * 200_000)
        assert sha1_file(p, chunk_size=4096) == sha1_file(p, chunk_size=131072)
