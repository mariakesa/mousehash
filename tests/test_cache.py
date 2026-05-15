"""Tests for artifacts/cache.py — the content-addressed computation cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mousehash.artifacts.cache import (
    ComputationSpec,
    cache_dir_for,
    cached_computation,
    find_cached_view,
    fingerprint_array,
    save_cached_view,
)
from mousehash.artifacts.io import load_json
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.core.ids import ManifestId


# ---------- ComputationSpec.hash ----------

class TestSpecHash:
    def _base(self, **kwargs) -> ComputationSpec:
        defaults = dict(family="f", scope="s", name="n", parameters={"a": 1, "b": 2})
        defaults.update(kwargs)
        return ComputationSpec(**defaults)

    def test_deterministic(self):
        assert self._base().hash() == self._base().hash()

    def test_parameter_order_invariant(self):
        a = self._base(parameters={"a": 1, "b": 2})
        b = self._base(parameters={"b": 2, "a": 1})
        assert a.hash() == b.hash()

    def test_label_does_not_affect_hash(self):
        a = self._base(label="v0")
        b = self._base(label="v1")
        assert a.hash() == b.hash()

    def test_parameter_change_changes_hash(self):
        a = self._base(parameters={"a": 1})
        b = self._base(parameters={"a": 2})
        assert a.hash() != b.hash()

    def test_family_change_changes_hash(self):
        a = self._base(family="representations")
        b = self._base(family="compression")
        assert a.hash() != b.hash()

    def test_input_fingerprints_order_invariant(self):
        a = self._base(input_fingerprints=["aaa", "bbb"])
        b = self._base(input_fingerprints=["bbb", "aaa"])
        assert a.hash() == b.hash()

    def test_input_fingerprint_change_changes_hash(self):
        a = self._base(input_fingerprints=["aaa"])
        b = self._base(input_fingerprints=["bbb"])
        assert a.hash() != b.hash()


# ---------- fingerprint_array ----------

class TestFingerprintArray:
    def test_deterministic(self):
        a = np.arange(10).astype(np.float32)
        assert fingerprint_array(a) == fingerprint_array(a)

    def test_pixel_change_changes_fingerprint(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = a.copy()
        b[0, 0] = 1
        assert fingerprint_array(a) != fingerprint_array(b)

    def test_dtype_change_changes_fingerprint(self):
        a = np.arange(10).astype(np.int32)
        b = a.astype(np.int64)
        assert fingerprint_array(a) != fingerprint_array(b)

    def test_shape_change_changes_fingerprint(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((2, 8), dtype=np.uint8)
        assert fingerprint_array(a) != fingerprint_array(b)


# ---------- cache_dir_for ----------

class TestCacheDir:
    def test_dir_is_under_artifact_root(self, data_root_tmp: Path):
        spec = ComputationSpec(family="representations", scope="set1", name="vit", parameters={"k": 1})
        d = cache_dir_for(spec)
        assert d == data_root_tmp / "artifacts" / "representations" / "set1" / spec.hash()

    def test_dir_changes_with_scope(self, data_root_tmp: Path):
        a = ComputationSpec(family="f", scope="alpha", name="n", parameters={})
        b = ComputationSpec(family="f", scope="beta", name="n", parameters={})
        assert cache_dir_for(a) != cache_dir_for(b)


# ---------- cached_computation ----------

def _stub_view_and_summary(manifest_id: str, shape: list[int]) -> tuple[AnalysisView, dict]:
    view = AnalysisView.new(
        kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
        manifest_id=ManifestId(manifest_id),
        shape=shape,
        axes={"observations": "x", "features": "y"},
        source_roles=["stimuli"],
        transformation_lineage=["stub"],
    )
    return view, {"shape": shape, "stub": True}


class TestCachedComputation:
    def test_first_call_miss_runs_compute(self, data_root_tmp: Path):
        spec = ComputationSpec(family="f", scope="s", name="n", parameters={"x": 1})
        calls = []

        def compute(out_dir: Path) -> tuple[AnalysisView, dict[str, Any]]:
            calls.append(out_dir)
            view, summary = _stub_view_and_summary("mf_a", [3, 4])
            view.artifact_path = str(out_dir)
            return view, summary

        view, summary, from_cache = cached_computation(spec, compute)
        assert from_cache is False
        assert len(calls) == 1
        assert view.shape == [3, 4]
        assert summary["stub"] is True

    def test_second_call_is_cache_hit(self, data_root_tmp: Path):
        spec = ComputationSpec(family="f", scope="s", name="n", parameters={"x": 1})
        calls = []

        def compute(out_dir: Path) -> tuple[AnalysisView, dict[str, Any]]:
            calls.append(out_dir)
            view, summary = _stub_view_and_summary("mf_a", [3, 4])
            view.artifact_path = str(out_dir)
            return view, summary

        cached_computation(spec, compute)
        view, summary, from_cache = cached_computation(spec, compute)
        assert from_cache is True
        assert len(calls) == 1  # compute called only once
        assert view.shape == [3, 4]

    def test_parameter_change_invalidates_cache(self, data_root_tmp: Path):
        def make_spec(x):
            return ComputationSpec(family="f", scope="s", name="n", parameters={"x": x})

        calls = []
        def compute(out_dir: Path) -> tuple[AnalysisView, dict[str, Any]]:
            calls.append(out_dir)
            view, summary = _stub_view_and_summary("mf_p", [2, 2])
            view.artifact_path = str(out_dir)
            return view, summary

        cached_computation(make_spec(1), compute)
        cached_computation(make_spec(2), compute)
        assert len(calls) == 2

    def test_label_change_is_cache_hit(self, data_root_tmp: Path):
        """Label does not participate in the hash, so changing it is still a hit."""
        def compute(out_dir: Path) -> tuple[AnalysisView, dict[str, Any]]:
            view, summary = _stub_view_and_summary("mf_l", [2, 2])
            view.artifact_path = str(out_dir)
            return view, summary

        a = ComputationSpec(family="f", scope="s", name="n", label="v0", parameters={})
        b = ComputationSpec(family="f", scope="s", name="n", label="v1", parameters={})
        cached_computation(a, compute)
        _, _, from_cache = cached_computation(b, compute)
        assert from_cache is True

    def test_writes_spec_view_summary_jsons(self, data_root_tmp: Path):
        spec = ComputationSpec(family="f", scope="s", name="n", parameters={"x": 1})

        def compute(out_dir: Path) -> tuple[AnalysisView, dict[str, Any]]:
            view, summary = _stub_view_and_summary("mf_j", [2, 2])
            view.artifact_path = str(out_dir)
            return view, summary

        cached_computation(spec, compute)
        d = cache_dir_for(spec)
        assert (d / "spec.json").exists()
        assert (d / "view.json").exists()
        assert (d / "summary.json").exists()
        # spec.json round-trips
        rebuilt = ComputationSpec.model_validate(load_json(d / "spec.json"))
        assert rebuilt.hash() == spec.hash()

    def test_save_cached_view_helper(self, data_root_tmp: Path):
        spec = ComputationSpec(family="f", scope="s", name="n", parameters={})
        view, summary = _stub_view_and_summary("mf_s", [2, 2])
        d = save_cached_view(spec, view, summary)
        assert (d / "view.json").exists()
        loaded = find_cached_view(spec)
        assert loaded is not None
        assert loaded[0].view_id == view.view_id
