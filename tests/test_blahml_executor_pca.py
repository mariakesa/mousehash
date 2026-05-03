from __future__ import annotations

from typing import Any

import pytest

from mousehash.blahml.registry import ManifestRegistry
from mousehash.blahml.spec_builder import build_resolved_spec


def _resolved_pca():
    reg = ManifestRegistry()
    registered = reg.by_id("run_pca")
    answers = {
        "input_artifact": {
            "scene_set_id": "allen_natural_scenes_v1",
            "representation_spec_id": "vit_b16_imagenet_cpu",
            "rule_id": "imagenet_top1_leq_397",
        },
        "fit_scope": "train_only",   # avoid the leakage warning
        "n_components": 5,
    }
    return registered, build_resolved_spec(
        registered.manifest,
        answers,
        manifest_sha256=registered.sha256,
    )


def test_executor_persists_run_and_dispatches(monkeypatch) -> None:
    """End-to-end: build_resolved_spec → run_from_resolved_spec.

    Monkeypatches both DataJoint inserts and the compute orchestrator so
    the test exercises the BlahML wiring without touching a live database.
    """
    inserted_run_specs: list[dict[str, Any]] = []
    inserted_decomp_specs: list[dict[str, Any]] = []
    compute_calls: list[dict[str, Any]] = []

    class _FakeToolRunSpec:
        @staticmethod
        def insert1(row, skip_duplicates=False):
            inserted_run_specs.append(row)

    class _FakeDecompositionSpec:
        @staticmethod
        def insert1(row, skip_duplicates=False):
            inserted_decomp_specs.append(row)

    def _fake_compute(**kwargs):
        compute_calls.append(kwargs)
        return dict(method="pca", n_components=kwargs["decomposition_spec_id"])

    import mousehash.schema.blahml as blahml_schema
    import mousehash.schema.decompositions as decomp_schema
    import mousehash.tools.decompositions.compute as compute_module

    monkeypatch.setattr(blahml_schema, "ToolRunSpec", _FakeToolRunSpec)
    monkeypatch.setattr(decomp_schema, "DecompositionSpec", _FakeDecompositionSpec)
    monkeypatch.setattr(compute_module, "compute_stimulus_decomposition", _fake_compute)

    from mousehash.blahml.executor import run_from_resolved_spec

    _, resolved = _resolved_pca()
    result = run_from_resolved_spec(resolved)

    assert result["tool_id"] == "run_pca"
    assert result["tool_run_spec_id"] == resolved.tool_run_spec_id
    assert result["decomposition_spec_id"] == f"blahml_{resolved.tool_run_spec_id}"

    # ToolRunSpec audit row was written exactly once.
    assert len(inserted_run_specs) == 1
    audit = inserted_run_specs[0]
    assert audit["tool_id"] == "run_pca"
    assert audit["manifest_sha256"] == resolved.manifest_sha256
    assert audit["tool_run_spec_id"] == resolved.tool_run_spec_id

    # DecompositionSpec bridge row mapped n_components and normalize_input.
    assert len(inserted_decomp_specs) == 1
    bridge = inserted_decomp_specs[0]
    assert bridge["method"] == "pca"
    assert bridge["input_kind"] == "logits"
    assert bridge["n_components"] == 5
    assert bridge["normalize_input"] is True
    assert bridge["mode"] == "train_only"

    # The deterministic compute orchestrator was invoked with the right keys.
    assert len(compute_calls) == 1
    call = compute_calls[0]
    assert call["scene_set_id"] == "allen_natural_scenes_v1"
    assert call["representation_spec_id"] == "vit_b16_imagenet_cpu"
    assert call["rule_id"] == "imagenet_top1_leq_397"
    assert call["decomposition_spec_id"] == f"blahml_{resolved.tool_run_spec_id}"


def test_executor_emits_warning_for_full_dataset_fit_scope(monkeypatch) -> None:
    class _Noop:
        @staticmethod
        def insert1(row, skip_duplicates=False):
            pass

    import mousehash.schema.blahml as blahml_schema
    import mousehash.schema.decompositions as decomp_schema
    import mousehash.tools.decompositions.compute as compute_module

    monkeypatch.setattr(blahml_schema, "ToolRunSpec", _Noop)
    monkeypatch.setattr(decomp_schema, "DecompositionSpec", _Noop)
    monkeypatch.setattr(
        compute_module, "compute_stimulus_decomposition", lambda **kw: {}
    )

    from mousehash.blahml.executor import run_from_resolved_spec

    reg = ManifestRegistry()
    registered = reg.by_id("run_pca")
    resolved = build_resolved_spec(
        registered.manifest,
        {
            "input_artifact": {
                "scene_set_id": "s",
                "representation_spec_id": "r",
                "rule_id": "u",
            },
            "fit_scope": "exploratory_full_dataset",
        },
        manifest_sha256=registered.sha256,
    )
    result = run_from_resolved_spec(resolved)
    assert any(
        w["check"] == "leakage_warning_if_fit_scope_full_dataset"
        for w in result["warnings"]
    )
