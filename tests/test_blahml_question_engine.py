from __future__ import annotations

from mousehash.blahml.question_engine import next_question
from mousehash.blahml.registry import ManifestRegistry
from mousehash.blahml.spec_builder import build_resolved_spec


def _pca_manifest():
    return ManifestRegistry().by_id("run_pca").manifest


def test_first_question_is_required_input() -> None:
    m = _pca_manifest()
    pending = next_question(m, answers={})
    assert pending is not None
    assert pending.section == "inputs"
    assert pending.parameter.name == "input_artifact"
    assert pending.parameter.required is True


def test_after_input_filled_asks_scientific_intent() -> None:
    """fit_scope is in SCIENTIFIC_INTENT_PARAMS, so it gets surfaced before
    the dialogue silently applies its default."""
    m = _pca_manifest()
    pending = next_question(m, answers={"input_artifact": {"x": 1}})
    assert pending is not None
    assert pending.parameter.name == "fit_scope"


def test_dialogue_resolves_after_intent_answered() -> None:
    m = _pca_manifest()
    answers = {
        "input_artifact": {
            "scene_set_id": "allen_natural_scenes_v1",
            "representation_spec_id": "vit_b16_imagenet_cpu",
            "rule_id": "imagenet_top1_leq_397",
        },
        "fit_scope": "exploratory_full_dataset",
    }
    assert next_question(m, answers) is None


def test_spec_builder_applies_defaults_for_unanswered_optional() -> None:
    m = _pca_manifest()
    answers = {
        "input_artifact": {
            "scene_set_id": "allen_natural_scenes_v1",
            "representation_spec_id": "vit_b16_imagenet_cpu",
            "rule_id": "imagenet_top1_leq_397",
        },
        "fit_scope": "exploratory_full_dataset",
    }
    spec = build_resolved_spec(
        m,
        answers,
        manifest_sha256="0" * 64,
    )
    assert spec.parameters["n_components"] == 10
    assert spec.parameters["normalize_input"] is True
    assert spec.parameters["fit_scope"] == "exploratory_full_dataset"
    assert spec.tool_id == "run_pca"
    assert len(spec.tool_run_spec_id) == 16


def test_spec_builder_coerces_string_to_int() -> None:
    m = _pca_manifest()
    answers = {
        "input_artifact": {
            "scene_set_id": "s",
            "representation_spec_id": "r",
            "rule_id": "u",
        },
        "n_components": "25",            # arrives as string from agent
    }
    spec = build_resolved_spec(m, answers, manifest_sha256="0" * 64)
    assert spec.parameters["n_components"] == 25


def test_identical_inputs_produce_identical_spec_id() -> None:
    """Idempotency contract: re-resolving the same dialogue must yield the
    same tool_run_spec_id so the DataJoint insert is a no-op."""
    m = _pca_manifest()
    answers = {
        "input_artifact": {"scene_set_id": "a", "representation_spec_id": "b", "rule_id": "c"},
    }
    a = build_resolved_spec(m, answers, manifest_sha256="0" * 64)
    b = build_resolved_spec(m, answers, manifest_sha256="0" * 64)
    assert a.tool_run_spec_id == b.tool_run_spec_id


def test_different_manifest_sha_produces_different_spec_id() -> None:
    """Manifest drift must invalidate the cached run id."""
    m = _pca_manifest()
    answers = {
        "input_artifact": {"scene_set_id": "a", "representation_spec_id": "b", "rule_id": "c"},
    }
    a = build_resolved_spec(m, answers, manifest_sha256="0" * 64)
    b = build_resolved_spec(m, answers, manifest_sha256="1" * 64)
    assert a.tool_run_spec_id != b.tool_run_spec_id
