"""Tests for the BlahML core: model parsing + question-engine priority order."""

from __future__ import annotations

import pytest

from mousehash.blahml import (
    SCIENTIFIC_INTENT_PARAMS,
    BlahManifest,
    BlahParameter,
    BlahQuestion,
    load_manifest,
    next_question,
)


def _make_test_manifest() -> BlahManifest:
    """Tiny synthetic manifest exercising all four priority buckets."""
    return BlahManifest(
        id="test_decoder",
        version="0.0.1",
        display_name="Test",
        workflow_family="testing",
        priority="MVP",
        description={"short": "test", "long": "test"},
        tool_binding={"python_function": "x.y"},
        inputs=[
            BlahParameter(
                name="data_view_id", type="artifact_ref", required=True,
                accepted_artifact_types=["observation_by_feature"],
                question=BlahQuestion(ask="Which view?", explain="..."),
            ),
        ],
        parameters=[
            BlahParameter(
                name="k_folds", type="integer", required=False, default=5,
                question=BlahQuestion(ask="How many folds?", explain="..."),
            ),
            BlahParameter(
                name="penalty", type="enum", required=False, default="l2", choices=["l1", "l2"],
                question=BlahQuestion(ask="Regularization?", explain="..."),
            ),
            BlahParameter(
                name="random_state", type="integer", required=True,
                question=BlahQuestion(ask="Seed?", explain="..."),
            ),
            BlahParameter(
                name="split_strategy", type="enum", required=False, default="kfold",
                choices=["loo", "kfold"],
                question=BlahQuestion(ask="Which CV?", explain="..."),
            ),
        ],
    )


class TestManifestParse:
    def test_packaged_manifest_loads(self):
        m = load_manifest("allen_logistic_decoder")
        assert m.id == "allen_logistic_decoder"
        assert m.priority == "MVP"
        assert {p.name for p in m.inputs} == {"neural_view_id", "labels_view_id"}
        names = {p.name for p in m.parameters}
        for required in {"split_strategy", "k_folds", "penalty", "class_weight",
                         "search_hyperparams", "C_grid", "random_state", "n_permutations"}:
            assert required in names
        # split_strategy is required + an enum
        split_param = next(p for p in m.parameters if p.name == "split_strategy")
        assert split_param.required is True
        assert split_param.type == "enum"
        assert set(split_param.choices) == {"loo", "stratified_kfold", "holdout"}

    def test_missing_manifest_raises(self):
        with pytest.raises(FileNotFoundError):
            load_manifest("not_a_real_manifest")


class TestNextQuestion:
    def test_starts_with_required_input(self):
        m = _make_test_manifest()
        q = next_question(m, {})
        assert q is not None
        assert q.name == "data_view_id"
        assert q.section == "inputs"

    def test_next_after_input_is_required_parameter(self):
        m = _make_test_manifest()
        q = next_question(m, {"data_view_id": "view_x"})
        assert q is not None
        assert q.name == "random_state"
        assert q.section == "parameters"

    def test_surfaces_scientific_intent_after_required(self):
        m = _make_test_manifest()
        q = next_question(m, {"data_view_id": "view_x", "random_state": 0})
        assert q is not None
        # split_strategy is in SCIENTIFIC_INTENT_PARAMS, asked despite default
        assert q.name == "split_strategy"
        assert "split_strategy" in SCIENTIFIC_INTENT_PARAMS

    def test_resolved_when_all_required_and_intent_filled(self):
        m = _make_test_manifest()
        answers = {"data_view_id": "view_x", "random_state": 0, "split_strategy": "kfold"}
        assert next_question(m, answers) is None

    def test_confirm_defaults_for_surfaces_extra(self):
        m = _make_test_manifest()
        answers = {"data_view_id": "view_x", "random_state": 0, "split_strategy": "kfold"}
        # Without confirm_defaults_for, we're done.
        assert next_question(m, answers) is None
        # With it, the requested key surfaces.
        q = next_question(m, answers, confirm_defaults_for={"penalty"})
        assert q is not None
        assert q.name == "penalty"

    def test_decoder_manifest_first_question_is_neural_view(self):
        """End-to-end against the real packaged manifest."""
        m = load_manifest("allen_logistic_decoder")
        q = next_question(m, {})
        assert q.name == "neural_view_id"

    def test_decoder_manifest_resolves_with_minimal_answers(self):
        m = load_manifest("allen_logistic_decoder")
        answers = {
            "neural_view_id": "view_a",
            "labels_view_id": "view_b",
            "split_strategy": "stratified_kfold",
        }
        assert next_question(m, answers) is None
