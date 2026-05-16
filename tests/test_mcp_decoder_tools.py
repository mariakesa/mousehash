"""Tests for mcp/decoder_tools.py — decode_animate_inanimate + decoder_next_question."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mousehash.artifacts.io import save_json, save_npy
from mousehash.core.analysis_view import AnalysisView, AnalysisViewKind
from mousehash.mcp.decoder_tools import decode_animate_inanimate, decoder_next_question


def _write_view(art_dir: Path, view: AnalysisView) -> None:
    """Write a view.json inside the artifact dir so find_view_by_id can pick it up."""
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "view.json").write_text(view.model_dump_json(), encoding="utf-8")
    (art_dir / "spec.json").write_text(json.dumps({"fake": True}), encoding="utf-8")
    save_json(art_dir / "summary.json", {"fake": True})


def _seed_neural_and_labels_views(data_root: Path) -> tuple[str, str]:
    """Plant a neural view + a labels view under the artifact cache."""
    art_root = data_root / "artifacts"

    n_neurons, n_images = 6, 30
    rng = np.random.default_rng(0)
    y = np.tile(np.array([0, 1]), n_images // 2).astype(np.int8)
    neural = rng.normal(scale=0.1, size=(n_neurons, n_images)).astype(np.float32)
    neural[0] = y.astype(np.float32) + rng.normal(scale=0.1, size=n_images).astype(np.float32)

    neural_dir = art_root / "neural_responses" / "scope" / "abcd"
    save_npy(neural_dir / "event_probabilities.npy", neural)
    neural_view = AnalysisView.new(
        kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
        manifest_id="mf_test", shape=[n_neurons, n_images],
        axes={"observations": "neurons_across_sessions", "features": "natural_scene_images"},
        source_roles=["neural_data", "stimuli"],
        transformation_lineage=["fake_neural"],
        artifact_path=str(neural_dir),
    )
    _write_view(neural_dir, neural_view)

    labels_dir = art_root / "representations" / "scope" / "wxyz"
    save_npy(labels_dir / "animate_inanimate.npy", y)
    labels_view = AnalysisView.new(
        kind=AnalysisViewKind.OBSERVATION_BY_FEATURE,
        manifest_id="mf_test", shape=[n_images, 1000],
        axes={"observations": "stimulus_presentations", "features": "imagenet_classifier_output"},
        source_roles=["stimuli"],
        transformation_lineage=["fake_vit"],
        artifact_path=str(labels_dir),
    )
    _write_view(labels_dir, labels_view)
    return neural_view.view_id, labels_view.view_id


class TestDecoderNextQuestion:
    def test_starts_with_neural_view_question(self):
        result = decoder_next_question("{}")
        assert result["resolved"] is False
        assert result["manifest_id"] == "allen_logistic_decoder"
        assert result["question"]["name"] == "neural_view_id"
        assert "Which neural view" in result["question"]["ask"]
        assert result["question"]["section"] == "inputs"

    def test_after_inputs_asks_split_strategy(self):
        answers = {"neural_view_id": "view_a", "labels_view_id": "view_b"}
        result = decoder_next_question(json.dumps(answers))
        assert result["resolved"] is False
        assert result["question"]["name"] == "split_strategy"
        assert set(result["question"]["choices"]) == {"loo", "stratified_kfold", "holdout"}

    def test_resolves_with_required_and_intent_answers(self):
        answers = {
            "neural_view_id": "view_a", "labels_view_id": "view_b",
            "split_strategy": "stratified_kfold",
        }
        result = decoder_next_question(json.dumps(answers))
        assert result["resolved"] is True
        assert result["question"] is None
        assert result["collected_answers"] == answers

    def test_invalid_json_returns_structured_error(self):
        result = decoder_next_question("not json")
        assert result["type"] == "JSONDecodeError"

    def test_non_object_json_returns_structured_error(self):
        result = decoder_next_question("[1, 2, 3]")
        assert result["type"] == "JSONDecodeError"


class TestDecodeAnimateInanimate:
    def test_returns_view_id_and_summary(self, data_root_tmp: Path):
        nv_id, lv_id = _seed_neural_and_labels_views(data_root_tmp)
        result = decode_animate_inanimate(
            neural_view_id=nv_id, labels_view_id=lv_id,
            split_strategy="stratified_kfold", k_folds=5,
        )
        assert "error" not in result, result
        assert result["view_id"].startswith("view_")
        assert Path(result["artifact_path"]).exists()
        assert result["summary"]["cv_balanced_accuracy"] >= 0.7
        assert result["from_cache"] is False

    def test_idempotent_cache_hit(self, data_root_tmp: Path):
        nv_id, lv_id = _seed_neural_and_labels_views(data_root_tmp)
        r1 = decode_animate_inanimate(neural_view_id=nv_id, labels_view_id=lv_id)
        r2 = decode_animate_inanimate(neural_view_id=nv_id, labels_view_id=lv_id)
        assert r1["from_cache"] is False
        assert r2["from_cache"] is True
        assert r1["view_id"] == r2["view_id"]

    def test_unknown_view_id_returns_structured_error(self, data_root_tmp: Path):
        result = decode_animate_inanimate(neural_view_id="view_nope", labels_view_id="view_also_nope")
        assert result["type"] == "ViewNotFoundError"

    def test_custom_C_grid_parses(self, data_root_tmp: Path):
        nv_id, lv_id = _seed_neural_and_labels_views(data_root_tmp)
        result = decode_animate_inanimate(
            neural_view_id=nv_id, labels_view_id=lv_id,
            split_strategy="stratified_kfold", k_folds=3,
            search_hyperparams=True, C_grid="0.1,1,10",
        )
        assert "error" not in result
        assert result["summary"]["search_hyperparams"] is True
