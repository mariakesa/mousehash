from __future__ import annotations

import pytest

from mousehash.blahml.loader import load_manifest
from mousehash.blahml.registry import ManifestRegistry
from mousehash.blahml.validator import (
    ManifestValidationError,
    validate_manifest_structure,
)


def test_registry_loads_pca_and_nmf() -> None:
    reg = ManifestRegistry()
    ids = reg.all_ids()
    assert "run_pca" in ids
    assert "run_nmf" in ids


def test_pca_manifest_shape() -> None:
    reg = ManifestRegistry()
    pca = reg.by_id("run_pca")
    m = pca.manifest

    assert m.id == "run_pca"
    assert m.priority == "MVP"
    assert m.tool_binding["python_function"] == "mousehash.tools.decompositions.pca.run_pca"

    param_names = {p.name for p in m.parameters}
    assert {"n_components", "normalize_input", "fit_scope"} <= param_names

    n_components = next(p for p in m.parameters if p.name == "n_components")
    assert n_components.default == 10
    assert n_components.range is not None
    assert n_components.range.min == 2

    fit_scope = next(p for p in m.parameters if p.name == "fit_scope")
    assert fit_scope.choices == ["exploratory_full_dataset", "train_only"]
    assert fit_scope.question.ask  # non-empty


def test_nmf_param_names_match_tool_kwargs() -> None:
    """The names in the manifest must match run_nmf's actual kwargs so the
    bridge in executor.py can pass them through unchanged."""
    import inspect

    from mousehash.tools.decompositions.nmf import run_nmf

    sig = inspect.signature(run_nmf)
    tool_kwargs = set(sig.parameters) - {"probabilities"}

    reg = ManifestRegistry()
    nmf = reg.by_id("run_nmf").manifest
    manifest_param_names = {p.name for p in nmf.parameters}

    # Every NMF manifest parameter must correspond to a real kwarg.
    assert manifest_param_names <= tool_kwargs, (
        f"Manifest names not in run_nmf signature: "
        f"{manifest_param_names - tool_kwargs}"
    )


def test_validator_rejects_enum_without_choices(tmp_path) -> None:
    bad_yaml = """
id: bad_tool
version: 0.0.1
display_name: Bad
workflow_family: test
priority: experimental
description:
  short: bad
  long: bad
tool_binding:
  python_function: mousehash.blahml.models.BlahManifest
inputs: []
parameters:
  - name: choice
    type: enum
    required: true
    question:
      ask: pick one
      explain: please pick
"""
    p = tmp_path / "bad.yaml"
    p.write_text(bad_yaml)
    m = load_manifest(p)
    with pytest.raises(ManifestValidationError, match="enum parameter has no choices"):
        validate_manifest_structure(m)


def test_validator_rejects_unknown_check(tmp_path) -> None:
    bad_yaml = """
id: bad_tool
version: 0.0.1
display_name: Bad
workflow_family: test
priority: experimental
description:
  short: bad
  long: bad
tool_binding:
  python_function: mousehash.blahml.models.BlahManifest
inputs: []
parameters: []
validation:
  checks:
    - this_check_does_not_exist
"""
    p = tmp_path / "bad.yaml"
    p.write_text(bad_yaml)
    m = load_manifest(p)
    with pytest.raises(ManifestValidationError, match="unknown validation check"):
        validate_manifest_structure(m)
