"""Load a `BlahManifest` from a YAML file packaged under `blahml/manifests/`."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml

from mousehash.blahml.models import BlahManifest


def load_manifest(name: str) -> BlahManifest:
    """Read `blahml/manifests/<name>.yaml` from the package and validate it.

    Args:
        name: manifest id with or without `.yaml` extension.

    Raises:
        FileNotFoundError: if no such manifest exists in the package.
    """
    fname = name if name.endswith(".yaml") else f"{name}.yaml"
    resource = files("mousehash.blahml.manifests").joinpath(fname)
    if not resource.is_file():
        raise FileNotFoundError(f"BlahML manifest {fname!r} not found under mousehash.blahml.manifests")
    text = resource.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return BlahManifest.model_validate(data)


def load_manifest_from_path(path: Path | str) -> BlahManifest:
    """Read a manifest from an explicit filesystem path (useful in tests)."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return BlahManifest.model_validate(data)
