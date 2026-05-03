from __future__ import annotations

from pathlib import Path

import yaml

from mousehash.blahml.models import BlahManifest


def load_manifest(path: Path) -> BlahManifest:
    """Parse a single YAML manifest into a validated ``BlahManifest``."""
    raw = yaml.safe_load(Path(path).read_text())
    return BlahManifest.model_validate(raw)
