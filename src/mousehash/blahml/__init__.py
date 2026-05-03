"""BlahML: a YAML manifest layer that turns natural-language research intent
into validated, reproducible MouseHash tool specs.

Manifests live in ``mousehash/blahml/manifests/`` as version-controlled YAML.
Resolved dialogues persist in DataJoint (``ToolRunSpec``).
"""
from mousehash.blahml.models import (
    BlahManifest,
    BlahParameter,
    BlahQuestion,
    BlahRange,
    ResolvedSpec,
)
from mousehash.blahml.registry import ManifestRegistry

__all__ = [
    "BlahManifest",
    "BlahParameter",
    "BlahQuestion",
    "BlahRange",
    "ResolvedSpec",
    "ManifestRegistry",
]
