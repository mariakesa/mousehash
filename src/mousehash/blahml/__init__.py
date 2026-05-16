"""BlahML: tool-spec manifests + the question engine that resolves them.

A `BlahManifest` declares the inputs/parameters a tool needs plus a
`BlahQuestion` per parameter (`ask`/`explain`/optional `examples`). The
question engine walks the manifest given the answers collected so far and
returns the next pending question — agents can call it in a loop, asking the
user once per turn, until the spec is fully resolved.

This is a minimal MVP port of the BlahML core (models + question_engine +
yaml loader). The DataJoint/audit-trail pieces from the prior code base are
out of scope until the persistence story lands.
"""

from mousehash.blahml.loader import load_manifest
from mousehash.blahml.models import (
    BlahManifest,
    BlahParameter,
    BlahQuestion,
    BlahRange,
    ParamType,
)
from mousehash.blahml.question_engine import (
    SCIENTIFIC_INTENT_PARAMS,
    PendingQuestion,
    next_question,
)

__all__ = [
    "BlahManifest",
    "BlahParameter",
    "BlahQuestion",
    "BlahRange",
    "ParamType",
    "PendingQuestion",
    "SCIENTIFIC_INTENT_PARAMS",
    "load_manifest",
    "next_question",
]
