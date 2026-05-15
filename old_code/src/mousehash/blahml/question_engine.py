from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mousehash.blahml.models import BlahManifest, BlahParameter

# Parameter names whose values change scientific interpretation. We ask
# about these even when they have a default — silently applying a default
# for a leakage-affecting choice is the failure mode BlahML is meant to
# prevent.
SCIENTIFIC_INTENT_PARAMS = {"fit_scope", "split_unit"}


@dataclass(frozen=True)
class PendingQuestion:
    """A manifest-defined question that needs a user answer, plus a stable
    pointer to the parameter slot it is filling."""

    parameter: BlahParameter
    section: str  # "inputs" or "parameters"

    @property
    def name(self) -> str:
        return self.parameter.name


def next_question(
    manifest: BlahManifest,
    answers: dict[str, Any],
    *,
    confirm_defaults_for: set[str] | None = None,
) -> PendingQuestion | None:
    """Return the next question to ask, or None if the spec is fully resolved.

    Priority order (per design doc §9):
      1. Required input artifacts.
      2. Required parameters with no default.
      3. Optional scientific-intent parameters (fit_scope, split_unit, ...).
      4. Optional parameters the caller has flagged for explicit confirmation.

    Optional parameters with defaults that are not in those buckets are
    silently filled by ``spec_builder`` — the agent should mention the default
    in its reply, but we don't pause the dialogue to ask.
    """
    confirm = confirm_defaults_for or set()

    for p in manifest.inputs:
        if p.required and p.name not in answers:
            return PendingQuestion(p, "inputs")

    for p in manifest.parameters:
        if p.required and p.name not in answers:
            return PendingQuestion(p, "parameters")

    for p in manifest.parameters:
        if p.name in answers:
            continue
        if p.name in SCIENTIFIC_INTENT_PARAMS:
            return PendingQuestion(p, "parameters")

    for p in manifest.parameters:
        if p.name in answers:
            continue
        if p.name in confirm:
            return PendingQuestion(p, "parameters")

    return None
