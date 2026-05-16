"""Question engine: given a manifest + answers-so-far, return the next ask.

The agent calls `next_question(manifest, answers)` once per turn:
  - if it returns a `PendingQuestion`, present its `ask`/`explain` to the
    user, collect a value, and add it to `answers` under `parameter.name`;
  - if it returns `None`, every required + scientific-intent parameter has
    been resolved and the tool can be invoked with the collected answers.

Parameters in `SCIENTIFIC_INTENT_PARAMS` are surfaced even when they carry
defaults — silently applying a default for a choice that changes the
scientific interpretation is exactly the failure mode BlahML is here to
prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mousehash.blahml.models import BlahManifest, BlahParameter

SCIENTIFIC_INTENT_PARAMS = {"fit_scope", "split_unit", "split_strategy"}


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
    """Return the next question to ask, or `None` if the spec is fully resolved.

    Priority order:
      1. Required input artifacts.
      2. Required parameters with no default.
      3. Optional scientific-intent parameters (`SCIENTIFIC_INTENT_PARAMS`).
      4. Optional parameters the caller has flagged for explicit confirmation.

    Optional parameters with defaults outside those buckets are silently
    filled by the caller — they may still surface in a report, but we don't
    pause the dialogue to ask.
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
