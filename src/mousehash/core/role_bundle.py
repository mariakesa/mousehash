"""Canonical six-role bundle.

Every MouseHash dataset is described by a `RoleBundle` containing six roles:
conditions, stimuli, behavior, neural_data, time_organization, metadata.
Each role carries a list of `RoleEvidence` items so the manifest is
evidence-backed rather than a bare checklist (see architecture doc §5).

A role with an empty evidence list is treated as `absent`; otherwise its
`status` defaults to the strongest evidence reported.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RoleStatus(str, Enum):
    PRESENT = "present"
    LIKELY = "likely"
    ABSENT = "absent"
    UNKNOWN = "unknown"
    DERIVABLE = "derivable"


class RoleConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


RoleName = Literal[
    "conditions",
    "stimuli",
    "behavior",
    "neural_data",
    "time_organization",
    "metadata",
]


class RoleEvidence(BaseModel):
    """A single piece of evidence supporting (or denying) a role's presence."""

    status: RoleStatus
    confidence: RoleConfidence
    source: str = Field(description="Where this evidence came from, e.g. 'nwb', 'allensdk', 'one'.")
    path: str | None = Field(default=None, description="Optional pointer within the source, e.g. an NWB internal path.")
    notes: str | None = None


class _RoleBase(BaseModel):
    """Base for each role. The status/confidence summarize the evidence list."""

    status: RoleStatus = RoleStatus.UNKNOWN
    confidence: RoleConfidence = RoleConfidence.LOW
    evidence: list[RoleEvidence] = Field(default_factory=list)

    def is_satisfied(self) -> bool:
        return self.status in (RoleStatus.PRESENT, RoleStatus.LIKELY, RoleStatus.DERIVABLE)


class ConditionsRole(_RoleBase):
    """Trial conditions, task variables, experimental groupings."""


class StimuliRole(_RoleBase):
    """Stimulus identifiers, features, or raw stimulus content."""


class BehaviorRole(_RoleBase):
    """Choices, running, pupil, wheel, licks, reaction time, etc."""


class NeuralDataRole(_RoleBase):
    """Spike times, units, calcium traces, LFP, fMRI signals."""


class TimeOrganizationRole(_RoleBase):
    """Trial intervals, stimulus presentations, event timestamps, clocks."""


class MetadataRole(_RoleBase):
    """Subject, electrodes, brain regions, recording protocol."""


class RoleBundle(BaseModel):
    """The six canonical roles bundled together for one dataset."""

    conditions: ConditionsRole = Field(default_factory=ConditionsRole)
    stimuli: StimuliRole = Field(default_factory=StimuliRole)
    behavior: BehaviorRole = Field(default_factory=BehaviorRole)
    neural_data: NeuralDataRole = Field(default_factory=NeuralDataRole)
    time_organization: TimeOrganizationRole = Field(default_factory=TimeOrganizationRole)
    metadata: MetadataRole = Field(default_factory=MetadataRole)

    def get(self, role: RoleName) -> _RoleBase:
        return getattr(self, role)

    def satisfied_roles(self) -> list[RoleName]:
        names: list[RoleName] = [
            "conditions",
            "stimuli",
            "behavior",
            "neural_data",
            "time_organization",
            "metadata",
        ]
        return [n for n in names if self.get(n).is_satisfied()]
