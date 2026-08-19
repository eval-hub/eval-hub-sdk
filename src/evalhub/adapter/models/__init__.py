"""Adapter models for the simplified BYOF SDK."""

from ...models.api import MessageOrigin
from .adapter import FrameworkAdapter
from .cards import (
    CapabilityEvalEntry,
    EnvironmentCardMetadata,
    EvalCardMetadata,
    SafetyEvalEntry,
)
from .job import (
    ErrorInfo,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactResult,
    OCIArtifactSpec,
)

__all__ = [
    # Core adapter
    "FrameworkAdapter",
    # Card metadata
    "CapabilityEvalEntry",
    "SafetyEvalEntry",
    "EvalCardMetadata",
    "EnvironmentCardMetadata",
    # Job models
    "JobSpec",
    "JobCallbacks",
    "JobResults",
    "JobStatusUpdate",
    "JobPhase",
    "ErrorInfo",
    "MessageInfo",
    "MessageOrigin",
    # OCI models
    "OCIArtifactSpec",
    "OCIArtifactResult",
]
