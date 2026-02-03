"""Adapter models for the simplified BYOF SDK."""

from .adapter import FrameworkAdapter
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
    # Job models
    "JobSpec",
    "JobCallbacks",
    "JobResults",
    "JobStatusUpdate",
    "JobPhase",
    "ErrorInfo",
    "MessageInfo",
    # OCI models
    "OCIArtifactSpec",
    "OCIArtifactResult",
]
