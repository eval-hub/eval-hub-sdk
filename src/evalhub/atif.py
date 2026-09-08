"""Convenience shim: exposes ATIF models at the `evalhub.atif` namespace.

Equivalent to importing from `evalhub.models.atif` directly.
"""

from evalhub.models.atif import (  # noqa: F401
    Agent,
    AudioSource,
    ContentPart,
    FinalMetrics,
    ImageSource,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)

__all__ = [
    "Agent",
    "AudioSource",
    "ContentPart",
    "FinalMetrics",
    "ImageSource",
    "Metrics",
    "Observation",
    "ObservationResult",
    "Step",
    "SubagentTrajectoryRef",
    "ToolCall",
    "Trajectory",
]
