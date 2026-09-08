# Copyright 2025 The Harbor Project Authors
# Copyright 2025 Alex Shaw
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: This file contains code adapted from Harbor
# (https://github.com/harbor-framework/harbor) and has been modified
# to isolate Pydantic models as a standalone dependency.

"""Trajectory model for ATIF (Agent Trajectory Interchange Format)."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .agent import Agent
from .final_metrics import FinalMetrics
from .step import Step


class Trajectory(BaseModel):
    """Agent Trajectory in ATIF (Agent Trajectory Interchange Format)."""

    schema_version: Literal[
        "ATIF-v1.0",
        "ATIF-v1.1",
        "ATIF-v1.2",
        "ATIF-v1.3",
        "ATIF-v1.4",
        "ATIF-v1.5",
        "ATIF-v1.6",
        "ATIF-v1.7",
        "ATIF-v1.8",
    ] = Field(
        default="ATIF-v1.8",
        description="String defining ATIF compatibility",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Identifier for the agent run this trajectory belongs to. Scoped "
            "to the run, NOT to an individual trajectory document: multiple "
            "Trajectory objects MAY share the same `session_id` when they "
            "represent the same logical run (e.g., a parent trajectory and "
            "its embedded subagents, or a trajectory and its continuation "
            "segments linked via `continued_trajectory_ref`). Therefore "
            "`session_id`s within a parent's `subagent_trajectories` array "
            "are NOT required to be unique. Use `trajectory_id` when a "
            "per-trajectory-document unique identifier is required (e.g., "
            "for `SubagentTrajectoryRef` resolution). Optional since "
            "ATIF-v1.7; producers SHOULD set this on root trajectories for "
            "run-level traceability, and MAY omit it on embedded subagents "
            "that inherit the parent's run identity."
        ),
    )
    trajectory_id: str | None = Field(
        default=None,
        description=(
            "Canonical per-trajectory-document identifier, distinct from "
            "`session_id`. Unlike `session_id` (which is run-scoped and MAY "
            "be shared), `trajectory_id` uniquely identifies THIS trajectory "
            "object. Used to resolve `SubagentTrajectoryRef` entries against "
            "the root's `subagent_trajectories` array without overloading "
            "`session_id`'s run-scoped semantics. Optional on standalone "
            "trajectories, but REQUIRED on any trajectory embedded in a "
            "parent's `subagent_trajectories` array. `trajectory_id`s within "
            "a single parent's `subagent_trajectories` array MUST be unique. "
            "Added in ATIF-v1.7."
        ),
    )
    agent: Agent = Field(
        default=...,
        description="Object specifying the agent configuration",
    )
    steps: list[Step] = Field(
        default=...,
        min_length=1,
        description="Array of step objects representing the complete interaction history",
    )
    notes: str | None = Field(
        default=None,
        description="Custom information, design notes, or explanations",
    )
    final_metrics: FinalMetrics | None = Field(
        default=None,
        description="Summary metrics for the entire trajectory",
    )
    continued_trajectory_ref: str | None = Field(
        default=None,
        description="Reference to the continuation trajectory file if this trajectory is continued in another file",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom root-level metadata",
    )
    subagent_trajectories: list["Trajectory"] | None = Field(
        default=None,
        description=(
            "Array of embedded subagent trajectories. Each element is a complete, "
            "independently-valid ATIF Trajectory with its own schema_version, "
            "agent, and step_id sequence starting at 1. Enables single-file "
            "storage of multi-agent workflows: when a "
            "SubagentTrajectoryRef.trajectory_path is null, consumers resolve "
            "the reference by matching SubagentTrajectoryRef.trajectory_id "
            "against Trajectory.trajectory_id of entries in this array. "
            "Uniqueness rules: every embedded subagent MUST set `trajectory_id`, "
            "and `trajectory_id`s within this array MUST be unique. "
            "`session_id`, by contrast, is run-scoped and MAY collide across "
            "siblings (or match the parent) when all trajectories belong to "
            "the same logical run; embedded subagents MAY also omit "
            "`session_id` entirely to inherit the parent's run identity. "
            "Added in ATIF-v1.7."
        ),
    )

    model_config = {"extra": "forbid"}

    def to_json_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        """Export trajectory to a dictionary suitable for JSON serialization.

        Args:
            exclude_none: If True, exclude fields with None values from output.

        Returns:
            Dictionary representation of the trajectory.
        """
        return self.model_dump(exclude_none=exclude_none, mode="json")

    @model_validator(mode="after")
    def validate_step_ids(self) -> "Trajectory":
        """Validate that step_ids are sequential starting from 1."""
        for i, step in enumerate(self.steps):
            expected_step_id = i + 1
            if step.step_id != expected_step_id:
                raise ValueError(
                    f"steps[{i}].step_id: expected {expected_step_id} "
                    f"(sequential from 1), got {step.step_id}"
                )
        return self

    @model_validator(mode="after")
    def validate_embedded_subagent_trajectory_ids(self) -> "Trajectory":
        """Every embedded subagent must carry a unique, non-null `trajectory_id`.

        Embedded subagents are resolved by matching
        `SubagentTrajectoryRef.trajectory_id` against
        `Trajectory.trajectory_id`, so the identifier must be present and
        unique within a parent's `subagent_trajectories` array. Note that
        no such constraint is placed on `session_id` — siblings MAY share
        a `session_id` (or omit it entirely to inherit the parent's) when
        they represent the same logical agent run.

        Path-less SubagentTrajectoryRefs (no `trajectory_path`) found in any
        step's observation results are also checked: their `trajectory_id`
        must match an entry in `subagent_trajectories`.
        """
        # Build the set of embedded trajectory IDs, validating each entry.
        seen: set[str] = set()
        for i, sub in enumerate(self.subagent_trajectories or []):
            if sub.trajectory_id is None:
                raise ValueError(
                    f"subagent_trajectories[{i}].trajectory_id is required "
                    f"for embedded subagents "
                    f"(agent.name={sub.agent.name!r}, "
                    f"session_id={sub.session_id!r})"
                )
            if sub.trajectory_id in seen:
                raise ValueError(
                    f"subagent_trajectories[{i}].trajectory_id "
                    f"{sub.trajectory_id!r} is not unique within "
                    f"subagent_trajectories"
                )
            seen.add(sub.trajectory_id)

        # Validate path-less SubagentTrajectoryRefs in observation results.
        # A ref without trajectory_path can only be resolved via the embedded
        # list; reject it if its trajectory_id is absent from that list.
        for step in self.steps:
            if step.observation is None:
                continue
            for result in step.observation.results:
                for ref in result.subagent_trajectory_ref or []:
                    if ref.trajectory_path:
                        continue
                    if ref.trajectory_id not in seen:
                        raise ValueError(
                            f"SubagentTrajectoryRef.trajectory_id "
                            f"{ref.trajectory_id!r} in step {step.step_id} "
                            f"observation is not present in "
                            f"subagent_trajectories"
                        )

        return self

    @model_validator(mode="after")
    def validate_tool_call_references(self) -> "Trajectory":
        """Validate that observation source_call_ids reference valid tool_call_ids."""
        for step in self.steps:
            # Detect duplicate tool_call_ids before building the reference set.
            if step.tool_calls:
                seen_ids: set[str] = set()
                for tc in step.tool_calls:
                    if tc.tool_call_id in seen_ids:
                        raise ValueError(
                            f"step {step.step_id} has duplicate tool_call_id "
                            f"'{tc.tool_call_id}'"
                        )
                    seen_ids.add(tc.tool_call_id)
                tool_call_ids = seen_ids
            else:
                tool_call_ids = set()

            if step.observation is None:
                continue

            # Check that source_call_ids reference valid tool_call_ids
            for result in step.observation.results:
                if result.source_call_id is not None:
                    if result.source_call_id not in tool_call_ids:
                        raise ValueError(
                            f"Observation result references source_call_id "
                            f"'{result.source_call_id}' which is not found in "
                            f"step {step.step_id}'s tool_calls"
                        )
        return self

    def has_multimodal_content(self) -> bool:
        """Check if any step contains multimodal content (images or audio).

        Returns:
            True if any step message or observation contains a media content
            part. Audio counts since ATIF-v1.8.
        """
        media_types = {"image", "audio"}
        for step in self.steps:
            if isinstance(step.message, list):
                for part in step.message:
                    if part.type in media_types:
                        return True
            if step.observation:
                for result in step.observation.results:
                    if isinstance(result.content, list):
                        for part in result.content:
                            if part.type in media_types:
                                return True
        return False
