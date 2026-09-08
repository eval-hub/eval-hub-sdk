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

"""Observation result model for ATIF trajectories."""

from typing import Any

from pydantic import BaseModel, Field

from .content import ContentPart
from .subagent_trajectory_ref import SubagentTrajectoryRef


class ObservationResult(BaseModel):
    """A single result within an observation."""

    source_call_id: str | None = Field(
        default=None,
        description=(
            "The `tool_call_id` from the _tool_calls_ array in _StepObject_ that this "
            "result corresponds to. If null or omitted, the result comes from an "
            "action that doesn't use the standard tool calling format (e.g., agent "
            "actions without tool calls or system-initiated operations)."
        ),
    )
    content: str | list[ContentPart] | None = Field(
        default=None,
        description=(
            "The output or result from the tool execution. String for text-only "
            "content, or array of ContentPart for multimodal content (added in ATIF-v1.6)."
        ),
    )
    subagent_trajectory_ref: list[SubagentTrajectoryRef] | None = Field(
        default=None,
        description="Array of references to delegated subagent trajectories",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Custom observation-result-level metadata (e.g., confidence score, "
            "retrieval score, source document ID). Added in ATIF-v1.7."
        ),
    )

    model_config = {"extra": "forbid"}
