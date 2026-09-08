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

"""Subagent trajectory reference model for ATIF trajectories."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class SubagentTrajectoryRef(BaseModel):
    """Reference to a delegated subagent trajectory.

    A subagent reference is resolved by one of two mechanisms:

    1. **Embedded form** — set `trajectory_id` to match the
       `Trajectory.trajectory_id` of an entry in the parent's
       `subagent_trajectories` array.
    2. **File-ref form** — set `trajectory_path` to the location
       (file path, S3 URL, etc.) of an external trajectory file.

    These two mechanisms are the only resolution keys. `session_id`, when
    present on the ref, is **informational only**: it records the run
    identity of the delegated subagent for debug / correlation / cross-
    trajectory search purposes, and MUST NOT be used as a matching key
    (it is run-scoped and MAY collide across siblings — see
    `Trajectory.session_id`). A ref therefore MUST set at least one of
    `trajectory_id` or `trajectory_path`; `session_id` alone is not a
    resolvable reference.
    """

    trajectory_id: str | None = Field(
        default=None,
        description=(
            "Canonical identifier of the delegated subagent trajectory. "
            "Matches `Trajectory.trajectory_id` of an entry in the parent's "
            "`subagent_trajectories` array and is the resolution key for "
            "embedded references. Added in ATIF-v1.7 to provide a document-"
            "unique matching key without overloading `session_id`."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Run identity of the delegated subagent trajectory. Informational "
            "only: recorded so consumers can correlate this ref back to the "
            "subagent's run for debug / search / display purposes. Run-scoped "
            "(see `Trajectory.session_id`) and therefore NOT a valid "
            "resolution key — consumers MUST resolve via `trajectory_id` "
            "(embedded) or `trajectory_path` (external file)."
        ),
    )
    trajectory_path: str | None = Field(
        default=None,
        description=(
            "Location of the complete subagent trajectory as an external "
            "file (file path, S3 URL, database reference, etc.). Resolution "
            "key for file-ref references. When both `trajectory_id` and "
            "`trajectory_path` are set, consumers MAY choose either; "
            "typically `trajectory_id` is preferred when the embedded "
            "trajectory is available in-memory."
        ),
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom metadata about the subagent execution",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_is_resolvable(self) -> "SubagentTrajectoryRef":
        """A ref must be resolvable: set `trajectory_id` or `trajectory_path`.

        `session_id` alone is not sufficient because it is run-scoped and
        MAY collide across siblings (see `Trajectory.session_id`), so it
        cannot unambiguously identify which subagent trajectory a ref
        points at.
        """
        if (
            not (self.trajectory_id or "").strip()
            and not (self.trajectory_path or "").strip()
        ):
            raise ValueError(
                "SubagentTrajectoryRef must be resolvable: set either "
                "`trajectory_id` (for embedded references) or "
                "`trajectory_path` (for external-file references). "
                "`session_id` alone is not a resolution key — it is "
                "run-scoped and may collide across siblings."
            )
        return self
