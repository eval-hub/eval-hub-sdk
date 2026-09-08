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

"""Observation model for ATIF trajectories."""

from pydantic import BaseModel, Field

from .observation_result import ObservationResult


class Observation(BaseModel):
    """Environment feedback/result after actions or system events."""

    results: list[ObservationResult] = Field(
        default=...,
        description="Array of result objects from tool calls or actions",
    )

    model_config = {"extra": "forbid"}
