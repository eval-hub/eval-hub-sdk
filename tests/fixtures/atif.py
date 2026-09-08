"""Canonical ATIF trajectory fixtures for use across tests.

Three realistic traces that mirror actual agent runs:
- minimal_trajectory: a single user step, no tool calls
- agent_trajectory: a 4-step Claude Code-style run with tool calls and observations
- subagent_trajectory: a parent trajectory that embeds a subagent trajectory
"""

import pytest
from evalhub.models.atif import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)


@pytest.fixture
def minimal_trajectory() -> Trajectory:
    """One user step — simplest valid ATIF document."""
    return Trajectory(
        schema_version="ATIF-v1.8",
        session_id="sess-minimal-001",
        trajectory_id="traj-minimal-001",
        agent=Agent(name="test-agent", version="0.1.0"),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="Hello, fix auth.py",
            )
        ],
    )


@pytest.fixture
def agent_trajectory() -> Trajectory:
    """Realistic 4-step agent run mirroring a Claude Code trajectory.

    Step 1: system prompt
    Step 2: user task
    Step 3: agent reads a file (tool call + observation)
    Step 4: agent edits the file (tool call + observation referencing step 3's call)
    """
    return Trajectory(
        schema_version="ATIF-v1.8",
        session_id="sess-code-fix-abc",
        trajectory_id="traj-code-fix-abc",
        agent=Agent(
            name="claude-code",
            version="1.0.0",
            model_name="claude-sonnet-4-6",
            tool_definitions=[
                {"name": "Read", "description": "Read a file"},
                {"name": "Edit", "description": "Edit a file"},
            ],
        ),
        steps=[
            Step(
                step_id=1,
                source="system",
                message="You are a helpful coding assistant.",
            ),
            Step(
                step_id=2,
                source="user",
                message="Fix the failing test in src/auth.py — the token expiry check is wrong.",
            ),
            Step(
                step_id=3,
                source="agent",
                message="I'll read the file first to understand the current implementation.",
                timestamp="2026-09-08T10:00:00Z",
                tool_calls=[
                    ToolCall(
                        tool_call_id="tc-read-001",
                        function_name="Read",
                        arguments={"file_path": "/src/auth.py"},
                    )
                ],
                observation=Observation(
                    results=[
                        ObservationResult(
                            source_call_id="tc-read-001",
                            content="def check_token_expiry(token):\n    return token.exp < time.time()\n",
                        )
                    ]
                ),
                metrics=Metrics(
                    prompt_tokens=1200,
                    completion_tokens=85,
                    cached_tokens=800,
                    cost_usd=0.0018,
                ),
            ),
            Step(
                step_id=4,
                source="agent",
                message="Found the bug: the comparison is inverted. Applying fix.",
                timestamp="2026-09-08T10:00:03Z",
                reasoning_content="The token is expired when exp < now, so the condition should return True when exp < time.time(). The current code returns True for valid tokens. Need to negate.",
                tool_calls=[
                    ToolCall(
                        tool_call_id="tc-edit-001",
                        function_name="Edit",
                        arguments={
                            "file_path": "/src/auth.py",
                            "old_string": "return token.exp < time.time()",
                            "new_string": "return token.exp <= time.time()",
                        },
                    )
                ],
                observation=Observation(
                    results=[
                        ObservationResult(
                            source_call_id="tc-edit-001",
                            content="File updated successfully.",
                        )
                    ]
                ),
                metrics=Metrics(
                    prompt_tokens=1800,
                    completion_tokens=140,
                    cached_tokens=1200,
                    cost_usd=0.0031,
                ),
            ),
        ],
        final_metrics=FinalMetrics(
            total_prompt_tokens=3000,
            total_completion_tokens=225,
            total_cached_tokens=2000,
            total_cost_usd=0.0049,
            total_steps=4,
        ),
    )


@pytest.fixture
def subagent_trajectory() -> Trajectory:
    """Parent trajectory that delegates to a subagent via embedded form.

    The parent agent dispatches a subagent (e.g. a specialised linter agent).
    The subagent trajectory is embedded in subagent_trajectories and referenced
    via SubagentTrajectoryRef in the observation result.
    """
    subagent = Trajectory(
        schema_version="ATIF-v1.8",
        trajectory_id="traj-linter-sub-001",
        agent=Agent(
            name="linter-agent", version="0.3.0", model_name="claude-haiku-4-5-20251001"
        ),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="Run ruff on the changed files and report issues.",
            ),
            Step(
                step_id=2,
                source="agent",
                message="Running ruff check.",
                tool_calls=[
                    ToolCall(
                        tool_call_id="tc-lint-001",
                        function_name="Bash",
                        arguments={"command": "ruff check src/auth.py"},
                    )
                ],
                observation=Observation(
                    results=[
                        ObservationResult(
                            source_call_id="tc-lint-001",
                            content="All checks passed.",
                        )
                    ]
                ),
                metrics=Metrics(
                    prompt_tokens=400, completion_tokens=30, cost_usd=0.0003
                ),
            ),
        ],
    )

    return Trajectory(
        schema_version="ATIF-v1.8",
        session_id="sess-orchestrator-xyz",
        trajectory_id="traj-orchestrator-xyz",
        agent=Agent(name="orchestrator-agent", version="1.0.0"),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="Fix auth.py and lint the result.",
            ),
            Step(
                step_id=2,
                source="agent",
                message="Delegating lint check to linter subagent.",
                tool_calls=[
                    ToolCall(
                        tool_call_id="tc-delegate-001",
                        function_name="SpawnSubagent",
                        arguments={"agent": "linter-agent", "task": "lint auth.py"},
                    )
                ],
                observation=Observation(
                    results=[
                        ObservationResult(
                            source_call_id="tc-delegate-001",
                            content="Subagent completed.",
                            subagent_trajectory_ref=[
                                SubagentTrajectoryRef(
                                    trajectory_id="traj-linter-sub-001",
                                )
                            ],
                        )
                    ]
                ),
                metrics=Metrics(
                    prompt_tokens=600, completion_tokens=50, cost_usd=0.0007
                ),
            ),
        ],
        subagent_trajectories=[subagent],
    )
