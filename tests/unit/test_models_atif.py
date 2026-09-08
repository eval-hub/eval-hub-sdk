"""Tests for ATIF trajectory pydantic models (evalhub.models.atif)."""

import evalhub
import evalhub.atif as atif_shim
import pytest
from evalhub.models.atif import (
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
from pydantic import ValidationError

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AGENT = Agent(name="test-agent", version="0.1.0")
_USER_STEP = Step(step_id=1, source="user", message="Hello")


def _minimal() -> Trajectory:
    return Trajectory(agent=_AGENT, steps=[_USER_STEP])


# ---------------------------------------------------------------------------
# TestAgent
# ---------------------------------------------------------------------------


class TestAgent:
    def test_required_fields(self) -> None:
        a = Agent(name="harbor", version="2.0.0")
        assert a.name == "harbor"
        assert a.version == "2.0.0"

    def test_optional_fields_default_none(self) -> None:
        a = Agent(name="x", version="y")
        assert a.model_name is None
        assert a.tool_definitions is None
        assert a.extra is None

    def test_with_tool_definitions(self) -> None:
        a = Agent(
            name="x",
            version="y",
            tool_definitions=[{"name": "bash", "description": "run shell"}],
        )
        assert len(a.tool_definitions) == 1  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Agent(name="x", version="y", unknown_field="oops")  # type: ignore[call-arg]

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            Agent(version="y")

    def test_version_required(self) -> None:
        with pytest.raises(ValidationError):
            Agent(name="x")


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_all_none_valid(self) -> None:
        m = Metrics()
        assert m.prompt_tokens is None
        assert m.cost_usd is None

    def test_with_values(self) -> None:
        m = Metrics(prompt_tokens=1000, completion_tokens=200, cost_usd=0.005)
        assert m.prompt_tokens == 1000
        assert m.cost_usd == pytest.approx(0.005)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Metrics(unknown="x")  # type: ignore[call-arg]

    def test_token_ids(self) -> None:
        m = Metrics(prompt_token_ids=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])
        assert m.prompt_token_ids == [1, 2, 3]
        assert m.logprobs == pytest.approx([-0.1, -0.2, -0.3])


# ---------------------------------------------------------------------------
# TestFinalMetrics
# ---------------------------------------------------------------------------


class TestFinalMetrics:
    def test_all_none_valid(self) -> None:
        fm = FinalMetrics()
        assert fm.total_steps is None

    def test_total_steps_non_negative(self) -> None:
        fm = FinalMetrics(total_steps=0)
        assert fm.total_steps == 0

    def test_total_steps_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinalMetrics(total_steps=-1)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            FinalMetrics(unknown="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_required_fields(self) -> None:
        tc = ToolCall(
            tool_call_id="tc-1", function_name="Read", arguments={"path": "/x"}
        )
        assert tc.tool_call_id == "tc-1"
        assert tc.function_name == "Read"

    def test_empty_arguments_valid(self) -> None:
        tc = ToolCall(tool_call_id="tc-2", function_name="Noop", arguments={})
        assert tc.arguments == {}

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(tool_call_id="x", function_name="y", arguments={}, bogus=True)  # type: ignore[call-arg]

    def test_missing_tool_call_id(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(function_name="Read", arguments={})


# ---------------------------------------------------------------------------
# TestSubagentTrajectoryRef
# ---------------------------------------------------------------------------


class TestSubagentTrajectoryRef:
    def test_trajectory_id_only_valid(self) -> None:
        ref = SubagentTrajectoryRef(trajectory_id="traj-abc")
        assert ref.trajectory_id == "traj-abc"
        assert ref.trajectory_path is None

    def test_trajectory_path_only_valid(self) -> None:
        ref = SubagentTrajectoryRef(trajectory_path="s3://bucket/traj.json")
        assert ref.trajectory_path == "s3://bucket/traj.json"

    def test_both_set_valid(self) -> None:
        ref = SubagentTrajectoryRef(
            trajectory_id="traj-abc", trajectory_path="s3://bucket/traj.json"
        )
        assert ref.trajectory_id == "traj-abc"
        assert ref.trajectory_path == "s3://bucket/traj.json"

    def test_neither_set_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be resolvable"):
            SubagentTrajectoryRef()

    def test_session_id_alone_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be resolvable"):
            SubagentTrajectoryRef(session_id="sess-001")

    def test_empty_trajectory_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be resolvable"):
            SubagentTrajectoryRef(trajectory_id="")

    def test_whitespace_trajectory_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be resolvable"):
            SubagentTrajectoryRef(trajectory_id="   ")

    def test_empty_trajectory_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be resolvable"):
            SubagentTrajectoryRef(trajectory_path="")

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SubagentTrajectoryRef(trajectory_id="x", bogus=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestImageSource
# ---------------------------------------------------------------------------


class TestImageSource:
    def test_valid_jpeg(self) -> None:
        img = ImageSource(media_type="image/jpeg", path="/images/screenshot.jpg")
        assert img.media_type == "image/jpeg"

    def test_all_valid_types(self) -> None:
        for mime in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            img = ImageSource(media_type=mime, path="/x")
            assert img.media_type == mime

    def test_invalid_mime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ImageSource(media_type="image/bmp", path="/x")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ImageSource(media_type="image/png", path="/x", extra_field="y")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestAudioSource
# ---------------------------------------------------------------------------


class TestAudioSource:
    def test_canonical_types(self) -> None:
        for mime in (
            "audio/wav",
            "audio/mpeg",
            "audio/mp4",
            "audio/aac",
            "audio/ogg",
            "audio/flac",
            "audio/webm",
            "audio/aiff",
        ):
            src = AudioSource(media_type=mime, path="/audio/clip.wav")
            assert src.media_type == mime

    def test_mp3_alias_normalised(self) -> None:
        src = AudioSource(media_type="audio/mp3", path="/clip.mp3")
        assert src.media_type == "audio/mpeg"

    def test_x_wav_alias_normalised(self) -> None:
        src = AudioSource(media_type="audio/x-wav", path="/clip.wav")
        assert src.media_type == "audio/wav"

    def test_m4a_alias_normalised(self) -> None:
        src = AudioSource(media_type="audio/m4a", path="/clip.m4a")
        assert src.media_type == "audio/mp4"

    def test_duration_sec_non_negative(self) -> None:
        src = AudioSource(media_type="audio/wav", path="/x", duration_sec=0.0)
        assert src.duration_sec == 0.0

    def test_duration_sec_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AudioSource(media_type="audio/wav", path="/x", duration_sec=-1.0)

    def test_invalid_mime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AudioSource(media_type="audio/unknown", path="/x")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            AudioSource(media_type="audio/wav", path="/x", bogus=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestContentPart
# ---------------------------------------------------------------------------


class TestContentPart:
    def test_text_part(self) -> None:
        cp = ContentPart(type="text", text="Hello")
        assert cp.text == "Hello"
        assert cp.source is None

    def test_text_requires_text_field(self) -> None:
        with pytest.raises(ValidationError, match="'text' field is required"):
            ContentPart(type="text")

    def test_text_forbids_source(self) -> None:
        with pytest.raises(ValidationError, match="'source' field is not allowed"):
            ContentPart(
                type="text",
                text="hi",
                source=ImageSource(media_type="image/png", path="/x"),
            )

    def test_image_part(self) -> None:
        cp = ContentPart(
            type="image",
            source=ImageSource(media_type="image/png", path="/screenshot.png"),
        )
        assert cp.type == "image"
        assert isinstance(cp.source, ImageSource)

    def test_image_requires_source(self) -> None:
        with pytest.raises(ValidationError, match="'source' field is required"):
            ContentPart(type="image")

    def test_image_forbids_text(self) -> None:
        with pytest.raises(ValidationError, match="'text' field is not allowed"):
            ContentPart(
                type="image",
                text="caption",
                source=ImageSource(media_type="image/png", path="/x"),
            )

    def test_image_rejects_audio_source(self) -> None:
        with pytest.raises(ValidationError):
            ContentPart(
                type="image",
                source=AudioSource(media_type="audio/wav", path="/clip.wav"),
            )

    def test_audio_part(self) -> None:
        cp = ContentPart(
            type="audio",
            source=AudioSource(media_type="audio/mpeg", path="/clip.mp3"),
        )
        assert cp.type == "audio"
        assert isinstance(cp.source, AudioSource)

    def test_audio_rejects_image_source(self) -> None:
        with pytest.raises(ValidationError):
            ContentPart(
                type="audio",
                source=ImageSource(media_type="image/png", path="/x"),
            )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ContentPart(type="text", text="hi", bogus=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestObservationResult
# ---------------------------------------------------------------------------


class TestObservationResult:
    def test_string_content(self) -> None:
        r = ObservationResult(source_call_id="tc-1", content="output text")
        assert r.content == "output text"

    def test_content_part_list(self) -> None:
        parts = [ContentPart(type="text", text="result")]
        r = ObservationResult(content=parts)
        assert isinstance(r.content, list)
        assert len(r.content) == 1

    def test_none_content_valid(self) -> None:
        r = ObservationResult()
        assert r.content is None
        assert r.source_call_id is None

    def test_with_subagent_ref(self) -> None:
        ref = SubagentTrajectoryRef(trajectory_id="traj-sub")
        r = ObservationResult(subagent_trajectory_ref=[ref])
        assert r.subagent_trajectory_ref is not None
        assert len(r.subagent_trajectory_ref) == 1

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ObservationResult(bogus=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestObservation
# ---------------------------------------------------------------------------


class TestObservation:
    def test_basic(self) -> None:
        obs = Observation(results=[ObservationResult(content="ok")])
        assert len(obs.results) == 1

    def test_empty_results_valid(self) -> None:
        obs = Observation(results=[])
        assert obs.results == []

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Observation(results=[], bogus=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestStep
# ---------------------------------------------------------------------------


class TestStep:
    def test_user_step(self) -> None:
        s = Step(step_id=1, source="user", message="task")
        assert s.source == "user"
        assert s.message == "task"

    def test_system_step(self) -> None:
        s = Step(step_id=1, source="system", message="You are a helpful assistant.")
        assert s.source == "system"

    def test_agent_step_with_tool_calls(self) -> None:
        tc = ToolCall(tool_call_id="tc-1", function_name="Read", arguments={})
        s = Step(
            step_id=1,
            source="agent",
            message="reading file",
            tool_calls=[tc],
            metrics=Metrics(prompt_tokens=100),
        )
        assert s.tool_calls is not None
        assert len(s.tool_calls) == 1

    def test_step_id_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            Step(step_id=0, source="user", message="x")

    def test_valid_iso8601_timestamp(self) -> None:
        s = Step(
            step_id=1, source="user", message="x", timestamp="2026-09-08T10:00:00Z"
        )
        assert s.timestamp == "2026-09-08T10:00:00Z"

    def test_invalid_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Invalid ISO 8601"):
            Step(step_id=1, source="user", message="x", timestamp="not-a-date")

    def test_tool_calls_only_for_agent(self) -> None:
        tc = ToolCall(tool_call_id="tc-1", function_name="Read", arguments={})
        with pytest.raises(
            ValidationError, match="only applicable when source is 'agent'"
        ):
            Step(step_id=1, source="user", message="x", tool_calls=[tc])

    def test_metrics_only_for_agent(self) -> None:
        with pytest.raises(
            ValidationError, match="only applicable when source is 'agent'"
        ):
            Step(step_id=1, source="system", message="x", metrics=Metrics())

    def test_llm_call_count_zero_forbids_metrics(self) -> None:
        with pytest.raises(
            ValidationError, match="must be absent when llm_call_count is 0"
        ):
            Step(
                step_id=1,
                source="agent",
                message="dispatch",
                llm_call_count=0,
                metrics=Metrics(prompt_tokens=100),
            )

    def test_llm_call_count_zero_forbids_reasoning_content(self) -> None:
        with pytest.raises(
            ValidationError, match="must be absent when llm_call_count is 0"
        ):
            Step(
                step_id=1,
                source="agent",
                message="dispatch",
                llm_call_count=0,
                reasoning_content="thinking...",
            )

    def test_llm_call_count_zero_no_llm_fields_valid(self) -> None:
        s = Step(
            step_id=1,
            source="agent",
            message="deterministic dispatch",
            llm_call_count=0,
        )
        assert s.llm_call_count == 0

    def test_message_as_content_part_list(self) -> None:
        parts = [ContentPart(type="text", text="hello")]
        s = Step(step_id=1, source="user", message=parts)
        assert isinstance(s.message, list)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Step(step_id=1, source="user", message="x", bogus=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestTrajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_minimal_valid(self) -> None:
        t = _minimal()
        assert t.schema_version == "ATIF-v1.8"
        assert len(t.steps) == 1

    def test_steps_must_be_sequential_from_1(self) -> None:
        with pytest.raises(ValidationError, match="expected 1"):
            Trajectory(
                agent=_AGENT,
                steps=[Step(step_id=2, source="user", message="x")],
            )

    def test_steps_gap_rejected(self) -> None:
        with pytest.raises(ValidationError, match="expected 2"):
            Trajectory(
                agent=_AGENT,
                steps=[
                    Step(step_id=1, source="user", message="a"),
                    Step(step_id=3, source="user", message="b"),
                ],
            )

    def test_tool_call_ref_integrity(self) -> None:
        step = Step(
            step_id=1,
            source="agent",
            message="reading",
            tool_calls=[
                ToolCall(tool_call_id="tc-1", function_name="Read", arguments={})
            ],
            observation=Observation(
                results=[ObservationResult(source_call_id="tc-MISSING", content="x")]
            ),
        )
        with pytest.raises(ValidationError, match="source_call_id.*tc-MISSING"):
            Trajectory(agent=_AGENT, steps=[step])

    def test_valid_tool_call_reference(self) -> None:
        step = Step(
            step_id=1,
            source="agent",
            message="reading",
            tool_calls=[
                ToolCall(tool_call_id="tc-1", function_name="Read", arguments={})
            ],
            observation=Observation(
                results=[
                    ObservationResult(source_call_id="tc-1", content="file contents")
                ]
            ),
        )
        t = Trajectory(agent=_AGENT, steps=[step])
        assert len(t.steps) == 1

    def test_duplicate_tool_call_id_rejected(self) -> None:
        step = Step(
            step_id=1,
            source="agent",
            message="reading",
            tool_calls=[
                ToolCall(tool_call_id="tc-dup", function_name="Read", arguments={}),
                ToolCall(tool_call_id="tc-dup", function_name="Write", arguments={}),
            ],
        )
        with pytest.raises(ValidationError, match="duplicate tool_call_id.*tc-dup"):
            Trajectory(agent=_AGENT, steps=[step])

    def test_subagent_without_trajectory_id_rejected(self) -> None:
        sub = Trajectory(agent=Agent(name="sub", version="1"), steps=[_USER_STEP])
        with pytest.raises(ValidationError, match="trajectory_id is required"):
            Trajectory(agent=_AGENT, steps=[_USER_STEP], subagent_trajectories=[sub])

    def test_duplicate_subagent_trajectory_id_rejected(self) -> None:
        sub1 = Trajectory(
            trajectory_id="dup",
            agent=Agent(name="sub1", version="1"),
            steps=[_USER_STEP],
        )
        sub2 = Trajectory(
            trajectory_id="dup",
            agent=Agent(name="sub2", version="1"),
            steps=[_USER_STEP],
        )
        with pytest.raises(ValidationError, match="not unique"):
            Trajectory(
                agent=_AGENT, steps=[_USER_STEP], subagent_trajectories=[sub1, sub2]
            )

    def test_pathless_ref_with_no_embedded_trajectories_rejected(self) -> None:
        ref = SubagentTrajectoryRef(trajectory_id="traj-missing")
        step = Step(
            step_id=1,
            source="agent",
            message="delegating",
            observation=Observation(
                results=[ObservationResult(subagent_trajectory_ref=[ref])]
            ),
        )
        with pytest.raises(
            ValidationError, match="not present in subagent_trajectories"
        ):
            Trajectory(agent=_AGENT, steps=[step])

    def test_pathless_ref_matched_against_embedded_trajectory(self) -> None:
        sub = Trajectory(
            trajectory_id="traj-sub-001",
            agent=Agent(name="sub", version="1"),
            steps=[_USER_STEP],
        )
        ref = SubagentTrajectoryRef(trajectory_id="traj-sub-001")
        step = Step(
            step_id=1,
            source="agent",
            message="delegating",
            observation=Observation(
                results=[ObservationResult(subagent_trajectory_ref=[ref])]
            ),
        )
        t = Trajectory(agent=_AGENT, steps=[step], subagent_trajectories=[sub])
        assert len(t.subagent_trajectories) == 1  # type: ignore[arg-type]

    def test_ref_with_trajectory_path_skips_embedded_check(self) -> None:
        ref = SubagentTrajectoryRef(trajectory_path="s3://bucket/sub.json")
        step = Step(
            step_id=1,
            source="agent",
            message="delegating",
            observation=Observation(
                results=[ObservationResult(subagent_trajectory_ref=[ref])]
            ),
        )
        t = Trajectory(agent=_AGENT, steps=[step])
        assert len(t.steps) == 1

    def test_has_multimodal_content_false_for_text_only(self) -> None:
        t = _minimal()
        assert t.has_multimodal_content() is False

    def test_has_multimodal_content_true_for_image_in_message(self) -> None:
        img_step = Step(
            step_id=1,
            source="user",
            message=[
                ContentPart(
                    type="image",
                    source=ImageSource(media_type="image/png", path="/shot.png"),
                )
            ],
        )
        t = Trajectory(agent=_AGENT, steps=[img_step])
        assert t.has_multimodal_content() is True

    def test_has_multimodal_content_true_for_audio_in_observation(self) -> None:
        audio_result = ObservationResult(
            content=[
                ContentPart(
                    type="audio",
                    source=AudioSource(media_type="audio/wav", path="/rec.wav"),
                )
            ]
        )
        step = Step(
            step_id=1,
            source="agent",
            message="listening",
            observation=Observation(results=[audio_result]),
        )
        t = Trajectory(agent=_AGENT, steps=[step])
        assert t.has_multimodal_content() is True

    def test_to_json_dict_excludes_none(self) -> None:
        t = _minimal()
        d = t.to_json_dict()
        assert "session_id" not in d
        assert "notes" not in d
        assert "final_metrics" not in d

    def test_to_json_dict_include_none(self) -> None:
        t = _minimal()
        d = t.to_json_dict(exclude_none=False)
        assert "session_id" in d
        assert d["session_id"] is None

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Trajectory(agent=_AGENT, steps=[_USER_STEP], mystery_field="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestTrajectoryRoundtrip (uses realistic fixtures)
# ---------------------------------------------------------------------------


class TestTrajectoryRoundtrip:
    def test_agent_trajectory_roundtrip(self, agent_trajectory: Trajectory) -> None:  # noqa: F811
        data = agent_trajectory.model_dump(mode="json")
        restored = Trajectory.model_validate(data)
        assert restored.trajectory_id == agent_trajectory.trajectory_id
        assert len(restored.steps) == len(agent_trajectory.steps)
        assert restored.final_metrics is not None
        assert restored.final_metrics.total_steps == 4

    def test_subagent_trajectory_roundtrip(
        self,
        subagent_trajectory: Trajectory,  # noqa: F811
    ) -> None:
        data = subagent_trajectory.model_dump(mode="json")
        restored = Trajectory.model_validate(data)
        assert restored.subagent_trajectories is not None
        assert len(restored.subagent_trajectories) == 1
        assert restored.subagent_trajectories[0].trajectory_id == "traj-linter-sub-001"

    def test_to_json_dict_no_none_values(self, agent_trajectory: Trajectory) -> None:  # noqa: F811
        d = agent_trajectory.to_json_dict(exclude_none=True)

        def _check_no_none(obj: object) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    assert v is not None, f"Unexpected None in serialised dict: {obj}"
                    _check_no_none(v)
            elif isinstance(obj, list):
                for item in obj:
                    _check_no_none(item)

        _check_no_none(d)

    def test_minimal_trajectory_roundtrip(self, minimal_trajectory: Trajectory) -> None:  # noqa: F811
        data = minimal_trajectory.model_dump(mode="json")
        restored = Trajectory.model_validate(data)
        assert restored.steps[0].message == "Hello, fix auth.py"

    def test_tool_call_obs_integrity_preserved(
        self,
        agent_trajectory: Trajectory,  # noqa: F811
    ) -> None:
        # Step 3 has tc-read-001, step 4 has tc-edit-001
        data = agent_trajectory.model_dump(mode="json")
        restored = Trajectory.model_validate(data)
        step3 = restored.steps[2]
        assert step3.observation is not None
        assert step3.observation.results[0].source_call_id == "tc-read-001"


# ---------------------------------------------------------------------------
# TestATIFNamespaceImports
# ---------------------------------------------------------------------------


class TestATIFNamespaceImports:
    _NAMES = [
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

    def test_importable_from_evalhub_models_atif(self) -> None:
        from evalhub.models import atif as atif_mod

        for name in self._NAMES:
            assert hasattr(atif_mod, name), f"Missing {name} in evalhub.models.atif"

    def test_importable_from_evalhub_atif_shim(self) -> None:
        for name in self._NAMES:
            assert hasattr(atif_shim, name), f"Missing {name} in evalhub.atif"

    def test_importable_from_top_level_evalhub(self) -> None:
        for name in self._NAMES:
            assert hasattr(evalhub, name), f"Missing {name} in evalhub"

    def test_trajectory_is_same_class_across_namespaces(self) -> None:
        from evalhub.atif import Trajectory as TrajectoryAtif
        from evalhub.models.atif import Trajectory as TrajectoryModel

        assert TrajectoryModel is TrajectoryAtif
        assert evalhub.Trajectory is TrajectoryModel
