"""Unit tests for evaluation job log client helpers and resources."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from evalhub.client.job_logs import (
    JobLogOptions,
    JobLogUpdate,
    build_log_query_params,
    build_logs_path,
    log_delta,
)
from evalhub.client.resources.jobs import AsyncJobsResource, SyncJobsResource
from evalhub.models.api import (
    BenchmarkConfig,
    BenchmarkStatus,
    EvaluationJob,
    EvaluationJobResource,
    EvaluationJobStatus,
    JobStatus,
    ModelConfig,
)

NOW = datetime.now(UTC)


def _make_job(state: JobStatus) -> EvaluationJob:
    return EvaluationJob(
        resource=EvaluationJobResource(id="job-1", created_at=NOW),
        status=EvaluationJobStatus(
            state=state,
            benchmarks=[
                BenchmarkStatus(
                    provider_id="test",
                    id="bench",
                    benchmark_index=0,
                    status=state,
                )
            ],
        ),
        name="test-job",
        model=ModelConfig(url="http://vllm:8000/v1", name="llama3"),
        benchmarks=[BenchmarkConfig(id="bench", provider_id="test")],
    )


@pytest.mark.unit
class TestJobLogHelpers:
    def test_build_logs_path_job_level(self) -> None:
        assert build_logs_path("job-1") == "/evaluations/jobs/job-1/logs"

    def test_build_logs_path_benchmark_level(self) -> None:
        assert (
            build_logs_path("job-1", benchmark_index=2)
            == "/evaluations/jobs/job-1/benchmarks/2/logs"
        )

    def test_build_log_query_params_defaults(self) -> None:
        assert build_log_query_params(JobLogOptions()) == {"tail_lines": "1000"}

    def test_build_log_query_params_all_fields(self) -> None:
        params = build_log_query_params(
            JobLogOptions(
                tail_lines=250,
                timestamps=True,
                since_seconds=30,
            )
        )
        assert params == {
            "tail_lines": "250",
            "timestamps": "true",
            "since_seconds": "30",
        }

    def test_job_log_options_validates_tail_lines(self) -> None:
        with pytest.raises(ValueError, match="tail_lines"):
            JobLogOptions(tail_lines=0)

    def test_job_log_options_validates_since_seconds(self) -> None:
        with pytest.raises(ValueError, match="since_seconds"):
            JobLogOptions(since_seconds=0)

    @pytest.mark.parametrize(
        ("seen", "current", "expected"),
        [
            ("", "line-1\nline-2", "line-1\nline-2"),
            ("line-1\n", "line-1\nline-2", "line-2"),
            ("old", "new", "new"),
            ("abc", "bcdef", "def"),
            ("line-1", "", ""),
        ],
    )
    def test_log_delta(self, seen: str, current: str, expected: str) -> None:
        assert log_delta(seen, current) == expected


@pytest.mark.unit
class TestSyncJobLogs:
    def test_get_logs_calls_plain_text_endpoint(self) -> None:
        client = Mock()
        response = Mock()
        response.text = "INFO starting\n"
        client._request_get.return_value = response
        resource = SyncJobsResource(client)

        logs = resource.get_logs(
            "job-1",
            benchmark_index=1,
            options=JobLogOptions(tail_lines=200, timestamps=True),
            tenant="tenant-a",
        )

        assert logs == "INFO starting\n"
        client._request_get.assert_called_once_with(
            "/evaluations/jobs/job-1/benchmarks/1/logs",
            params={"tail_lines": "200", "timestamps": "true"},
            tenant="tenant-a",
        )

    def test_watch_logs_streams_until_terminal(self) -> None:
        running = _make_job(JobStatus.RUNNING)
        completed = _make_job(JobStatus.COMPLETED)
        client = Mock()
        resource = SyncJobsResource(client)

        with (
            patch.object(resource, "get", side_effect=[running, completed]) as mock_get,
            patch.object(
                resource,
                "get_logs",
                side_effect=["INFO start\n", "INFO start\nINFO done\n"],
            ) as mock_logs,
            patch("evalhub.client.resources.jobs.time.sleep"),
        ):
            updates = list(
                resource.watch_logs("job-1", poll_interval=0.01, timeout=5.0)
            )

        assert mock_get.call_count == 2
        assert mock_logs.call_count == 2
        assert len(updates) == 2
        assert updates[0] == JobLogUpdate(logs="INFO start\n", job=running)
        assert updates[1] == JobLogUpdate(logs="INFO done\n", job=completed)

    def test_watch_logs_timeout(self) -> None:
        running = _make_job(JobStatus.RUNNING)
        client = Mock()
        resource = SyncJobsResource(client)

        with (
            patch.object(resource, "get", return_value=running),
            patch.object(resource, "get_logs", return_value=""),
            patch("evalhub.client.resources.jobs.time.sleep"),
            patch("evalhub.client.resources.jobs.time.time", side_effect=[0.0, 2.0]),
        ):
            with pytest.raises(TimeoutError, match="did not complete within"):
                list(resource.watch_logs("job-1", poll_interval=0.01, timeout=1.0))


@pytest.mark.unit
class TestAsyncJobLogs:
    @pytest.mark.asyncio
    async def test_get_logs_calls_plain_text_endpoint(self) -> None:
        client = AsyncMock()
        response = Mock()
        response.text = "INFO starting\n"
        client._request_get.return_value = response
        resource = AsyncJobsResource(client)

        logs = await resource.get_logs("job-1")

        assert logs == "INFO starting\n"
        client._request_get.assert_awaited_once_with(
            "/evaluations/jobs/job-1/logs",
            params={"tail_lines": "1000"},
            tenant=None,
        )

    @pytest.mark.asyncio
    async def test_watch_logs_streams_until_terminal(self) -> None:
        running = _make_job(JobStatus.RUNNING)
        completed = _make_job(JobStatus.COMPLETED)
        client = AsyncMock()
        resource = AsyncJobsResource(client)

        with (
            patch.object(
                resource,
                "get",
                side_effect=[running, completed],
                new_callable=AsyncMock,
            ) as mock_get,
            patch.object(
                resource,
                "get_logs",
                side_effect=["line-1\n", "line-1\nline-2\n"],
                new_callable=AsyncMock,
            ) as mock_logs,
            patch(
                "evalhub.client.resources.jobs.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            updates: list[JobLogUpdate] = []
            async for update in resource.watch_logs(
                "job-1", poll_interval=0.01, timeout=5.0
            ):
                updates.append(update)

        assert mock_get.await_count == 2
        assert mock_logs.await_count == 2
        assert updates[0].logs == "line-1\n"
        assert updates[1].logs == "line-2\n"
        assert updates[1].job.effective_state == JobStatus.COMPLETED
