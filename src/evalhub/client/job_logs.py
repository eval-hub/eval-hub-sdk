"""Shared helpers for evaluation job log APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from ..models import EvaluationJob, JobStatus

if TYPE_CHECKING:
    pass

DEFAULT_LOG_TAIL_LINES = 1000
MAX_LOG_TAIL_LINES = 10000
TERMINAL_JOB_STATES = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.PARTIALLY_FAILED,
    }
)


@dataclass(frozen=True)
class JobLogOptions:
    """Query options for evaluation job log endpoints."""

    tail_lines: int = DEFAULT_LOG_TAIL_LINES
    timestamps: bool = False
    since_seconds: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.tail_lines <= MAX_LOG_TAIL_LINES:
            msg = f"tail_lines must be between 1 and {MAX_LOG_TAIL_LINES}"
            raise ValueError(msg)
        if self.since_seconds is not None and self.since_seconds < 1:
            raise ValueError("since_seconds must be >= 1 when provided")


@dataclass(frozen=True)
class JobLogUpdate:
    """Incremental log and status update while watching a job."""

    logs: str
    job: EvaluationJob


class _AsyncJobLogClient(Protocol):
    async def _request_get(self, path: str, **kwargs: Any) -> Any:
        ...


class _SyncJobLogClient(Protocol):
    def _request_get(self, path: str, **kwargs: Any) -> Any:
        ...


def build_log_query_params(options: JobLogOptions) -> dict[str, str]:
    """Build query parameters for log endpoints."""
    params = {"tail_lines": str(options.tail_lines)}
    if options.timestamps:
        params["timestamps"] = "true"
    if options.since_seconds is not None:
        params["since_seconds"] = str(options.since_seconds)
    return params


def build_logs_path(job_id: str, benchmark_index: int | None = None) -> str:
    """Build the API path for job or benchmark log endpoints."""
    if benchmark_index is None:
        return f"/evaluations/jobs/{job_id}/logs"
    return f"/evaluations/jobs/{job_id}/benchmarks/{benchmark_index}/logs"


def _suffix_prefix_overlap(seen: str, current: str) -> int:
    """Return length of the longest suffix of *seen* matching a prefix of *current*."""
    if not seen or not current:
        return 0
    # KMP prefix function on current + sentinel + seen is O(len(seen) + len(current)).
    combined = f"{current}\x00{seen}"
    pi = _kmp_prefix_lengths(combined)
    return min(pi[-1], len(current))


def _kmp_prefix_lengths(text: str) -> list[int]:
    """Build the KMP prefix-function (pi) table for *text*."""
    pi = [0] * len(text)
    for i in range(1, len(text)):
        j = pi[i - 1]
        while j > 0 and text[i] != text[j]:
            j = pi[j - 1]
        if text[i] == text[j]:
            j += 1
        pi[i] = j
    return pi


def log_delta(seen: str, current: str) -> str:
    """Return only the portion of *current* that has not been emitted yet."""
    if not current:
        return ""
    if not seen:
        return current
    if current.startswith(seen):
        return current[len(seen) :]
    overlap = _suffix_prefix_overlap(seen, current)
    return current[overlap:]


def is_terminal_job(job: EvaluationJob) -> bool:
    """Return True when the job has reached a terminal state."""
    return job.effective_state in TERMINAL_JOB_STATES


async def fetch_job_logs(
    client: _AsyncJobLogClient,
    job_id: str,
    *,
    benchmark_index: int | None = None,
    options: JobLogOptions | None = None,
    tenant: str | None = None,
) -> str:
    """Fetch a snapshot of evaluation job logs."""
    opts = options or JobLogOptions()
    response = await client._request_get(
        build_logs_path(job_id, benchmark_index),
        params=build_log_query_params(opts),
        tenant=tenant,
    )
    return cast(str, response.text)


def fetch_job_logs_sync(
    client: _SyncJobLogClient,
    job_id: str,
    *,
    benchmark_index: int | None = None,
    options: JobLogOptions | None = None,
    tenant: str | None = None,
) -> str:
    """Fetch a snapshot of evaluation job logs (sync)."""
    opts = options or JobLogOptions()
    response = client._request_get(
        build_logs_path(job_id, benchmark_index),
        params=build_log_query_params(opts),
        tenant=tenant,
    )
    return cast(str, response.text)
