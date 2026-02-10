"""Unit tests for EvalHub client components.

These tests can run in two modes:

1. Unit test mode (default): Uses mocks for fast, isolated testing
2. Integration test mode: Set EVALHUB_TEST_BASE_URL environment variable to test
   against a real EvalHub server

Example:
    # Run with mocks (fast)
    $ uv run pytest tests/unit/test_evalhub_client.py

    # Run against real server (requires running server)
    $ EVALHUB_TEST_BASE_URL=http://localhost:8080 uv run pytest tests/unit/test_evalhub_client.py
"""

import os
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest
from evalhub import (
    AsyncEvalHubClient,
    AsyncEvaluationsClient,
    AsyncProvidersClient,
    SyncEvalHubClient,
    SyncEvaluationsClient,
    SyncProvidersClient,
)
from evalhub.client.base import (
    BaseAsyncClient,
    BaseSyncClient,
    JobCanNotBeCancelledError,
    JobNotFoundError,
)
from evalhub.models.api import (
    BenchmarkInfo,
    EvaluationJob,
    EvaluationRequest,
    JobStatus,
    ModelConfig,
)

# Environment variable to enable real server testing
EVALHUB_TEST_BASE_URL = os.environ.get("EVALHUB_TEST_BASE_URL")


@pytest.fixture
def use_real_server() -> bool:
    """Determine if tests should use a real server or mocks."""
    return EVALHUB_TEST_BASE_URL is not None


@pytest.fixture
def base_url() -> str:
    """Get the base URL for the test server."""
    return EVALHUB_TEST_BASE_URL or "http://test.example.com"


@pytest.fixture
def mock_request_or_real(use_real_server: bool) -> type[Any]:
    """Context manager that either mocks _request or passes through to real server.

    Usage in tests:
        with mock_request_or_real(client, mock_response) as should_assert_call:
            result = client.some_method()
            if should_assert_call:
                # In mock mode, verify the mock was called correctly
                pass
    """

    class MockOrReal:
        def __init__(
            self,
            client: BaseAsyncClient | BaseSyncClient,
            mock_response: Mock | None = None,
        ) -> None:
            self.client = client
            self.mock_response = mock_response
            self.patch_context: Any = None
            self.use_real = use_real_server

        def __enter__(self) -> Any:
            if not self.use_real and self.mock_response:
                # Mock mode: patch the _request method
                self.patch_context = patch.object(
                    self.client, "_request", return_value=self.mock_response
                )
                self.mock_request = self.patch_context.__enter__()
                return self.mock_request
            else:
                # Real server mode: no mocking
                return None

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self.patch_context:
                self.patch_context.__exit__(exc_type, exc_val, exc_tb)

    return MockOrReal


class TestClientInheritance:
    """Test cases for client inheritance structure."""

    def test_async_providers_client_inherits_from_base(self) -> None:
        """Test that AsyncProvidersClient inherits from BaseAsyncClient."""
        assert issubclass(AsyncProvidersClient, BaseAsyncClient)

    def test_async_evaluations_client_inherits_from_base(self) -> None:
        """Test that AsyncEvaluationsClient inherits from BaseAsyncClient."""
        assert issubclass(AsyncEvaluationsClient, BaseAsyncClient)

    def test_async_evalhub_client_inherits_from_base(self) -> None:
        """Test that AsyncEvalHubClient inherits from BaseAsyncClient."""
        assert issubclass(AsyncEvalHubClient, BaseAsyncClient)

    def test_sync_providers_client_inherits_from_base(self) -> None:
        """Test that SyncProvidersClient inherits from BaseSyncClient."""
        assert issubclass(SyncProvidersClient, BaseSyncClient)

    def test_sync_evaluations_client_inherits_from_base(self) -> None:
        """Test that SyncEvaluationsClient inherits from BaseSyncClient."""
        assert issubclass(SyncEvaluationsClient, BaseSyncClient)

    def test_sync_evalhub_client_inherits_from_base(self) -> None:
        """Test that SyncEvalHubClient inherits from BaseSyncClient."""
        assert issubclass(SyncEvalHubClient, BaseSyncClient)

    @pytest.mark.asyncio
    async def test_async_evalhub_client_has_all_resources(self) -> None:
        """Test that AsyncEvalHubClient has all nested resources."""
        client = AsyncEvalHubClient()

        # BaseAsyncClient methods
        assert hasattr(client, "health")
        assert hasattr(client, "close")

        # Resource properties
        assert hasattr(client, "providers")
        assert hasattr(client, "benchmarks")
        assert hasattr(client, "collections")
        assert hasattr(client, "jobs")

        # Resource methods
        assert hasattr(client.providers, "list")
        assert hasattr(client.providers, "get")
        assert hasattr(client.benchmarks, "list")
        assert hasattr(client.collections, "list")
        assert hasattr(client.collections, "get")
        assert hasattr(client.jobs, "submit")
        assert hasattr(client.jobs, "get")
        assert hasattr(client.jobs, "list")
        assert hasattr(client.jobs, "cancel")
        assert hasattr(client.jobs, "wait_for_completion")

        await client.close()

    def test_sync_evalhub_client_has_all_resources(self) -> None:
        """Test that SyncEvalHubClient has all nested resources."""
        client = SyncEvalHubClient()

        # BaseSyncClient methods
        assert hasattr(client, "health")
        assert hasattr(client, "close")

        # Resource properties
        assert hasattr(client, "providers")
        assert hasattr(client, "benchmarks")
        assert hasattr(client, "collections")
        assert hasattr(client, "jobs")

        # Resource methods
        assert hasattr(client.providers, "list")
        assert hasattr(client.providers, "get")
        assert hasattr(client.benchmarks, "list")
        assert hasattr(client.collections, "list")
        assert hasattr(client.collections, "get")
        assert hasattr(client.jobs, "submit")
        assert hasattr(client.jobs, "get")
        assert hasattr(client.jobs, "list")
        assert hasattr(client.jobs, "cancel")
        assert hasattr(client.jobs, "wait_for_completion")

        client.close()


class TestProvidersClient:
    """Test cases for ProvidersClient."""

    @pytest.fixture
    def mock_providers_data(self) -> dict[str, Any]:
        """Mock provider data for tests (as returned by Go API)."""
        return {
            "total_count": 2,
            "items": [
                {
                    "id": "lm_evaluation_harness",
                    "name": "LM Evaluation Harness",
                    "description": "Evaluation harness for language models",
                    "type": "lm_evaluation_harness",
                    "benchmarks": [],
                },
                {
                    "id": "ragas",
                    "name": "RAGAS",
                    "description": "RAG Assessment framework",
                    "type": "ragas",
                    "benchmarks": [],
                },
            ],
        }

    @pytest.fixture
    def mock_benchmarks_data(self) -> dict[str, Any]:
        """Mock benchmark data for tests (as returned by API)."""
        return {
            "total_count": 2,
            "items": [
                {
                    "id": "gsm8k",
                    "label": "GSM8K",
                    "description": "Grade School Math 8K",
                    "category": "math",
                    "metrics": ["accuracy"],
                    "provider_id": "lm_evaluation_harness",
                    "num_few_shot": 5,
                    "dataset_size": 1000,
                    "tags": [],
                },
                {
                    "id": "mmlu",
                    "label": "MMLU",
                    "description": "Massive Multitask Language Understanding",
                    "category": "knowledge",
                    "metrics": ["accuracy"],
                    "provider_id": "lm_evaluation_harness",
                    "num_few_shot": 0,
                    "dataset_size": 5000,
                    "tags": [],
                },
            ],
        }

    def test_list_providers(
        self,
        base_url: str,
        use_real_server: bool,
        mock_providers_data: dict[str, Any],
        mock_request_or_real: Any,
    ) -> None:
        """Test listing providers (synchronous).

        Works with both mock and real server:
        - Mock mode: Uses mock_providers_data
        - Real server mode: Calls actual API (requires server at EVALHUB_TEST_BASE_URL)
        """
        client = SyncProvidersClient(base_url=base_url)

        if not use_real_server:
            mock_response = Mock()
            mock_response.json.return_value = mock_providers_data

            with mock_request_or_real(client, mock_response) as mock_request:
                providers = client.list()

                assert len(providers) >= 2
                assert any(p.id == "lm_evaluation_harness" for p in providers)

                if mock_request:
                    mock_request.assert_called_once()
        else:
            # Real server mode - just verify the call works
            try:
                providers = client.list()
                assert isinstance(providers, list)
                # In real mode, we can't guarantee specific data, just that it works
                print(f"✓ Real server returned {len(providers)} providers")
            except Exception as e:
                pytest.skip(f"Real server not available or returned error: {e}")

        client.close()

    def test_list_benchmarks(
        self,
        base_url: str,
        use_real_server: bool,
        mock_benchmarks_data: dict[str, Any],
        mock_request_or_real: Any,
    ) -> None:
        """Test listing benchmarks (synchronous)."""
        client = SyncProvidersClient(base_url=base_url)

        if not use_real_server:
            mock_response = Mock()
            mock_response.json.return_value = mock_benchmarks_data

            with mock_request_or_real(client, mock_response):
                benchmarks = client.list_benchmarks()

                assert len(benchmarks) >= 2
                assert isinstance(benchmarks[0], BenchmarkInfo)
                assert any(b.benchmark_id == "gsm8k" for b in benchmarks)
        else:
            try:
                benchmarks = client.list_benchmarks()
                assert isinstance(benchmarks, list)
                if benchmarks:
                    assert isinstance(benchmarks[0], BenchmarkInfo)
                print(f"✓ Real server returned {len(benchmarks)} benchmarks")
            except Exception as e:
                pytest.skip(f"Real server not available: {e}")

        client.close()

    def test_list_benchmarks_with_filters(
        self,
        base_url: str,
        use_real_server: bool,
        mock_benchmarks_data: dict[str, Any],
        mock_request_or_real: Any,
    ) -> None:
        """Test listing benchmarks with filters (synchronous)."""
        client = SyncProvidersClient(base_url=base_url)

        if not use_real_server:
            mock_response = Mock()
            # Filter to just math benchmarks
            mock_response.json.return_value = {
                "total_count": 1,
                "items": [mock_benchmarks_data["items"][0]],
            }

            with mock_request_or_real(client, mock_response) as mock_request:
                benchmarks = client.list_benchmarks(category="math", limit=10)

                assert len(benchmarks) >= 1
                assert benchmarks[0].benchmark_id == "gsm8k"
                assert benchmarks[0].category == "math"

                # Verify request was called with params in mock mode
                if mock_request:
                    call_args = mock_request.call_args
                    assert "params" in call_args.kwargs
                    assert call_args.kwargs["params"]["category"] == "math"
                    assert call_args.kwargs["params"]["limit"] == "10"
        else:
            try:
                benchmarks = client.list_benchmarks(category="math", limit=10)
                assert isinstance(benchmarks, list)
                if benchmarks:
                    assert isinstance(benchmarks[0], BenchmarkInfo)
                print(f"✓ Real server returned {len(benchmarks)} math benchmarks")
            except Exception as e:
                pytest.skip(f"Real server not available: {e}")

        client.close()


class TestEvaluationsClient:
    """Test cases for EvaluationsClient."""

    @pytest.fixture
    def mock_job_data(self) -> dict[str, Any]:
        """Mock evaluation job data for tests."""
        return {
            "resource": {
                "id": "job_123",
                "tenant": "default",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
            },
            "status": {"state": JobStatus.PENDING.value},
            "model": {"url": "http://localhost:8000/v1", "name": "gpt-3.5-turbo"},
            "benchmarks": [
                {
                    "id": "gsm8k",
                    "provider_id": "lm_evaluation_harness",
                    "parameters": {},
                }
            ],
        }

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode - would create actual jobs",
    )
    def test_submit_evaluation(self, mock_job_data: dict[str, Any]) -> None:
        """Test submitting an evaluation (synchronous).

        Note: Skipped in real server mode to avoid creating actual evaluation jobs.
        """
        client = SyncEvaluationsClient()
        mock_response = Mock()
        mock_response.json.return_value = mock_job_data

        with patch.object(client, "_request", return_value=mock_response):
            model = ModelConfig(url="http://localhost:8000/v1", name="gpt-3.5-turbo")
            request = EvaluationRequest(benchmark_id="gsm8k", model=model)
            job = client.submit(request)

            assert isinstance(job, EvaluationJob)
            assert job.id == "job_123"
            assert job.state == JobStatus.PENDING

        client.close()

    def test_get_job_status(
        self,
        base_url: str,
        use_real_server: bool,
        mock_job_data: dict[str, Any],
        mock_request_or_real: Any,
    ) -> None:
        """Test getting job status (synchronous)."""
        client = SyncEvaluationsClient(base_url=base_url)

        if not use_real_server:
            mock_response = Mock()
            mock_response.json.return_value = mock_job_data

            with mock_request_or_real(client, mock_response):
                job = client.get_job("job_123")

                assert isinstance(job, EvaluationJob)
                assert job.id == "job_123"
                assert job.state == JobStatus.PENDING
        else:
            # In real server mode, we need a valid job_id
            # This test would require creating a job first or having a known test job
            pytest.skip("Requires a valid job_id on the real server")

        client.close()

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode - would cancel actual jobs",
    )
    def test_cancel_job_success(self) -> None:
        """Test successful job cancellation (synchronous).

        Note: Skipped in real server mode to avoid canceling actual jobs.
        """
        client = SyncEvaluationsClient()
        mock_response = Mock()

        with patch.object(client, "_request", return_value=mock_response):
            result = client.cancel("job_123")
            assert result is True

        client.close()

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode - tests specific error condition",
    )
    def test_cancel_job_not_found(self) -> None:
        """Test job cancellation raises JobNotFoundError on 404."""
        client = SyncEvaluationsClient()

        mock_response = Mock()
        mock_response.status_code = 404
        http_error = httpx.HTTPStatusError(
            "Not found", request=Mock(), response=mock_response
        )

        with patch.object(client, "_request", side_effect=http_error):
            with pytest.raises(JobNotFoundError) as exc_info:
                client.cancel("non_existent")
            assert exc_info.value.job_id == "non_existent"
            assert exc_info.value.cause is http_error

        client.close()

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode - tests specific error condition",
    )
    def test_cancel_job_cannot_be_cancelled(self) -> None:
        """Test job cancellation raises JobCanNotBeCancelledError on 400."""
        client = SyncEvaluationsClient()

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "JobCanNotBeCancelled",
            "message": "The job job_456 can not be cancelled because it is 'completed'.",
        }
        http_error = httpx.HTTPStatusError(
            "Bad request", request=Mock(), response=mock_response
        )

        with patch.object(client, "_request", side_effect=http_error):
            with pytest.raises(JobCanNotBeCancelledError) as exc_info:
                client.cancel("job_456")
            assert exc_info.value.job_id == "job_456"
            assert "completed" in str(exc_info.value.reason)
            assert exc_info.value.cause is http_error

        client.close()

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode - tests specific error condition",
    )
    def test_cancel_job_cannot_be_cancelled_without_body(self) -> None:
        """Test JobCanNotBeCancelledError when response body is not JSON."""
        client = SyncEvaluationsClient()

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.side_effect = ValueError("No JSON")
        http_error = httpx.HTTPStatusError(
            "Bad request", request=Mock(), response=mock_response
        )

        with patch.object(client, "_request", side_effect=http_error):
            with pytest.raises(JobCanNotBeCancelledError) as exc_info:
                client.cancel("job_789")
            assert exc_info.value.job_id == "job_789"
            assert exc_info.value.reason is None

        client.close()


class TestEvalHubClient:
    """Test cases for complete EvalHubClient."""

    def test_sync_client_initialization(self) -> None:
        """Test sync client initialization with custom parameters."""
        client = SyncEvalHubClient(
            base_url="https://evalhub.example.com",
            auth_token="test-token",
            timeout=60.0,
            max_retries=5,
        )

        assert client.base_url == "https://evalhub.example.com"
        assert client.api_base == "https://evalhub.example.com/api/v1"
        assert client.auth_token == "test-token"
        assert client.max_retries == 5

        client.close()

    def test_sync_client_has_nested_resources(
        self, base_url: str, use_real_server: bool, mock_request_or_real: Any
    ) -> None:
        """Test that SyncEvalHubClient has nested resource structure."""
        client = SyncEvalHubClient(base_url=base_url)

        if not use_real_server:
            # Test providers resource
            mock_response_providers = Mock()
            mock_response_providers.json.return_value = {"total_count": 0, "items": []}
            with patch.object(client, "_request", return_value=mock_response_providers):
                providers = client.providers.list()
                assert isinstance(providers, list)

            # Test benchmarks resource
            mock_response_benchmarks = Mock()
            mock_response_benchmarks.json.return_value = {"total_count": 0, "items": []}
            with patch.object(
                client, "_request", return_value=mock_response_benchmarks
            ):
                benchmarks = client.benchmarks.list()
                assert isinstance(benchmarks, list)
        else:
            try:
                # Test with real server
                providers = client.providers.list()
                assert isinstance(providers, list)

                benchmarks = client.benchmarks.list()
                assert isinstance(benchmarks, list)

                print(
                    f"✓ Real server: {len(providers)} providers, {len(benchmarks)} benchmarks"
                )
            except Exception as e:
                pytest.skip(f"Real server not available: {e}")

        client.close()

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode - would create actual jobs",
    )
    def test_sync_client_has_jobs_resource(self) -> None:
        """Test that SyncEvalHubClient has jobs resource.

        Note: Skipped in real server mode to avoid creating actual evaluation jobs.
        """
        from evalhub.models.api import BenchmarkConfig, JobSubmissionRequest

        client = SyncEvalHubClient()
        mock_job_data = {
            "resource": {
                "id": "job_123",
                "tenant": "default",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
            },
            "status": {"state": JobStatus.PENDING.value},
            "model": {"url": "http://localhost:8000/v1", "name": "gpt-3.5-turbo"},
            "benchmarks": [
                {
                    "id": "gsm8k",
                    "provider_id": "lm_evaluation_harness",
                    "parameters": {},
                }
            ],
        }
        mock_response = Mock()
        mock_response.json.return_value = mock_job_data

        with patch.object(client, "_request", return_value=mock_response):
            # Should be able to call job methods via jobs resource
            model = ModelConfig(url="http://localhost:8000/v1", name="gpt-3.5-turbo")
            benchmark = BenchmarkConfig(
                id="gsm8k", provider_id="lm_evaluation_harness", parameters={}
            )
            request = JobSubmissionRequest(model=model, benchmarks=[benchmark])
            job = client.jobs.submit(request)
            assert isinstance(job, EvaluationJob)

        with patch.object(client, "_request", return_value=mock_response):
            job_status = client.jobs.get("job_123")
            assert isinstance(job_status, EvaluationJob)

        client.close()

    def test_sync_client_context_manager(self) -> None:
        """Test SyncEvalHubClient as context manager."""
        with SyncEvalHubClient() as client:
            assert client.base_url == "http://localhost:8080"
            assert client.api_base == "http://localhost:8080/api/v1"

    @pytest.mark.asyncio
    async def test_async_client_context_manager(self) -> None:
        """Test AsyncEvalHubClient as async context manager."""
        async with AsyncEvalHubClient() as client:
            assert client.base_url == "http://localhost:8080"
            mock_response = Mock()
            mock_response.json.return_value = {"status": "healthy"}

            with patch.object(client, "_request", return_value=mock_response):
                health = await client.health()
                assert health["status"] == "healthy"
