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

import pytest
from evalhub import (
    AsyncEvalHubClient,
    SyncEvalHubClient,
)
from evalhub.client.base import (
    BaseAsyncClient,
    BaseSyncClient,
)
from evalhub.models.api import (
    EvaluationJob,
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

    def test_async_evalhub_client_inherits_from_base(self) -> None:
        """Test that AsyncEvalHubClient inherits from BaseAsyncClient."""
        assert issubclass(AsyncEvalHubClient, BaseAsyncClient)

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
            "name": "gsm8k-eval",
            "description": "Evaluate GSM8K benchmark",
            "tags": ["math", "reasoning"],
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
            request = JobSubmissionRequest(
                name="gsm8k-eval",
                description="Evaluate GSM8K benchmark",
                tags=["math", "reasoning"],
                model=model,
                benchmarks=[benchmark],
            )
            job = client.jobs.submit(request)
            assert isinstance(job, EvaluationJob)
            assert job.name == "gsm8k-eval"
            assert job.description == "Evaluate GSM8K benchmark"
            assert job.tags == ["math", "reasoning"]

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
