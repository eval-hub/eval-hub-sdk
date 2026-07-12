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
    AgentMetadata,
    Benchmark,
    BenchmarkAgentMetadata,
    BenchmarkConfig,
    CollectionRef,
    EvaluationExports,
    EvaluationExportsOCI,
    EvaluationJob,
    JobStatus,
    JobSubmissionRequest,
    ModelConfig,
    OCIConnectionConfig,
    OCICoordinates,
    Provider,
    Resource,
    ScoreRange,
)

pytestmark = pytest.mark.unit

# Environment variable to enable real server testing
EVALHUB_TEST_BASE_URL = os.environ.get("EVALHUB_TEST_BASE_URL")

JOBS_LIST_PARAM_CASES = [
    pytest.param(
        {"status": JobStatus.RUNNING}, {"status": "running"}, id="status-only"
    ),
    pytest.param({"limit": 10}, {"limit": "10"}, id="limit-only"),
    pytest.param(
        {"status": JobStatus.COMPLETED, "limit": 5},
        {"status": "completed", "limit": "5"},
        id="status-and-limit",
    ),
    pytest.param({}, {}, id="no-params"),
]


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

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode - would create actual jobs",
    )
    def test_sync_client_submit_job_with_collection(self) -> None:
        """Test that SyncEvalHubClient can submit jobs using a collection reference."""
        from evalhub.models.api import JobSubmissionRequest

        client = SyncEvalHubClient()
        mock_job_data = {
            "resource": {
                "id": "job_coll_1",
                "tenant": "default",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
            },
            "name": "collection-eval",
            "description": "Evaluate via collection",
            "tags": ["collection-test"],
            "status": {"state": JobStatus.PENDING.value},
            "model": {"url": "http://localhost:8000/v1", "name": "gpt-3.5-turbo"},
            "collection": {"id": "healthcare_v1"},
        }
        mock_response = Mock()
        mock_response.json.return_value = mock_job_data

        with patch.object(client, "_request", return_value=mock_response):
            model = ModelConfig(url="http://localhost:8000/v1", name="gpt-3.5-turbo")
            request = JobSubmissionRequest(
                name="collection-eval",
                description="Evaluate via collection",
                tags=["collection-test"],
                model=model,
                collection=CollectionRef(id="healthcare_v1"),
            )
            job = client.jobs.submit(request)
            assert isinstance(job, EvaluationJob)
            assert job.name == "collection-eval"
            assert job.benchmarks is None
            assert job.collection is not None
            assert job.collection.id == "healthcare_v1"

        client.close()

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode - would create actual jobs",
    )
    def test_sync_client_submit_job_with_exports_oci(self) -> None:
        """Test that SyncEvalHubClient can submit jobs with OCI exports configuration."""
        client = SyncEvalHubClient()
        mock_job_data = {
            "resource": {
                "id": "job_oci_1",
                "tenant": "default",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
            },
            "name": "oci-export-eval",
            "description": "Evaluate with OCI exports",
            "tags": [],
            "status": {"state": JobStatus.PENDING.value},
            "model": {"url": "http://localhost:8000/v1", "name": "test-model"},
            "benchmarks": [{"id": "mmlu", "provider_id": "lm_eval", "parameters": {}}],
            "exports": {
                "oci": {
                    "coordinates": {
                        "oci_host": "quay.io",
                        "oci_repository": "my-org/my-repo",
                        "oci_tag": "eval-123",
                    },
                    "k8s": {"connection": "my-pull-secret"},
                }
            },
        }
        mock_response = Mock()
        mock_response.json.return_value = mock_job_data

        with patch.object(client, "_request", return_value=mock_response):
            request = JobSubmissionRequest(
                name="oci-export-eval",
                description="Evaluate with OCI exports",
                model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
                benchmarks=[
                    BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
                ],
                exports=EvaluationExports(
                    oci=EvaluationExportsOCI(
                        coordinates=OCICoordinates(
                            oci_host="quay.io",
                            oci_repository="my-org/my-repo",
                            oci_tag="eval-123",
                        ),
                        k8s=OCIConnectionConfig(connection="my-pull-secret"),
                    ),
                ),
            )
            job = client.jobs.submit(request)
            assert isinstance(job, EvaluationJob)
            assert job.name == "oci-export-eval"

        client.close()

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode",
    )
    @pytest.mark.parametrize("kwargs, expected_params", JOBS_LIST_PARAM_CASES)
    def test_sync_jobs_list_query_params(
        self, kwargs: dict[str, Any], expected_params: dict[str, str]
    ) -> None:
        """Test that jobs.list sends correct query parameter names to the server."""
        with SyncEvalHubClient() as client:
            mock_response = Mock()
            mock_response.json.return_value = {"total_count": 0, "items": []}

            with patch.object(
                client, "_request_get", return_value=mock_response
            ) as mock_get:
                client.jobs.list(**kwargs)
                mock_get.assert_called_once()
                _, call_kwargs = mock_get.call_args
                assert call_kwargs.get("params", {}) == expected_params

    @pytest.mark.skipif(
        EVALHUB_TEST_BASE_URL is not None,
        reason="Skipping in real server mode",
    )
    @pytest.mark.asyncio
    @pytest.mark.parametrize("kwargs, expected_params", JOBS_LIST_PARAM_CASES)
    async def test_async_jobs_list_query_params(
        self, kwargs: dict[str, Any], expected_params: dict[str, str]
    ) -> None:
        """Test that async jobs.list sends correct query parameter names to the server."""
        async with AsyncEvalHubClient() as client:
            mock_response = Mock()
            mock_response.json.return_value = {"total_count": 0, "items": []}

            with patch.object(
                client, "_request_get", return_value=mock_response
            ) as mock_get:
                await client.jobs.list(**kwargs)
                mock_get.assert_called_once()
                _, call_kwargs = mock_get.call_args
                assert call_kwargs.get("params", {}) == expected_params

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


class TestAgentMetadataModels:
    """Unit tests for agent-discoverability model round-trips."""

    def test_provider_agent_metadata_round_trip(self) -> None:
        data: dict[str, Any] = {
            "resource": {"id": "test"},
            "name": "Test Provider",
            "agent": {
                "evaluates": ["data-ingestion", "metrics-reporting"],
                "recommended_when": ["User wants to validate the pipeline"],
                "target_type": "model",
                "summary": "Validate the EvalHub pipeline",
                "complements": [],
                "hints": ["Requires JSON data mounted at /data"],
                "result_interpretation": ["row_count: number of records"],
            },
        }
        provider = Provider(**data)
        assert provider.agent is not None
        assert provider.agent.target_type == "model"
        assert provider.agent.evaluates == ["data-ingestion", "metrics-reporting"]
        assert provider.agent.hints == ["Requires JSON data mounted at /data"]
        assert provider.agent.result_interpretation == ["row_count: number of records"]

    def test_provider_without_agent_defaults_to_none(self) -> None:
        provider = Provider(resource=Resource(id="p"), name="P")
        assert provider.agent is None

    def test_benchmark_agent_metadata_round_trip(self) -> None:
        data: dict[str, Any] = {
            "id": "datafile",
            "name": "Datafile",
            "category": "general",
            "agent": {
                "result_interpretation": "row_count shows records read",
                "score_ranges": [
                    {"range": "0-50", "meaning": "Low data volume"},
                    {"range": "51-100", "meaning": "Expected range"},
                ],
            },
        }
        benchmark = Benchmark(**data)
        assert benchmark.agent is not None
        assert benchmark.agent.result_interpretation == "row_count shows records read"
        assert len(benchmark.agent.score_ranges) == 2
        assert benchmark.agent.score_ranges[0].range == "0-50"
        assert benchmark.agent.score_ranges[0].meaning == "Low data volume"
        assert benchmark.agent.score_ranges[1].range == "51-100"

    def test_benchmark_without_agent_defaults_to_none(self) -> None:
        benchmark = Benchmark(
            id="b", name="B", category="general", primary_score=None, pass_criteria=None
        )
        assert benchmark.agent is None

    def test_score_range_model(self) -> None:
        sr = ScoreRange(range="0-100", meaning="full scale")
        assert sr.range == "0-100"
        assert sr.meaning == "full scale"

    def test_benchmark_agent_metadata_empty_score_ranges(self) -> None:
        bam = BenchmarkAgentMetadata(result_interpretation="some text")
        assert bam.score_ranges == []

    def test_agent_metadata_defaults(self) -> None:
        am = AgentMetadata()
        assert am.evaluates == []
        assert am.recommended_when == []
        assert am.target_type is None
        assert am.summary is None
        assert am.complements == []
        assert am.hints == []
        assert am.result_interpretation == []


class TestProvidersListFiltering:
    """Unit tests for client-side filtering in providers.list()."""

    def _make_providers_response(self) -> dict:
        return {
            "total_count": 3,
            "items": [
                {
                    "resource": {"id": "test"},
                    "name": "Test",
                    "agent": {
                        "target_type": "model",
                        "evaluates": ["data-ingestion", "metrics-reporting"],
                    },
                },
                {
                    "resource": {"id": "garak"},
                    "name": "Garak",
                    "agent": {
                        "target_type": "agent",
                        "evaluates": ["safety"],
                    },
                },
                {
                    "resource": {"id": "ragas"},
                    "name": "RAGAS",
                },
            ],
        }

    def test_list_no_filter_returns_all(self) -> None:
        with SyncEvalHubClient() as client:
            mock_resp = Mock()
            mock_resp.json.return_value = self._make_providers_response()
            with patch.object(client, "_request_get", return_value=mock_resp):
                result = client.providers.list()
        assert len(result) == 3

    def test_list_filter_by_target_type_model(self) -> None:
        with SyncEvalHubClient() as client:
            mock_resp = Mock()
            mock_resp.json.return_value = self._make_providers_response()
            with patch.object(client, "_request_get", return_value=mock_resp):
                result = client.providers.list(target_type="model")
        assert len(result) == 1
        assert result[0].resource.id == "test"

    def test_list_filter_by_target_type_agent(self) -> None:
        with SyncEvalHubClient() as client:
            mock_resp = Mock()
            mock_resp.json.return_value = self._make_providers_response()
            with patch.object(client, "_request_get", return_value=mock_resp):
                result = client.providers.list(target_type="agent")
        assert len(result) == 1
        assert result[0].resource.id == "garak"

    def test_list_filter_by_evaluates(self) -> None:
        with SyncEvalHubClient() as client:
            mock_resp = Mock()
            mock_resp.json.return_value = self._make_providers_response()
            with patch.object(client, "_request_get", return_value=mock_resp):
                result = client.providers.list(evaluates="data-ingestion")
        assert len(result) == 1
        assert result[0].resource.id == "test"

    def test_list_filter_by_evaluates_no_match(self) -> None:
        with SyncEvalHubClient() as client:
            mock_resp = Mock()
            mock_resp.json.return_value = self._make_providers_response()
            with patch.object(client, "_request_get", return_value=mock_resp):
                result = client.providers.list(evaluates="hallucination")
        assert result == []

    def test_list_filter_excludes_providers_without_agent(self) -> None:
        with SyncEvalHubClient() as client:
            mock_resp = Mock()
            mock_resp.json.return_value = self._make_providers_response()
            with patch.object(client, "_request_get", return_value=mock_resp):
                result = client.providers.list(target_type="model")
        ids = [p.resource.id for p in result]
        assert "ragas" not in ids
