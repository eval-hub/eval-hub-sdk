from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from evalhub.mcp.server import app, set_client
from evalhub.models import (
    Benchmark,
    BenchmarkConfig,
    EvaluationJob,
    EvaluationJobResource,
    EvaluationJobStatus,
    JobStatus,
    ModelConfig,
    Provider,
    Resource,
)
from inline_snapshot import snapshot
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest_asyncio.fixture(loop_scope="function")
async def client_session() -> AsyncGenerator[ClientSession, None]:
    """Create a client session connected to the MCP server with mocked data."""
    mock_client = MagicMock()
    mock_provider = Provider(
        resource=Resource(id="test-provider"),
        name="Test Provider",
        description="A test provider",
        benchmarks=[],
    )
    mock_benchmark = Benchmark(
        id="test-benchmark",
        name="Test Benchmark",
        description="A test benchmark",
        category="test-category",
        metrics=["accuracy"],
        num_few_shot=5,
        dataset_size=100,
        tags=["test"],
        primary_score=None,
        pass_criteria=None,
    )
    mock_providers_resource = MagicMock()
    mock_providers_resource.list = AsyncMock(return_value=[mock_provider])
    mock_client.providers = mock_providers_resource
    mock_benchmarks_resource = MagicMock()
    mock_benchmarks_resource.list = AsyncMock(return_value=[mock_benchmark])
    mock_client.benchmarks = mock_benchmarks_resource
    mock_job = EvaluationJob(
        resource=EvaluationJobResource(
            id="test-job-123",
            tenant="test-tenant",
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            updated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        ),
        status=EvaluationJobStatus(
            state=JobStatus.PENDING,
        ),
        model=ModelConfig(url="http://test-model.com", name="test-model"),
        benchmarks=[
            BenchmarkConfig(
                id="test-benchmark",
                provider_id="test-provider",
                parameters={},
            )
        ],
    )
    mock_jobs_resource = MagicMock()
    mock_jobs_resource.submit = AsyncMock(return_value=mock_job)
    mock_client.jobs = mock_jobs_resource

    # Inject the mock client into the MCP server
    set_client(mock_client)

    # Use the helper function with error suppression for teardown
    try:
        async with create_connected_server_and_client_session(
            app, raise_exceptions=True
        ) as session:
            yield session
    except RuntimeError as e:
        # Suppress anyio cancel scope errors during teardown
        if "cancel scope" not in str(e):
            raise
    finally:
        # Clean up the mock client
        set_client(None)


@pytest.mark.anyio
async def test_list_resources(client_session: ClientSession) -> None:
    result = await client_session.list_resources()
    assert result.model_dump(mode="json") == snapshot(
        {
            "meta": None,
            "nextCursor": None,
            "resources": [
                {
                    "name": "List Providers",
                    "title": None,
                    "uri": "evalhub://providers",
                    "description": "List all registered EvalHub providers",
                    "mimeType": "application/json",
                    "size": None,
                    "icons": None,
                    "annotations": None,
                    "meta": None,
                },
                {
                    "name": "List Benchmarks",
                    "title": None,
                    "uri": "evalhub://benchmarks",
                    "description": "List all available EvalHub benchmarks",
                    "mimeType": "application/json",
                    "size": None,
                    "icons": None,
                    "annotations": None,
                    "meta": None,
                },
            ],
        }
    )


@pytest.mark.anyio
async def test_create_evaluation_job_tool(client_session: ClientSession) -> None:
    """Test the create_evaluation_job MCP tool."""
    result = await client_session.call_tool(
        "create_evaluation_job",
        {
            "model_url": "http://test-model.com",
            "model_name": "test-model",
            "benchmarks": [
                {
                    "benchmark_id": "test-benchmark",
                    "provider_id": "test-provider",
                    "parameters": {},
                }
            ],
        },
    )

    assert result.model_dump(mode="json") == snapshot(
        {
            "meta": None,
            "content": [
                {
                    "type": "text",
                    "text": snapshot(
                        """\
Successfully created evaluation job!

{
  "job_id": "test-job-123",
  "state": "pending",
  "model": {
    "url": "http://test-model.com",
    "name": "test-model"
  },
  "benchmarks": [
    {
      "id": "test-benchmark",
      "provider_id": "test-provider",
      "parameters": {}
    }
  ],
  "created_at": "2024-01-01T12:00:00+00:00"
}\
"""
                    ),
                    "annotations": None,
                    "meta": None,
                }
            ],
            "structuredContent": None,
            "isError": False,
        }
    )
