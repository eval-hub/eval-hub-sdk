"""Unit tests for the EvalHub MCP Server.

Tests mock the AsyncEvalHubClient and verify that MCP resources and tools
correctly delegate to the client methods.
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from evalhub.mcp.server import (
    _serialize_list,
    _serialize_model,
    cancel_job,
    get_collection,
    get_job,
    get_provider,
    handle_completion,
    list_benchmarks,
    list_collections,
    list_jobs,
    list_jobs_by_status,
    list_provider_benchmarks,
    list_providers,
    mcp,
    set_client,
    submit_evaluation,
)
from evalhub.models.api import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkReference,
    Collection,
    CollectionRef,
    EvaluationJob,
    EvaluationJobResource,
    EvaluationJobStatus,
    ExperimentConfig,
    JobStatus,
    ModelAuth,
    ModelConfig,
    Provider,
    Resource,
)
from mcp.types import CompletionArgument, ResourceTemplateReference

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_resource(resource_id: str = "test-id") -> Resource:
    return Resource(
        id=resource_id,
        tenant="test-tenant",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_provider(provider_id: str = "lm_eval") -> Provider:
    return Provider(
        resource=_make_resource(provider_id),
        name="LM Eval",
        description="LM Evaluation Harness provider",
        benchmarks=[
            Benchmark(
                id="gsm8k",
                name="GSM8K",
                description="Grade school math",
                category="math",
                metrics=["accuracy"],
                num_few_shot=None,
                dataset_size=None,
                primary_score=None,
                pass_criteria=None,
            )
        ],
    )


def _make_benchmark(benchmark_id: str = "gsm8k") -> Benchmark:
    return Benchmark(
        id=benchmark_id,
        name="GSM8K",
        description="Grade school math",
        category="math",
        metrics=["accuracy"],
        num_few_shot=None,
        dataset_size=None,
        primary_score=None,
        pass_criteria=None,
    )


def _make_collection(collection_id: str = "standard") -> Collection:
    return Collection(
        resource=_make_resource(collection_id),
        name="Standard Collection",
        description="A standard benchmark collection",
        benchmarks=[
            BenchmarkReference(id="gsm8k", provider_id="lm_eval"),
        ],
    )


def _make_job(
    job_id: str = "job-123", status: JobStatus = JobStatus.PENDING
) -> EvaluationJob:
    return EvaluationJob(
        resource=EvaluationJobResource(
            id=job_id,
            tenant="test-tenant",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        status=EvaluationJobStatus(state=status),
        name="test-eval",
        model=ModelConfig(url="http://model:8000", name="llama3"),
        benchmarks=[BenchmarkConfig(id="gsm8k", provider_id="lm_eval")],
    )


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AsyncEvalHubClient and set it on the MCP server."""
    client = MagicMock()

    client.providers = MagicMock()
    client.providers.list = AsyncMock(return_value=[_make_provider()])
    client.providers.get = AsyncMock(return_value=_make_provider())

    client.benchmarks = MagicMock()
    client.benchmarks.list = AsyncMock(return_value=[_make_benchmark()])

    client.collections = MagicMock()
    client.collections.list = AsyncMock(return_value=[_make_collection()])
    client.collections.get = AsyncMock(return_value=_make_collection())

    client.jobs = MagicMock()
    client.jobs.list = AsyncMock(return_value=[_make_job()])
    client.jobs.get = AsyncMock(return_value=_make_job())
    client.jobs.submit = AsyncMock(return_value=_make_job())
    client.jobs.cancel = AsyncMock(return_value=True)

    set_client(client)
    return client


# ---------------------------------------------------------------------------
# Resource listing tests
# ---------------------------------------------------------------------------


async def test_list_resources() -> None:
    """Verify static resources are registered."""
    resources = await mcp.list_resources()
    uris = [str(r.uri) for r in resources]
    assert "evalhub://providers" in uris
    assert "evalhub://benchmarks" in uris
    assert "evalhub://collections" in uris
    assert "evalhub://jobs" in uris


async def test_list_resource_templates() -> None:
    """Verify resource templates are registered."""
    templates = await mcp.list_resource_templates()
    uri_templates = [t.uriTemplate for t in templates]
    assert "evalhub://providers/{provider_id}" in uri_templates
    assert "evalhub://providers/{provider_id}/benchmarks" in uri_templates
    assert "evalhub://collections/{collection_id}" in uri_templates
    assert "evalhub://jobs/{job_id}" in uri_templates
    assert "evalhub://jobs?status={status}" in uri_templates


# ---------------------------------------------------------------------------
# Resource read tests
# ---------------------------------------------------------------------------


async def test_read_providers(mock_client: MagicMock) -> None:
    result = await list_providers()
    data = json.loads(result)
    assert data["count"] == 1
    assert data["items"][0]["name"] == "LM Eval"
    mock_client.providers.list.assert_awaited_once()


async def test_read_provider_by_id(mock_client: MagicMock) -> None:
    result = await get_provider("lm_eval")
    data = json.loads(result)
    assert data["name"] == "LM Eval"
    assert data["resource"]["id"] == "lm_eval"
    mock_client.providers.get.assert_awaited_once_with("lm_eval")


async def test_read_benchmarks(mock_client: MagicMock) -> None:
    result = await list_benchmarks()
    data = json.loads(result)
    assert data["count"] == 1
    assert data["items"][0]["id"] == "gsm8k"
    mock_client.benchmarks.list.assert_awaited_once()


async def test_read_benchmarks_by_provider(mock_client: MagicMock) -> None:
    result = await list_provider_benchmarks("lm_eval")
    data = json.loads(result)
    assert data["count"] == 1
    mock_client.benchmarks.list.assert_awaited_once_with(provider_id="lm_eval")


async def test_read_collections(mock_client: MagicMock) -> None:
    result = await list_collections()
    data = json.loads(result)
    assert data["count"] == 1
    assert data["items"][0]["name"] == "Standard Collection"
    mock_client.collections.list.assert_awaited_once()


async def test_read_collection_by_id(mock_client: MagicMock) -> None:
    result = await get_collection("standard")
    data = json.loads(result)
    assert data["name"] == "Standard Collection"
    mock_client.collections.get.assert_awaited_once_with("standard")


async def test_read_jobs(mock_client: MagicMock) -> None:
    result = await list_jobs()
    data = json.loads(result)
    assert data["count"] == 1
    assert data["items"][0]["name"] == "test-eval"
    mock_client.jobs.list.assert_awaited_once()


async def test_read_job_by_id(mock_client: MagicMock) -> None:
    result = await get_job("job-123")
    data = json.loads(result)
    assert data["name"] == "test-eval"
    mock_client.jobs.get.assert_awaited_once_with("job-123")


async def test_read_jobs_by_status(mock_client: MagicMock) -> None:
    mock_client.jobs.list = AsyncMock(
        return_value=[_make_job(status=JobStatus.RUNNING)]
    )
    result = await list_jobs_by_status("running")
    data = json.loads(result)
    assert data["count"] == 1
    mock_client.jobs.list.assert_awaited_once_with(status=JobStatus.RUNNING)


async def test_read_jobs_by_invalid_status(mock_client: MagicMock) -> None:
    result = await list_jobs_by_status("bogus")
    data = json.loads(result)
    assert data["count"] == 0
    assert data["items"] == []
    assert "Invalid status" in data["error"]
    assert "bogus" in data["error"]
    mock_client.jobs.list.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tool listing tests
# ---------------------------------------------------------------------------


async def test_list_tools() -> None:
    """Verify tools are registered."""
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "submit_evaluation" in tool_names
    assert "cancel_job" in tool_names
    assert len(tool_names) == 2


async def test_submit_evaluation_schema() -> None:
    """Verify the generated inputSchema contains typed $defs for Pydantic models."""
    tools = await mcp.list_tools()
    tool = next(t for t in tools if t.name == "submit_evaluation")
    schema = tool.inputSchema

    # Required top-level params
    assert "name" in schema["required"]
    assert "model" in schema["required"]

    # Pydantic models generate $defs with full property definitions
    defs = schema["$defs"]
    assert "ModelConfig" in defs
    assert "BenchmarkConfig" in defs

    # ModelConfig has url and name as required
    model_def = defs["ModelConfig"]
    assert "url" in model_def["properties"]
    assert "name" in model_def["properties"]
    assert "url" in model_def["required"]
    assert "name" in model_def["required"]

    # BenchmarkConfig has id and provider_id as required
    bench_def = defs["BenchmarkConfig"]
    assert "id" in bench_def["properties"]
    assert "provider_id" in bench_def["properties"]
    assert "id" in bench_def["required"]
    assert "provider_id" in bench_def["required"]


async def test_submit_evaluation_wire_path(mock_client: MagicMock) -> None:
    """Invoke submit_evaluation through FastMCP's call_tool with JSON-like dicts."""
    await mcp.call_tool(
        "submit_evaluation",
        {
            "name": "wire-eval",
            "model": {"url": "http://model:8000/v1", "name": "llama3"},
            "benchmarks": [
                {"id": "gsm8k", "provider_id": "lm_eval"},
                {
                    "id": "mmlu",
                    "provider_id": "lm_eval",
                    "parameters": {"num_few_shot": 5},
                },
            ],
            "experiment": {
                "name": "my-experiment",
                "tags": [{"key": "team", "value": "nlp"}],
            },
        },
    )

    mock_client.jobs.submit.assert_awaited_once()
    request = mock_client.jobs.submit.call_args[0][0]
    assert isinstance(request.model, ModelConfig)
    assert request.model.url == "http://model:8000/v1"
    assert request.model.name == "llama3"
    assert len(request.benchmarks) == 2
    assert isinstance(request.benchmarks[0], BenchmarkConfig)
    assert request.benchmarks[0].id == "gsm8k"
    assert request.benchmarks[1].parameters == {"num_few_shot": 5}
    assert isinstance(request.experiment, ExperimentConfig)
    assert request.experiment.name == "my-experiment"
    assert len(request.experiment.tags) == 1
    assert request.experiment.tags[0].key == "team"


# ---------------------------------------------------------------------------
# Tool call tests (direct invocation)
# ---------------------------------------------------------------------------


async def test_submit_evaluation(mock_client: MagicMock) -> None:
    result = await submit_evaluation(
        name="my-eval",
        model=ModelConfig(url="http://model:8000", name="llama3"),
        benchmarks=[BenchmarkConfig(id="gsm8k", provider_id="lm_eval")],
    )
    data = json.loads(result)
    assert data["name"] == "test-eval"

    mock_client.jobs.submit.assert_awaited_once()
    call_args = mock_client.jobs.submit.call_args
    request = call_args[0][0]
    assert request.name == "my-eval"
    assert request.model.url == "http://model:8000"
    assert request.model.name == "llama3"
    assert len(request.benchmarks) == 1
    assert request.benchmarks[0].id == "gsm8k"
    assert request.benchmarks[0].provider_id == "lm_eval"


async def test_submit_evaluation_with_collection(mock_client: MagicMock) -> None:
    result = await submit_evaluation(
        name="collection-eval",
        model=ModelConfig(url="http://model:8000", name="llama3"),
        collection=CollectionRef(id="standard"),
    )
    json.loads(result)  # validate JSON output

    call_args = mock_client.jobs.submit.call_args
    request = call_args[0][0]
    assert request.name == "collection-eval"
    assert request.benchmarks is None
    assert request.collection.id == "standard"


async def test_submit_evaluation_with_model_auth(mock_client: MagicMock) -> None:
    await submit_evaluation(
        name="auth-eval",
        model=ModelConfig(
            url="http://model:8000",
            name="llama3",
            auth=ModelAuth(secret_ref="my-secret"),
        ),
        benchmarks=[BenchmarkConfig(id="gsm8k", provider_id="lm_eval")],
    )

    call_args = mock_client.jobs.submit.call_args
    request = call_args[0][0]
    assert request.model.auth is not None
    assert request.model.auth.secret_ref == "my-secret"


async def test_submit_evaluation_with_experiment(mock_client: MagicMock) -> None:
    await submit_evaluation(
        name="exp-eval",
        model=ModelConfig(url="http://model:8000", name="llama3"),
        benchmarks=[BenchmarkConfig(id="gsm8k", provider_id="lm_eval")],
        experiment=ExperimentConfig(name="my-experiment"),
    )

    call_args = mock_client.jobs.submit.call_args
    request = call_args[0][0]
    assert request.experiment is not None
    assert request.experiment.name == "my-experiment"


async def test_submit_evaluation_both_benchmarks_and_collection(
    mock_client: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        await submit_evaluation(
            name="bad-eval",
            model=ModelConfig(url="http://model:8000", name="llama3"),
            benchmarks=[BenchmarkConfig(id="gsm8k", provider_id="lm_eval")],
            collection=CollectionRef(id="standard"),
        )


async def test_submit_evaluation_neither_benchmarks_nor_collection(
    mock_client: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        await submit_evaluation(
            name="bad-eval",
            model=ModelConfig(url="http://model:8000", name="llama3"),
        )


async def test_submit_evaluation_empty_benchmarks(
    mock_client: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        await submit_evaluation(
            name="bad-eval",
            model=ModelConfig(url="http://model:8000", name="llama3"),
            benchmarks=[],
        )


async def test_cancel_job_tool(mock_client: MagicMock) -> None:
    result = await cancel_job("job-123")
    data = json.loads(result)
    assert data["job_id"] == "job-123"
    assert data["cancelled"] is True
    mock_client.jobs.cancel.assert_awaited_once_with("job-123", hard_delete=False)


async def test_cancel_job_hard_delete(mock_client: MagicMock) -> None:
    result = await cancel_job("job-123", hard_delete=True)
    data = json.loads(result)
    assert data["cancelled"] is True
    mock_client.jobs.cancel.assert_awaited_once_with("job-123", hard_delete=True)


# ---------------------------------------------------------------------------
# Completion tests
# ---------------------------------------------------------------------------


def _template_ref(uri: str) -> ResourceTemplateReference:
    return ResourceTemplateReference(type="ref/resource", uri=uri)


def _arg(name: str, value: str = "") -> CompletionArgument:
    return CompletionArgument(name=name, value=value)


async def test_completion_provider_id(mock_client: MagicMock) -> None:
    mock_client.providers.list = AsyncMock(
        return_value=[_make_provider("lm_eval"), _make_provider("ragas")]
    )
    result = await handle_completion(
        _template_ref("evalhub://providers/{provider_id}"),
        _arg("provider_id"),
        None,
    )
    assert result is not None
    assert "lm_eval" in result.values
    assert "ragas" in result.values


async def test_completion_provider_id_partial(mock_client: MagicMock) -> None:
    mock_client.providers.list = AsyncMock(
        return_value=[_make_provider("lm_eval"), _make_provider("ragas")]
    )
    result = await handle_completion(
        _template_ref("evalhub://providers/{provider_id}"),
        _arg("provider_id", "lm"),
        None,
    )
    assert result is not None
    assert result.values == ["lm_eval"]


async def test_completion_collection_id(mock_client: MagicMock) -> None:
    mock_client.collections.list = AsyncMock(
        return_value=[_make_collection("standard"), _make_collection("safety")]
    )
    result = await handle_completion(
        _template_ref("evalhub://collections/{collection_id}"),
        _arg("collection_id"),
        None,
    )
    assert result is not None
    assert "standard" in result.values
    assert "safety" in result.values


async def test_completion_job_id(mock_client: MagicMock) -> None:
    mock_client.jobs.list = AsyncMock(
        return_value=[_make_job("job-1"), _make_job("job-2")]
    )
    result = await handle_completion(
        _template_ref("evalhub://jobs/{job_id}"),
        _arg("job_id"),
        None,
    )
    assert result is not None
    assert "job-1" in result.values
    assert "job-2" in result.values


async def test_completion_status(mock_client: MagicMock) -> None:
    result = await handle_completion(
        _template_ref("evalhub://jobs?status={status}"),
        _arg("status", "run"),
        None,
    )
    assert result is not None
    assert result.values == ["running"]


async def test_completion_returns_none_for_prompts(mock_client: MagicMock) -> None:
    from mcp.types import PromptReference

    result = await handle_completion(
        PromptReference(type="ref/prompt", name="something"),
        _arg("arg"),
        None,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


def test_serialize_model() -> None:
    provider = _make_provider()
    result = json.loads(_serialize_model(provider))
    assert result["name"] == "LM Eval"
    assert result["resource"]["id"] == "lm_eval"


def test_serialize_list() -> None:
    providers = [_make_provider("p1"), _make_provider("p2")]
    result = json.loads(_serialize_list(providers))
    assert result["count"] == 2
    assert len(result["items"]) == 2
