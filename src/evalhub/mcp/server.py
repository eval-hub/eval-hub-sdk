"""EvalHub MCP Server - exposes EvalHub Client capabilities via MCP."""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server import FastMCP
from mcp.types import (
    Completion,
    CompletionArgument,
    CompletionContext,
    PromptReference,
    ResourceTemplateReference,
)

from ..client.evalhub import AsyncEvalHubClient
from ..models import (
    BenchmarkConfig,
    CollectionRef,
    ExperimentConfig,
    JobStatus,
    JobSubmissionRequest,
    ModelConfig,
)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "evalhub",
    instructions=(
        "EvalHub MCP Server provides access to the EvalHub evaluation service. "
        "Use resources to browse providers, benchmarks, collections, and jobs. "
        "Use tools to submit evaluation jobs or cancel running ones."
    ),
)

_client: AsyncEvalHubClient | None = None


def set_client(client: AsyncEvalHubClient) -> None:
    """Set the AsyncEvalHubClient instance for the MCP server to use."""
    global _client
    _client = client


def _get_client() -> AsyncEvalHubClient:
    if _client is None:
        raise RuntimeError("EvalHub client not initialized. Call set_client() first.")
    return _client


def _serialize_list(items: list[Any]) -> str:
    """Serialize a list of Pydantic models to JSON."""
    return json.dumps(
        {
            "items": [item.model_dump(mode="json") for item in items],
            "count": len(items),
        },
        indent=2,
    )


def _serialize_model(model: Any) -> str:
    """Serialize a single Pydantic model to JSON."""
    return json.dumps(model.model_dump(mode="json"), indent=2)


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------


@mcp.resource(
    "evalhub://providers",
    name="providers",
    description="List all registered evaluation providers",
)
async def list_providers() -> str:
    client = _get_client()
    providers = await client.providers.list()
    return _serialize_list(providers)


@mcp.resource(
    "evalhub://providers/{provider_id}",
    name="provider",
    description="Get a specific provider by ID, including its benchmarks",
)
async def get_provider(provider_id: str) -> str:
    client = _get_client()
    provider = await client.providers.get(provider_id)
    return _serialize_model(provider)


@mcp.resource(
    "evalhub://providers/{provider_id}/benchmarks",
    name="provider_benchmarks",
    description="List benchmarks available from a specific provider",
)
async def list_provider_benchmarks(provider_id: str) -> str:
    client = _get_client()
    benchmarks = await client.benchmarks.list(provider_id=provider_id)
    return _serialize_list(benchmarks)


@mcp.resource(
    "evalhub://benchmarks",
    name="benchmarks",
    description="List all available benchmarks",
)
async def list_benchmarks() -> str:
    client = _get_client()
    benchmarks = await client.benchmarks.list()
    return _serialize_list(benchmarks)


@mcp.resource(
    "evalhub://collections",
    name="collections",
    description="List all benchmark collections",
)
async def list_collections() -> str:
    client = _get_client()
    collections = await client.collections.list()
    return _serialize_list(collections)


@mcp.resource(
    "evalhub://collections/{collection_id}",
    name="collection",
    description="Get a specific benchmark collection by ID",
)
async def get_collection(collection_id: str) -> str:
    client = _get_client()
    collection = await client.collections.get(collection_id)
    return _serialize_model(collection)


@mcp.resource("evalhub://jobs", name="jobs", description="List all evaluation jobs")
async def list_jobs() -> str:
    client = _get_client()
    jobs = await client.jobs.list()
    return _serialize_list(jobs)


@mcp.resource(
    "evalhub://jobs/{job_id}",
    name="job",
    description="Get a specific evaluation job by ID",
)
async def get_job(job_id: str) -> str:
    client = _get_client()
    job = await client.jobs.get(job_id)
    return _serialize_model(job)


@mcp.resource(
    "evalhub://jobs?status={status}",
    name="jobs_by_status",
    description="List evaluation jobs filtered by status (pending, running, completed, failed, cancelled)",
)
async def list_jobs_by_status(status: str) -> str:
    client = _get_client()
    try:
        job_status = JobStatus(status)
    except ValueError:
        valid = ", ".join(s.value for s in JobStatus)
        return json.dumps(
            {
                "error": f"Invalid status '{status}'. Valid values: {valid}",
                "items": [],
                "count": 0,
            }
        )
    jobs = await client.jobs.list(status=job_status)
    return _serialize_list(jobs)


# ---------------------------------------------------------------------------
# MCP Completions
# ---------------------------------------------------------------------------


@mcp.completion()
async def handle_completion(
    ref: PromptReference | ResourceTemplateReference,
    argument: CompletionArgument,
    context: CompletionContext | None,
) -> Completion | None:
    """Provide autocompletion for resource template parameters."""
    if not isinstance(ref, ResourceTemplateReference):
        return None

    client = _get_client()
    uri = ref.uri
    name = argument.name
    partial = argument.value.lower()

    try:
        if name == "provider_id" and "providers" in uri:
            providers = await client.providers.list()
            ids = [p.resource.id for p in providers]
            values = [v for v in ids if v.lower().startswith(partial)]
            return Completion(values=values)

        if name == "collection_id" and "collections" in uri:
            collections = await client.collections.list()
            ids = [c.resource.id for c in collections]
            values = [v for v in ids if v.lower().startswith(partial)]
            return Completion(values=values)

        if name == "job_id" and "jobs" in uri:
            jobs = await client.jobs.list()
            ids = [j.resource.id for j in jobs]
            values = [v for v in ids if v.lower().startswith(partial)]
            return Completion(values=values)

        if name == "status" and "jobs" in uri:
            values = [s.value for s in JobStatus if s.value.startswith(partial)]
            return Completion(values=values)
    except Exception:
        logger.debug("Completion lookup failed", exc_info=True)

    return None


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="submit_evaluation",
    description=(
        "Submit a new evaluation job to EvalHub. "
        "Provide either 'benchmarks' or 'collection', not both. "
        "Use the providers and benchmarks resources to discover available provider_id and benchmark id values."
    ),
)
async def submit_evaluation(
    name: str,
    model: dict[str, Any],
    benchmarks: list[dict[str, Any]] | None = None,
    collection: dict[str, Any] | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    experiment: dict[str, Any] | None = None,
) -> str:
    """Submit a new evaluation job.

    Evaluation Job fields have been separated for easier fill by AI agents.

    Args:
        name: Job name.
        model: Model to evaluate. Keys: "url" (model endpoint), "name" (model identifier),
            optional "auth" with "secret_ref" (Kubernetes Secret name for model credentials).
            Examples:
              Remote vLLM:  {"url": "http://vllm-server.models.svc.cluster.local:8000/v1", "name": "meta-llama/Llama-3.2-1B-Instruct"}
              With auth:    {"url": "http://model:8000/v1", "name": "my-model", "auth": {"secret_ref": "model-api-key"}}
        benchmarks: List of benchmarks to run. Each entry has "id", "provider_id", and optional "parameters".
            Mutually exclusive with 'collection'.
            Examples:
              Simple:     [{"id": "demo_benchmark", "provider_id": "demo"}]
              With params: [{"id": "quick_perf_test", "provider_id": "guidellm", "parameters": {"profile": "constant", "rate": 5, "max_seconds": 10, "max_requests": 20}}]
              Multiple:   [{"id": "gsm8k", "provider_id": "lm_eval"}, {"id": "mmlu", "provider_id": "lm_eval"}]
        collection: Collection reference to run all benchmarks in a predefined collection.
            Mutually exclusive with 'benchmarks'. Keys: "id" (collection identifier),
            optional "benchmarks" to run only a subset.
            Examples:
              Full collection: {"id": "standard"}
              Subset:          {"id": "standard", "benchmarks": [{"id": "gsm8k", "provider_id": "lm_eval"}]}
        description: Optional job description.
        tags: Optional list of tags for organizing jobs, e.g. ["nightly", "regression"].
        experiment: Optional MLflow experiment config. Keys: "name", optional "tags" (list of {"key": ..., "value": ...}),
            optional "artifact_location".
            Example: {"name": "llama3-eval-experiment", "tags": [{"key": "team", "value": "nlp"}]}
    """
    has_benchmarks = bool(benchmarks)
    has_collection = collection is not None
    if has_benchmarks == has_collection:
        raise ValueError("Provide exactly one of 'benchmarks' or 'collection'.")

    client = _get_client()

    model_config = ModelConfig(**model)

    benchmark_configs = None
    if benchmarks is not None:
        benchmark_configs = [BenchmarkConfig(**b) for b in benchmarks]

    collection_ref = None
    if collection is not None:
        collection_ref = CollectionRef(**collection)

    experiment_config = None
    if experiment is not None:
        experiment_config = ExperimentConfig(**experiment)

    request = JobSubmissionRequest(
        name=name,
        description=description,
        tags=tags or [],
        model=model_config,
        benchmarks=benchmark_configs,
        collection=collection_ref,
        experiment=experiment_config,
    )

    job = await client.jobs.submit(request)
    return _serialize_model(job)


@mcp.tool(
    name="cancel_job",
    description="Cancel a running evaluation job.",
)
async def cancel_job(job_id: str, hard_delete: bool = False) -> str:
    """Cancel an evaluation job.

    Args:
        job_id: The job identifier.
        hard_delete: If true, permanently delete instead of just cancelling.
    """
    client = _get_client()
    success = await client.jobs.cancel(job_id, hard_delete=hard_delete)
    return json.dumps({"job_id": job_id, "cancelled": success})
