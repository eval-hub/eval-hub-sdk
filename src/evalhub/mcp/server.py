"""MCP server for EvalHub SDK.

This module provides an MCP (Model Context Protocol) server that exposes
EvalHub provider and benchmark operations as MCP resources.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Sequence

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import AnyUrl
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from ..client import AsyncEvalHubClient
from ..models import BenchmarkConfig, JobSubmissionRequest, ModelConfig

logger = logging.getLogger(__name__)

# Create MCP server instance
app = Server("EvalHub SDK client based MCP Server")

# Global client instance (initialized in main or by user)
_client: AsyncEvalHubClient | None = None


def set_client(client: AsyncEvalHubClient | None) -> None:
    """Set the AsyncEvalHubClient instance to use for MCP resources.

    Args:
        client: The AsyncEvalHubClient instance or None to clear
    """
    global _client
    _client = client


@app.list_resources()
async def list_resources() -> list[types.Resource]:
    """List all available MCP resources.

    Returns:
        List of resource definitions
    """
    if _client is None:
        return []

    return [
        types.Resource(
            uri=AnyUrl("evalhub://providers"),
            name="List Providers",
            description="List all registered EvalHub providers",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("evalhub://benchmarks"),
            name="List Benchmarks",
            description="List all available EvalHub benchmarks",
            mimeType="application/json",
        ),
    ]


@app.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    """List all available MCP resource templates.

    Returns:
        List of resource template definitions
    """
    if _client is None:
        return []

    return [
        types.ResourceTemplate(
            uriTemplate="evalhub://providers/{provider_id}",
            name="Get Provider",
            description="Get information about a specific EvalHub provider by ID",
            mimeType="application/json",
        ),
    ]


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available MCP tools.

    Returns:
        List of tool definitions
    """
    if _client is None:
        return []

    return [
        types.Tool(
            name="create_evaluation_job",
            description="Create a new evaluation job to evaluate a model on specified benchmarks",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_url": {
                        "type": "string",
                        "description": "Model endpoint URL (e.g., vLLM or OpenAI-compatible endpoint)",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Model name or identifier",
                    },
                    "benchmarks": {
                        "type": "array",
                        "description": "List of benchmarks to evaluate",
                        "items": {
                            "type": "object",
                            "properties": {
                                "benchmark_id": {
                                    "type": "string",
                                    "description": "Benchmark identifier",
                                },
                                "provider_id": {
                                    "type": "string",
                                    "description": "Provider identifier",
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Benchmark-specific parameters",
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["benchmark_id", "provider_id"],
                        },
                        "minItems": 1,
                    },
                    "timeout_minutes": {
                        "type": "integer",
                        "description": "Job timeout in minutes (optional)",
                    },
                    "retry_attempts": {
                        "type": "integer",
                        "description": "Number of retry attempts on failure (optional)",
                    },
                },
                "required": ["model_url", "model_name", "benchmarks"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls.

    Args:
        name: The tool name
        arguments: Tool arguments

    Returns:
        List of text content with tool results

    Raises:
        ValueError: If client is not initialized or tool is unknown
        httpx.HTTPError: If API request fails
    """
    if _client is None:
        raise ValueError("Client not initialized. Call set_client() first.")

    if name == "create_evaluation_job":
        # Extract parameters
        model_url = arguments["model_url"]
        model_name = arguments["model_name"]
        benchmarks_data = arguments["benchmarks"]

        # Build the job submission request
        model = ModelConfig(url=model_url, name=model_name)
        benchmarks = [
            BenchmarkConfig(
                id=b["benchmark_id"],
                provider_id=b["provider_id"],
                parameters=b.get("parameters", {}),
            )
            for b in benchmarks_data
        ]

        request = JobSubmissionRequest(
            model=model,
            benchmarks=benchmarks,
        )

        # Submit the job
        job = await _client.jobs.submit(request)

        # Return the job information
        result = {
            "job_id": job.id,
            "state": job.state.value,
            "model": {"url": job.model.url, "name": job.model.name},
            "benchmarks": [
                {
                    "id": b.id,
                    "provider_id": b.provider_id,
                    "parameters": b.parameters,
                }
                for b in job.benchmarks
            ],
            "created_at": job.resource.created_at.isoformat(),
        }

        return [
            types.TextContent(
                type="text",
                text=f"Successfully created evaluation job!\n\n{json.dumps(result, indent=2)}",
            )
        ]
    else:
        raise ValueError(f"Unknown tool: {name}")


@app.completion()
async def handle_completion(
    ref: types.PromptReference | types.ResourceTemplateReference,
    argument: types.CompletionArgument,
    context: types.CompletionContext | None,
) -> types.Completion | None:
    """Provide completion suggestions for resource template parameters.

    Args:
        ref: The prompt or resource template being completed
        argument: The argument that needs completion
        context: Optional context with previously resolved values

    Returns:
        Completion suggestions or None if no suggestions available
    """
    if _client is None:
        return None

    # Handle resource template completions
    if isinstance(ref, types.ResourceTemplateReference):
        uri_template = str(ref.uri)

        # Provide provider ID completions, here can be optimized later
        if (
            uri_template == "evalhub://providers/{provider_id}"
            and argument.name == "provider_id"
        ):
            providers = await _client.providers.list()
            provider_ids = sorted([provider.resource.id for provider in providers])
            return types.Completion(
                values=provider_ids,
                total=len(provider_ids),
                hasMore=False,
            )

    return None


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a specific MCP resource.

    Args:
        uri: The resource URI to read

    Returns:
        JSON-serialized resource data

    Raises:
        ValueError: If client is not initialized or URI is invalid
        httpx.HTTPError: If API request fails
    """
    if _client is None:
        raise ValueError("Client not initialized. Call set_client() first.")

    # Convert uri to string if it's a Pydantic AnyUrl object
    uri_str = str(uri)

    if uri_str == "evalhub://providers":
        providers = await _client.providers.list()
        return json.dumps([p.model_dump() for p in providers])
    elif uri_str == "evalhub://benchmarks":
        benchmarks = await _client.benchmarks.list()
        return json.dumps([b.model_dump() for b in benchmarks])
    elif uri_str.startswith("evalhub://providers/"):
        provider_id = uri_str.replace("evalhub://providers/", "")
        # TODO: API seems doesn't support individual provider fetch, so filter from list
        providers = await _client.providers.list()
        provider = next((p for p in providers if p.resource.id == provider_id), None)
        if provider is None:
            raise ValueError(f"Provider not found: {provider_id}")
        return provider.model_dump_json()
    else:
        raise ValueError(f"Unknown resource URI: {uri_str}")


def run_server(
    base_url: str | None = None,
    host: str = "0.0.0.0",
    port: int = 3001,
    json_response: bool = True,
    log_level: str = "INFO",
    cors_allow_origins: Sequence[str] | None = None,
) -> None:
    """Run the MCP server.

    Args:
        base_url: Base URL for the EvalHub API. If not provided, uses
                  EVALHUB_BASE_URL environment variable or default.
        host: Host to bind the server to (default: 0.0.0.0)
        port: Port to listen on (default: 3001)
        json_response: Enable JSON responses instead of SSE streams (default: True)
        log_level: Logging level (default: INFO)
        cors_allow_origins: List of allowed CORS origins. If None, CORS middleware
                           is not applied. Use ["*"] to allow all origins.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create -sdk[client] instance
    if base_url is None:
        base_url = os.getenv("EVALHUB_BASE_URL", "http://localhost:8080")

    client = AsyncEvalHubClient(base_url=base_url)
    set_client(client)

    # Create the session manager with stateless mode
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=json_response,
        stateless=True,
    )

    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        method = scope.get("method", "")
        path = scope.get("path", "")
        logger.debug(f"Handling {method} request to {path}")
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(starlette_app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager."""
        async with session_manager.run():
            logger.info("EvalHub MCP server started!")
            try:
                yield
            finally:
                logger.info("EvalHub MCP server shutting down...")

    base_app = Starlette(
        debug=True,
        routes=[
            Mount("/", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    # Conditionally apply CORS middleware
    final_app: Starlette | CORSMiddleware
    if cors_allow_origins is not None:
        final_app = CORSMiddleware(
            base_app,
            allow_origins=cors_allow_origins,
            allow_methods=[
                "GET",
                "POST",
                "DELETE",
                "OPTIONS",
            ],  # MCP streamable HTTP methods
            allow_headers=["*"],  # Allow all headers
            expose_headers=["Mcp-Session-Id"],
        )
    else:
        final_app = base_app

    import uvicorn

    uvicorn.run(final_app, host=host, port=port)


if __name__ == "__main__":
    run_server(cors_allow_origins=["*"])  # local mode = all CORS Origins
