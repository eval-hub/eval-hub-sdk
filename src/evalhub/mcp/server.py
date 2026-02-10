"""MCP server for EvalHub SDK.

This module provides an MCP (Model Context Protocol) server that exposes
EvalHub provider and benchmark operations as MCP resources.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from ..client import AsyncEvalHubClient

logger = logging.getLogger(__name__)

# Create MCP server instance
app = Server("EvalHub SDK client based MCP Server")

# Global client instance (initialized in main or by user)
_client: AsyncEvalHubClient | None = None


def set_client(client: AsyncEvalHubClient) -> None:
    """Set the AsyncEvalHubClient instance to use for MCP resources.

    Args:
        client: The AsyncEvalHubClient instance
    """
    global _client
    _client = client


@app.list_resources()
async def list_resources() -> list[types.Resource]:
    """List all available MCP resources.

    Returns:
        List of resource definitions
    """
    return [
        types.Resource(
            uri="evalhub://providers",
            name="List Providers",
            description="List all registered EvalHub providers",
            mimeType="application/json",
        ),
        types.Resource(
            uri="evalhub://benchmarks",
            name="List Benchmarks",
            description="List all available benchmarks",
            mimeType="application/json",
        ),
    ]


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
        provider = await _client.providers.get(provider_id)
        return provider.model_dump_json()
    else:
        raise ValueError(f"Unknown resource URI: {uri_str}")


def run_server(
    base_url: str | None = None,
    host: str = "0.0.0.0",
    port: int = 3001,
    json_response: bool = True,
    log_level: str = "INFO",
) -> None:
    """Run the MCP server.

    Args:
        base_url: Base URL for the EvalHub API. If not provided, uses
                  EVALHUB_BASE_URL environment variable or default.
        host: Host to bind the server to (default: 0.0.0.0)
        port: Port to listen on (default: 3001)
        json_response: Enable JSON responses instead of SSE streams (default: True)
        log_level: Logging level (default: INFO)
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create client instance
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

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
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

    # Create an ASGI application using the transport
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    # Wrap ASGI application with CORS middleware to expose Mcp-Session-Id header
    # for browser-based clients (ensures 500 errors get proper CORS headers)
    starlette_app = CORSMiddleware(
        starlette_app,
        allow_origins=["*"],  # Allow all origins - adjust as needed for production
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],  # MCP streamable HTTP methods
        allow_headers=["*"],  # Allow all headers
        expose_headers=["Mcp-Session-Id"],
    )

    import uvicorn

    uvicorn.run(starlette_app, host=host, port=port)


if __name__ == "__main__":
    run_server()
