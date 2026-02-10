from collections.abc import AsyncGenerator

import pytest
from inline_snapshot import snapshot
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent, ListResourcesResult

from evalhub.mcp.server import app


@pytest.fixture
def anyio_backend():  
    return "asyncio"


@pytest.fixture
async def client_session() -> AsyncGenerator[ClientSession]:
    async with create_connected_server_and_client_session(app, raise_exceptions=True) as _session:
        yield _session


@pytest.mark.anyio
async def test_call_add_tool(client_session: ClientSession):
    result = await client_session.list_resources()
    assert result == snapshot(
        ListResourcesResult(resources =[])
    )