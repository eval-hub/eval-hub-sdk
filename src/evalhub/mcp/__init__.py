"""MCP (Model Context Protocol) server for EvalHub SDK.

This module provides MCP resources for EvalHub providers and benchmarks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .server import mcp, run_server, set_client

__all__ = ["mcp", "run_server", "set_client"]


def __getattr__(name: str) -> Any:
    """Lazy import to avoid import conflicts when running as __main__."""
    if name in __all__:
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
