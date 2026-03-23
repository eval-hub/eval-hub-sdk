"""EvalHub CLI - command-line interface for EvalHub."""

from .client import create_client, get_client, handle_api_errors
from .formatter import format_option, output

__all__ = [
    "create_client",
    "format_option",
    "get_client",
    "handle_api_errors",
    "main",
    "output",
]


def main() -> None:
    """Entry point that delegates to bootstrap."""
    from .bootstrap import main as _bootstrap_main

    _bootstrap_main()
