"""EvalHub CLI - command-line interface for EvalHub."""

from .client import create_client, get_client, handle_api_errors
from .main import main

__all__ = ["create_client", "get_client", "handle_api_errors", "main"]


def main() -> None:
    """Entry point that delegates to bootstrap."""
    from .bootstrap import main as _main

    _main()
