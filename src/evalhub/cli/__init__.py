"""EvalHub CLI - command-line interface for EvalHub."""

__all__ = ["main"]


def main() -> None:
    """Entry point that delegates to bootstrap."""
    from .bootstrap import main as _main

    _main()
