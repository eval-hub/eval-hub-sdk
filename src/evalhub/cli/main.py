"""EvalHub CLI entry point and command groups."""

import click

import evalhub


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=evalhub.__version__, prog_name="evalhub")
def main() -> None:
    """EvalHub CLI - manage evaluations, providers, collections, and configuration."""


@main.command()
def version() -> None:
    """Print version and build info."""
    click.echo(f"evalhub {evalhub.__version__}")


@main.group()
def eval() -> None:
    """Submit and manage evaluation jobs."""


@main.group()
def collections() -> None:
    """Browse and manage benchmark collections."""


@main.group()
def providers() -> None:
    """List and inspect evaluation providers."""


@main.group()
def config() -> None:
    """View and update CLI configuration."""
