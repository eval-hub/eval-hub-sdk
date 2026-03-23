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


@main.command()
@click.option("--base-url", default=None, help="Base URL for the EvalHub API (env: EVALHUB_BASE_URL).")
@click.option("--tenant", default=None, help="Tenant identifier (env: EVALHUB_TENANT).")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to.")
@click.option("--port", default=3001, type=int, help="Port to listen on.")
def mcp(base_url: str | None, tenant: str | None, host: str, port: int) -> None:
    """Start the MCP server."""
    from evalhub.mcp.server import run_server

    run_server(base_url=base_url, tenant=tenant, host=host, port=port, cors_allow_origins=["*"])
