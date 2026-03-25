"""CLI entry point for the EvalHub MCP Server.

Usage:
    python -m evalhub.mcp --tenant <namespace>
    evalhub-mcp --tenant <namespace>
"""

from __future__ import annotations

import subprocess

import click


def _resolve_auth_token(auth_token: str | None) -> str:
    """Resolve auth token from CLI arg or `oc whoami -t`."""
    if auth_token:
        return auth_token

    try:
        result = subprocess.run(
            ["oc", "whoami", "-t"],
            capture_output=True,
            text=True,
            check=True,
        )
        token = result.stdout.strip()
        if not token:
            raise click.ClickException("`oc whoami -t` returned an empty token.")
        return token
    except FileNotFoundError:
        raise click.ClickException(
            "`oc` command not found. Install the OpenShift CLI or provide --auth-token."
        )
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"`oc whoami -t` failed: {e.stderr.strip()}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--tenant",
    required=True,
    envvar="EVALHUB_TENANT",
    help="Kubernetes namespace / tenant identifier.",
)
@click.option(
    "--base-url",
    default="http://localhost:8080",
    envvar="EVALHUB_BASE_URL",
    show_default=True,
    help="EvalHub service base URL.",
)
@click.option(
    "--auth-token",
    default=None,
    envvar="EVALHUB_TOKEN",
    help="Auth token (default: obtained via `oc whoami -t`).",
)
@click.option(
    "--insecure",
    is_flag=True,
    default=False,
    help="Skip TLS verification.",
)
def main(
    tenant: str,
    base_url: str,
    auth_token: str | None,
    insecure: bool,
) -> None:
    """EvalHub MCP Server - Model Context Protocol server for EvalHub."""
    token = _resolve_auth_token(auth_token)

    from ..client.evalhub import AsyncEvalHubClient
    from .server import mcp as mcp_server
    from .server import set_client

    client = AsyncEvalHubClient(
        base_url=base_url,
        auth_token=token,
        tenant=tenant,
        insecure=insecure,
    )
    set_client(client)

    import asyncio

    asyncio.run(mcp_server.run_stdio_async())


if __name__ == "__main__":
    main()
