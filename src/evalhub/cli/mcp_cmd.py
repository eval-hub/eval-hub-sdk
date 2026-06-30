"""MCP command group — Go binary lifecycle management."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

import click

from evalhub import __version__

from . import config as cfg
from ._process import (
    find_binary,
    live_pid,
    require_not_running,
    run_foreground,
    spawn_background,
    stop_daemon,
)

MCP_STATE_DIR = cfg.DEFAULT_CONFIG_DIR / "mcp"
PID_FILE = MCP_STATE_DIR / "pid"
LOG_FILE = MCP_STATE_DIR / "mcp.log"
CONFIG_FILE = MCP_STATE_DIR / "config.yaml"

_STARTUP_WAIT = 2.0
_STOP_TIMEOUT = 5.0


def _generate_config(
    ctx: click.Context,
    *,
    default_transport: str = "http",
) -> tuple[list[str], dict[str, object]]:
    """Build MCP config from the active CLI profile.

    Returns (extra_cli_args, mcp_config_dict).
    """
    data = cfg.load_config()
    profile = cfg.get_profile(data, ctx.obj.get("profile"))
    mcp_config = cfg.build_mcp_config(profile, default_transport=default_transport)
    cfg.save_config(mcp_config, CONFIG_FILE)
    return ["--config", str(CONFIG_FILE)], mcp_config


_JSONRPC_VERSION = "2.0"


def _mcp_post(
    url: str,
    method: str,
    *,
    params: dict[str, Any] | None = None,
    msg_id: int | None = 1,
    session_id: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """POST a JSON-RPC request/notification and return (parsed result, session-id)."""
    body: dict[str, Any] = {"jsonrpc": _JSONRPC_VERSION, "method": method}
    if msg_id is not None:
        body["id"] = msg_id
    if params:
        body["params"] = params
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            sid = resp.headers.get("Mcp-Session-Id") or session_id
            raw = resp.read().decode()
    except (OSError, ValueError):
        return None, session_id
    # Streamable HTTP may return SSE (event: …\ndata: …) or plain JSON.
    data_line = raw
    for line in raw.splitlines():
        if line.startswith("data: "):
            data_line = line[len("data: ") :]
            break
    try:
        return json.loads(data_line).get("result"), sid
    except (json.JSONDecodeError, ValueError):
        return None, sid


def _read_version_resource(url: str, session_id: str | None) -> dict[str, Any]:
    """Read the evalhub://server/version resource and return parsed fields."""
    ver, _ = _mcp_post(
        url,
        "resources/read",
        params={"uri": "evalhub://server/version"},
        msg_id=2,
        session_id=session_id,
    )
    contents = (ver or {}).get("contents", [])
    if not contents:
        return {}
    try:
        return json.loads(contents[0].get("text", "{}"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, ValueError):
        return {}


def _fetch_server_info(
    host: str = "localhost", port: int = 3001
) -> dict[str, Any] | None:
    """Perform an MCP handshake and return serverInfo + version resource data."""
    url = f"http://{host}:{port}/mcp"

    result, sid = _mcp_post(
        url,
        "initialize",
        params={
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "evalhub-cli", "version": __version__},
        },
    )
    if result is None:
        return None
    info: dict[str, Any] = result.get("serverInfo", {})

    _mcp_post(url, "notifications/initialized", msg_id=None, session_id=sid)

    version_data = _read_version_resource(url, sid)
    if "git_hash" in version_data:
        info["git_hash"] = version_data["git_hash"]

    return info


@click.group()
def mcp() -> None:
    """Manage the local evalhub-mcp Go binary."""


@mcp.command("run")
@click.pass_context
def mcp_run(ctx: click.Context) -> None:
    """Run the evalhub-mcp binary in the foreground.

    Uses the mcp_transport value from the active profile if set,
    otherwise defaults to stdio. The active CLI profile is used to
    generate ~/.config/evalhub/mcp/config.yaml automatically.
    """
    binary = find_binary("evalhub-mcp", "EVALHUB_MCP_BIN")
    extra, _ = _generate_config(ctx, default_transport="stdio")
    run_foreground([binary, *extra], ctx)


@mcp.command("start")
@click.pass_context
def mcp_start(ctx: click.Context) -> None:
    """Start the Go MCP binary as a background daemon.

    Uses the mcp_transport value from the active profile if set,
    otherwise defaults to http. The active CLI profile is used to
    generate ~/.config/evalhub/mcp/config.yaml automatically.
    """
    require_not_running(PID_FILE, "MCP server", "evalhub mcp stop")

    binary = find_binary("evalhub-mcp", "EVALHUB_MCP_BIN")
    extra, mcp_config = _generate_config(ctx)
    if mcp_config.get("transport") == "stdio":
        raise click.ClickException(
            "Cannot start in background with stdio transport.\n"
            "Use 'evalhub mcp run' for stdio, or set a network transport:\n"
            "  evalhub config set mcp_transport http"
        )
    cmd = [binary, *extra]

    proc = spawn_background(cmd, MCP_STATE_DIR, LOG_FILE)
    time.sleep(_STARTUP_WAIT)

    if proc.poll() is not None:
        output = LOG_FILE.read_text().strip()
        msg = f"MCP server crashed on startup (exit code {proc.returncode})."
        if output:
            msg += f"\nLog output:\n{output}"
        raise click.ClickException(msg)

    PID_FILE.write_text(str(proc.pid))
    click.echo(f"MCP server started (PID {proc.pid}).")
    click.echo(f"  Transport: {mcp_config['transport']}")
    click.echo(f"  URL:       http://{mcp_config['host']}:{mcp_config['port']}")
    click.echo(f"  Logs:      {LOG_FILE}")


@mcp.command("stop")
def mcp_stop() -> None:
    """Stop the background MCP server."""
    stop_daemon(PID_FILE, _STOP_TIMEOUT, "MCP server")


@mcp.command("status")
def mcp_status() -> None:
    """Check if the background MCP server is running."""
    pid = live_pid(PID_FILE)
    if pid is None:
        click.echo("MCP server is not running.")
        return

    click.echo(f"MCP server is running (PID {pid}).")

    mcp_cfg = cfg.load_config(CONFIG_FILE)
    host = str(mcp_cfg.get("host", "localhost"))
    port = int(mcp_cfg.get("port", 3001))
    info = _fetch_server_info(host, port)
    if info:
        name = info.get("name", "unknown")
        version = info.get("version", "unknown")
        git_hash = info.get("git_hash", "")
        click.echo(f"  Name:    {name}")
        click.echo(f"  Version: {version}")
        if git_hash:
            click.echo(f"  Commit:  {git_hash}")
    click.echo(f"  URL:     http://{host}:{port}")
    click.echo(f"  Logs:    {LOG_FILE}")
