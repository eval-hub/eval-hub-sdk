"""MCP command group — Go binary lifecycle management."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import click
import yaml

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
GENERATED_CONFIG = "mcp-config.yaml"

_STARTUP_WAIT = 2.0
_STOP_TIMEOUT = 5.0

_DEFAULT_PORT = 3001


def _resolve_mcp_config(ctx: click.Context) -> tuple[dict[str, Any], Path]:
    """Return (profile_dict, config_dir) for the active MCP profile."""
    data = cfg.load_config()
    profile_name = ctx.obj.get("profile")
    cfg_dir = cfg.resolve_component_config_dir(
        data, MCP_STATE_DIR, profile=profile_name
    )
    return cfg.get_profile(data, profile_name), cfg_dir


def _generate_merged_config(
    profile: dict[str, Any],
    config_dir: Path,
    defaults: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Merge root profile connection keys with MCP config.

    If ``config_dir / "config.yaml"`` does not exist it is treated as
    empty — the generated config will contain only values sourced from
    the root CLI profile (if any).  MCP config keys take precedence.
    *defaults* are applied first and overridden by both profile and
    MCP config values.  The merged result is written to
    ``config_dir / GENERATED_CONFIG`` and returned along with the path.
    """
    merged: dict[str, Any] = dict(defaults) if defaults else {}
    for key in ("base_url", "token", "tenant", "insecure"):
        val = profile.get(key)
        if val is not None:
            merged[key] = val

    mcp_path = config_dir / "config.yaml"
    if mcp_path.exists():
        try:
            mcp_data = yaml.safe_load(mcp_path.read_text()) or {}
        except (yaml.YAMLError, TypeError) as exc:
            raise click.ClickException(
                f"Failed to parse MCP config {mcp_path}: {exc}"
            ) from exc
        merged.update(mcp_data)

    if "insecure" in merged:
        merged["insecure"] = cfg.parse_bool(merged["insecure"])

    if "port" in merged:
        try:
            merged["port"] = int(merged["port"])
        except (TypeError, ValueError):
            raise click.ClickException(
                f"Invalid port value: {merged['port']!r} (must be an integer)"
            )

    dest = config_dir / GENERATED_CONFIG
    cfg.save_config(merged, dest)
    return merged, dest


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

    \b
    Reads MCP-specific settings from the MCP config file and merges
    connection keys (base_url, token, tenant, insecure) from the
    active CLI profile. MCP config values take precedence.

    \b
    Examples:
      evalhub mcp run
      evalhub --profile staging mcp run
    """
    binary = find_binary("evalhub-mcp", "EVALHUB_MCP_BIN")
    profile, cfg_dir = _resolve_mcp_config(ctx)
    _, config_path = _generate_merged_config(
        profile, cfg_dir, defaults={"transport": "stdio"}
    )
    run_foreground([binary, "--config", str(config_path)], ctx)


@mcp.command("start")
@click.pass_context
def mcp_start(ctx: click.Context) -> None:
    """Start the Go MCP binary as a background daemon.

    \b
    Reads MCP-specific settings from the MCP config file and merges
    connection keys (base_url, token, tenant, insecure) from the
    active CLI profile. MCP config values take precedence.

    \b
    Examples:
      evalhub mcp start
      evalhub --profile staging mcp start
    """
    require_not_running(PID_FILE, "MCP server", "evalhub mcp stop")

    binary = find_binary("evalhub-mcp", "EVALHUB_MCP_BIN")
    profile, cfg_dir = _resolve_mcp_config(ctx)
    merged, config_path = _generate_merged_config(
        profile, cfg_dir, defaults={"transport": "http"}
    )
    transport = merged["transport"]
    host = merged.get("host", "localhost")
    port = merged.get("port", _DEFAULT_PORT)

    if transport == "stdio":
        raise click.ClickException(
            "'evalhub mcp start' can be used only in non-stdio transport mode.\n"
            "Use 'evalhub mcp run' for stdio, or update your MCP config\n"
            "to use a different transport:\n"
            "  evalhub config set mcp_config_file <myconfig.yaml>\n"
            "where myconfig.yaml contains:\n"
            "  transport: http"
        )
    cmd = [binary, "--config", str(config_path)]

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
    click.echo(f"  Transport: {transport}")
    click.echo(f"  URL:       http://{host}:{port}")
    click.echo(f"  Logs:      {LOG_FILE}")


@mcp.command("stop")
def mcp_stop() -> None:
    """Stop the background MCP server."""
    stop_daemon(PID_FILE, _STOP_TIMEOUT, "MCP server")


@mcp.command("status")
@click.pass_context
def mcp_status(ctx: click.Context) -> None:
    """Check if the background MCP server is running."""
    pid = live_pid(PID_FILE)
    if pid is None:
        click.echo("MCP server is not running.")
        return

    click.echo(f"MCP server is running (PID {pid}).")

    _, cfg_dir = _resolve_mcp_config(ctx)
    config_path = cfg_dir / GENERATED_CONFIG
    try:
        merged = yaml.safe_load(config_path.read_text()) or {}
    except (FileNotFoundError, yaml.YAMLError):
        merged = {}
    host = merged.get("host", "localhost")
    port = merged.get("port", _DEFAULT_PORT)

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
