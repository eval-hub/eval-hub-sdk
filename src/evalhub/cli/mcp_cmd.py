"""MCP command group — Go binary lifecycle management."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import click

from evalhub import __version__

from . import config as cfg

MCP_STATE_DIR = cfg.DEFAULT_CONFIG_DIR / "mcp"
PID_FILE = MCP_STATE_DIR / "pid"
LOG_FILE = MCP_STATE_DIR / "mcp.log"
CONFIG_FILE = MCP_STATE_DIR / "config.yaml"

_STARTUP_WAIT = 2.0
_STOP_TIMEOUT = 5.0


def _find_mcp_binary() -> str:
    env = os.environ.get("EVALHUB_MCP_BIN")
    if env:
        return env
    found = shutil.which("evalhub-mcp")
    if found:
        return found
    raise click.ClickException(
        "Could not find the 'evalhub-mcp' binary.\n"
        "Install it and ensure it is on your PATH, or set EVALHUB_MCP_BIN."
    )


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


_GRACEFUL_SIGNAL: signal.Signals = (
    signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM  # type: ignore[attr-defined]
)
_FORCE_SIGNAL: signal.Signals = (
    signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
)


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid


def _generate_config(
    ctx: click.Context,
) -> tuple[list[str], dict[str, object]]:
    """Build MCP config from the active CLI profile.

    Returns (extra_cli_args, mcp_config_dict).
    """
    data = cfg.load_config()
    profile = cfg.get_profile(data, ctx.obj.get("profile"))
    mcp_config = cfg.build_mcp_config(profile)
    cfg.save_config(mcp_config, CONFIG_FILE)
    return ["--config", str(CONFIG_FILE)], mcp_config


def _load_mcp_config() -> dict[str, Any]:
    """Load the generated MCP config.yaml (host/port/transport)."""
    return cfg.load_config(CONFIG_FILE)


def _fetch_server_info(
    host: str = "localhost", port: int = 3001
) -> dict[str, Any] | None:
    """Send an MCP initialize request and return the serverInfo, or None on failure."""
    url = f"http://{host}:{port}/mcp"
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "evalhub-cli", "version": __version__},
            },
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            raw = resp.read().decode()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    # Streamable HTTP may return SSE (event: …\ndata: …) or plain JSON.
    data_line = raw
    for line in raw.splitlines():
        if line.startswith("data: "):
            data_line = line[len("data: ") :]
            break
    try:
        body = json.loads(data_line)
    except (json.JSONDecodeError, ValueError):
        return None
    return body.get("result", {}).get("serverInfo")  # type: ignore[no-any-return]


@click.group()
def mcp() -> None:
    """Manage the evalhub-mcp Go binary (run, start, stop, status)."""


@mcp.command("run")
@click.pass_context
def mcp_run(ctx: click.Context) -> None:
    """Run the evalhub-mcp binary in the foreground.

    The active CLI profile is used to generate
    ~/.config/evalhub/mcp/config.yaml automatically.
    """
    binary = _find_mcp_binary()
    extra, _ = _generate_config(ctx)
    cmd = [binary, *extra]

    result = subprocess.run(
        cmd,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    ctx.exit(result.returncode)


@mcp.command("start")
@click.pass_context
def mcp_start(ctx: click.Context) -> None:
    """Start the Go MCP binary as a background daemon.

    Transport is read from the active profile (default: http).
    The active CLI profile is used to generate
    ~/.config/evalhub/mcp/config.yaml automatically.
    """
    pid = _read_pid()
    if pid is not None and not _is_process_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        pid = None
    if pid is not None:
        raise click.ClickException(
            f"MCP server is already running (PID {pid}). "
            "Stop it first with: evalhub mcp stop"
        )

    binary = _find_mcp_binary()
    extra, mcp_config = _generate_config(ctx)
    if mcp_config.get("transport") == "stdio":
        raise click.ClickException(
            "Cannot start in background with stdio transport.\n"
            "Use 'evalhub mcp run' for stdio, or set a network transport:\n"
            "  evalhub config set mcp_transport http"
        )
    cmd = [binary, *extra]

    MCP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = LOG_FILE.open("w")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )

    time.sleep(_STARTUP_WAIT)

    log_fh.close()

    if proc.poll() is not None:
        output = LOG_FILE.read_text().strip()
        msg = f"MCP server crashed on startup (exit code {proc.returncode})."
        if output:
            msg += f"\nLog output:\n{output}"
        raise click.ClickException(msg)

    PID_FILE.write_text(str(proc.pid))
    click.echo(f"MCP server started (PID {proc.pid}).")
    click.echo(f"Logs: {LOG_FILE}")


@mcp.command("stop")
def mcp_stop() -> None:
    """Stop the background MCP server."""
    pid = _read_pid()
    if pid is None or not _is_process_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        click.echo("MCP server is not running.")
        return

    os.kill(pid, _GRACEFUL_SIGNAL)

    deadline = time.monotonic() + _STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            PID_FILE.unlink(missing_ok=True)
            click.echo("MCP server stopped.")
            return
        time.sleep(0.2)

    os.kill(pid, _FORCE_SIGNAL)
    PID_FILE.unlink(missing_ok=True)
    click.echo("MCP server force-killed.")


@mcp.command("status")
def mcp_status() -> None:
    """Check if the background MCP server is running."""
    pid = _read_pid()
    if pid is not None and not _is_process_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        pid = None
    if pid is None:
        click.echo("MCP server is not running.")
        return

    click.echo(f"MCP server is running (PID {pid}).")

    mcp_cfg = _load_mcp_config()
    host = mcp_cfg.get("host", "localhost")
    port = int(mcp_cfg.get("port", 3001))
    info = _fetch_server_info(host, port)
    if info:
        name = info.get("name", "unknown")
        version = info.get("version", "unknown")
        click.echo(f"  Name:    {name}")
        click.echo(f"  Version: {version}")
    click.echo(f"  URL:     http://{host}:{port}")
    click.echo(f"  Logs:    {LOG_FILE}")
