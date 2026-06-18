"""MCP command group — Python MCP server and Go binary lifecycle management."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import click

from . import config as cfg

MCP_STATE_DIR = cfg.DEFAULT_CONFIG_DIR / "mcp"
PID_FILE = MCP_STATE_DIR / "pid"
LOG_FILE = MCP_STATE_DIR / "log"

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


def _build_mcp_env() -> dict[str, str]:
    """Build environment for the Go binary. Inherits the current environment as-is."""
    return dict(os.environ)


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _graceful_signal() -> signal.Signals:
    if sys.platform == "win32":
        return signal.CTRL_BREAK_EVENT  # type: ignore[attr-defined,return-value]
    return signal.SIGTERM


def _force_signal() -> signal.Signals:
    if sys.platform == "win32":
        return signal.SIGTERM
    return signal.SIGKILL


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid


def _clean_stale_pid() -> None:
    pid = _read_pid()
    if pid is not None and not _is_process_alive(pid):
        PID_FILE.unlink(missing_ok=True)


@click.group(invoke_without_command=True)
@click.option(
    "--tenant",
    default=None,
    envvar="EVALHUB_TENANT",
    help="[DEPRECATED] Kubernetes namespace / tenant identifier (overrides profile config).",
)
@click.pass_context
def mcp(ctx: click.Context, tenant: str | None) -> None:
    """Start the EvalHub MCP server, or manage the Go MCP binary."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        import mcp as _mcp  # noqa: F401
    except ModuleNotFoundError:
        raise click.ClickException(
            "MCP server requires the 'mcp' extra.\n"
            "Install it with: pip install 'eval-hub-sdk[mcp]'"
        ) from None

    data = cfg.load_config()
    prof = cfg.get_profile(data, ctx.obj.get("profile"))

    resolved_url = ctx.obj.get("base_url") or prof.get(
        "base_url", "http://localhost:8080"
    )
    resolved_token = ctx.obj.get("token") or prof.get("token")
    resolved_tenant = tenant or prof.get("tenant")
    resolved_insecure = str(prof.get("insecure", "false")).lower() in (
        "true",
        "1",
        "yes",
    )
    resolved_timeout = float(prof.get("timeout", 30.0))

    import asyncio

    from ..client.evalhub import AsyncEvalHubClient
    from ..mcp.server import mcp as mcp_server
    from ..mcp.server import set_client

    client = AsyncEvalHubClient(
        base_url=resolved_url,
        auth_token=resolved_token,
        tenant=resolved_tenant,
        insecure=resolved_insecure,
        timeout=resolved_timeout,
    )
    set_client(client)
    asyncio.run(mcp_server.run_stdio_async())


@mcp.command("run")
@click.argument("go_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def mcp_run(ctx: click.Context, go_args: tuple[str, ...]) -> None:
    """Run the Go MCP binary in the foreground.

    Any arguments after -- are passed directly to the evalhub-mcp binary.
    """
    binary = _find_mcp_binary()
    env = _build_mcp_env()
    cmd = [binary, *go_args]

    MCP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("w") as log_fh:
        result = subprocess.run(
            cmd,
            env=env,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=log_fh,
        )
    ctx.exit(result.returncode)


@mcp.command("start")
@click.argument("go_args", nargs=-1, type=click.UNPROCESSED)
def mcp_start(go_args: tuple[str, ...]) -> None:
    """Start the Go MCP binary as a background HTTP daemon.

    Any arguments after -- are passed directly to the evalhub-mcp binary.
    Automatically injects --transport http.
    """
    _clean_stale_pid()
    pid = _read_pid()
    if pid is not None:
        raise click.ClickException(
            f"MCP server is already running (PID {pid}). "
            "Stop it first with: evalhub mcp stop"
        )

    binary = _find_mcp_binary()
    env = _build_mcp_env()
    cmd = [binary, "--transport", "http", *go_args]

    MCP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = LOG_FILE.open("w")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        cmd,
        env=env,
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

    os.kill(pid, _graceful_signal())

    deadline = time.monotonic() + _STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            PID_FILE.unlink(missing_ok=True)
            click.echo("MCP server stopped.")
            return
        time.sleep(0.2)

    os.kill(pid, _force_signal())
    PID_FILE.unlink(missing_ok=True)
    click.echo("MCP server force-killed.")


@mcp.command("status")
def mcp_status() -> None:
    """Check if the background MCP server is running."""
    _clean_stale_pid()
    pid = _read_pid()
    if pid is not None and _is_process_alive(pid):
        click.echo(f"MCP server is running (PID {pid}).")
    else:
        click.echo("MCP server is not running.")
