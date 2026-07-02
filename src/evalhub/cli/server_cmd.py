"""Server command group — eval-hub-server lifecycle management."""

from __future__ import annotations

import ssl
import time
import urllib.request
from pathlib import Path

import click
import yaml

from . import config as cfg
from ._process import (
    find_binary,
    live_pid,
    require_not_running,
    run_foreground,
    spawn_background,
    stop_daemon,
)

SERVER_STATE_DIR = cfg.DEFAULT_CONFIG_DIR / "server"
PID_FILE = SERVER_STATE_DIR / "pid"
LOG_FILE = SERVER_STATE_DIR / "server.log"

_STARTUP_TIMEOUT = 30.0
_STARTUP_POLL = 0.5
_STOP_TIMEOUT = 5.0
_DEFAULT_PORT = 8080


def _read_server_config(config_dir: Path) -> tuple[int, bool]:
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        return _DEFAULT_PORT, False
    try:
        data = yaml.safe_load(config_path.read_text())
        svc = data.get("service", {})
        port = int(svc.get("port", _DEFAULT_PORT))
        cert = svc.get("tls_cert_file", "")
        key = svc.get("tls_key_file", "")
        tls = bool(cert and key)
        return port, tls
    except (yaml.YAMLError, TypeError, ValueError, AttributeError) as exc:
        raise click.ClickException(
            f"Failed to parse server config {config_path}: {exc}"
        ) from exc


def _health_check(port: int, *, tls: bool = False) -> bool:
    scheme = "https" if tls else "http"
    url = f"{scheme}://localhost:{port}/api/v1/health"
    req = urllib.request.Request(url, method="GET")
    try:
        ctx: ssl.SSLContext | None = None
        if tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=2, context=ctx) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def _wait_for_healthy(port: int, timeout: float, *, tls: bool = False) -> bool:
    deadline = time.monotonic() + timeout
    delay = _STARTUP_POLL
    while time.monotonic() < deadline:
        if _health_check(port, tls=tls):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 2.0)
    return _health_check(port, tls=tls)


def _resolve_config_dir(ctx: click.Context) -> Path:
    data = cfg.load_config()
    return cfg.resolve_component_config_dir(
        data,
        SERVER_STATE_DIR,
        profile=ctx.obj.get("profile"),
    )


def _require_config(config_dir: Path) -> None:
    if not (config_dir / "config.yaml").exists():
        raise click.ClickException(
            f"No server config found at {config_dir / 'config.yaml'}.\n"
            "Set one first with: evalhub config set server_config_file <path>"
        )


@click.group()
def server() -> None:
    """Manage the local eval-hub-server binary."""


@server.command("run")
@click.pass_context
def server_run(ctx: click.Context) -> None:
    """Run eval-hub-server in the foreground.

    \b
    Examples:
      evalhub server run
      evalhub --profile staging server run
    """
    binary = find_binary("eval-hub-server", "EVALHUB_SERVER_BIN")
    cfg_dir = _resolve_config_dir(ctx)
    _require_config(cfg_dir)

    run_foreground([binary, "-local", "-configdir", str(cfg_dir)], ctx)


@server.command("start")
@click.pass_context
def server_start(ctx: click.Context) -> None:
    """Start eval-hub-server as a background daemon.

    \b
    Examples:
      evalhub server start
      evalhub --profile staging server start
    """
    require_not_running(PID_FILE, "Server", "evalhub server stop")

    binary = find_binary("eval-hub-server", "EVALHUB_SERVER_BIN")
    cfg_dir = _resolve_config_dir(ctx)
    _require_config(cfg_dir)

    port, tls = _read_server_config(cfg_dir)
    scheme = "https" if tls else "http"
    cmd = [binary, "-local", "-configdir", str(cfg_dir)]

    proc = spawn_background(cmd, SERVER_STATE_DIR, LOG_FILE)

    if not _wait_for_healthy(port, _STARTUP_TIMEOUT, tls=tls):
        if proc.poll() is not None:
            output = LOG_FILE.read_text().strip()
            msg = f"Server crashed on startup (exit code {proc.returncode})."
            if output:
                msg += f"\nLog output:\n{output}"
            raise click.ClickException(msg)
        proc.terminate()
        try:
            proc.wait(timeout=_STOP_TIMEOUT)
        except Exception:
            proc.kill()
            proc.wait(timeout=2)
        raise click.ClickException(
            f"Server did not become healthy within {_STARTUP_TIMEOUT}s.\n"
            f"Health check: {scheme}://localhost:{port}/api/v1/health\n"
            f"Check logs at: {LOG_FILE}"
        )

    PID_FILE.write_text(str(proc.pid))
    click.echo(f"Server started (PID {proc.pid}).")
    click.echo(f"  URL:  {scheme}://localhost:{port}")
    click.echo(f"  Logs: {LOG_FILE}")


@server.command("stop")
def server_stop() -> None:
    """Stop the background eval-hub-server.

    \b
    Examples:
      evalhub server stop
    """
    stop_daemon(PID_FILE, _STOP_TIMEOUT, "Server")


@server.command("status")
@click.pass_context
def server_status(ctx: click.Context) -> None:
    """Check if eval-hub-server is running.

    Works for both background (server start) and foreground (server run)
    by probing the health endpoint directly.

    \b
    Examples:
      evalhub server status
    """
    cfg_dir = _resolve_config_dir(ctx)
    port, tls = _read_server_config(cfg_dir)
    scheme = "https" if tls else "http"

    pid = live_pid(PID_FILE)
    healthy = _health_check(port, tls=tls)

    if not healthy and pid is None:
        click.echo("Server is not running.")
        return

    if pid is not None:
        click.echo(f"Server is running (PID {pid}).")
    else:
        click.echo("Server is running.")

    click.echo(f"  Health: {'healthy' if healthy else 'not responding'}")
    click.echo(f"  URL:    {scheme}://localhost:{port}")
    if pid is not None:
        click.echo(f"  Logs:   {LOG_FILE}")
