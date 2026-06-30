"""Server command group — eval-hub-server lifecycle management."""

from __future__ import annotations

import os
import shutil
import signal
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import click
import yaml

from . import config as cfg

SERVER_STATE_DIR = cfg.DEFAULT_CONFIG_DIR / "server"
PID_FILE = SERVER_STATE_DIR / "pid"
LOG_FILE = SERVER_STATE_DIR / "server.log"

_STARTUP_TIMEOUT = 30.0
_STARTUP_POLL = 0.5
_STOP_TIMEOUT = 5.0
_DEFAULT_PORT = 8080

_GRACEFUL_SIGNAL: signal.Signals = (
    signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM  # type: ignore[attr-defined]
)
_FORCE_SIGNAL: signal.Signals = (
    signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
)


def _find_server_binary() -> str:
    env = os.environ.get("EVALHUB_SERVER_BIN")
    if env:
        return env
    found = shutil.which("eval-hub-server")
    if found:
        return found
    raise click.ClickException(
        "Could not find the 'eval-hub-server' binary.\n"
        "Install it and ensure it is on your PATH, or set EVALHUB_SERVER_BIN."
    )


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid


def _live_pid() -> int | None:
    pid = _read_pid()
    if pid is not None and not _is_process_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        return None
    return pid


def _read_server_port(config_dir: Path) -> int:
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        return _DEFAULT_PORT
    try:
        data = yaml.safe_load(config_path.read_text())
        return int(data.get("service", {}).get("port", _DEFAULT_PORT))
    except (yaml.YAMLError, TypeError, ValueError, AttributeError):
        return _DEFAULT_PORT


def _is_tls_enabled(config_dir: Path) -> bool:
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        return False
    try:
        data = yaml.safe_load(config_path.read_text())
        svc = data.get("service", {})
        cert = str(svc.get("tls_cert_file", ""))
        key = str(svc.get("tls_key_file", ""))
        return cert != "" and key != ""
    except (yaml.YAMLError, TypeError, AttributeError):
        return False


def _server_scheme(tls: bool) -> str:
    return "https" if tls else "http"


def _health_check(port: int, *, tls: bool = False) -> bool:
    scheme = _server_scheme(tls)
    url = f"{scheme}://localhost:{port}/api/v1/health"
    req = urllib.request.Request(url, method="GET")
    try:
        ctx: ssl.SSLContext | None = None
        if tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=2, context=ctx) as resp:
            return int(resp.status) == 200
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


def _server_config_dir(profile_name: str) -> Path:
    return SERVER_STATE_DIR / profile_name


def _resolve_config_dir(ctx: click.Context) -> Path:
    data = cfg.load_config()
    profile = ctx.obj.get("profile")
    value = cfg.get_value(data, "server_config_file", profile=profile)
    if value:
        return Path(str(value)).parent
    profile_name = profile or cfg.get_active_profile(data)
    return _server_config_dir(profile_name)


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
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override the config directory (default: profile-based).",
)
@click.pass_context
def server_run(ctx: click.Context, config_dir: str | None) -> None:
    """Run eval-hub-server in the foreground.

    \b
    Examples:
      evalhub server run
      evalhub server run --config-dir /path/to/config
      evalhub --profile staging server run
    """
    binary = _find_server_binary()
    cfg_dir = Path(config_dir) if config_dir else _resolve_config_dir(ctx)
    _require_config(cfg_dir)

    cmd = [binary, "-local", "-configdir", str(cfg_dir)]
    result = subprocess.run(
        cmd,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    ctx.exit(result.returncode)


@server.command("start")
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override the config directory (default: profile-based).",
)
@click.pass_context
def server_start(ctx: click.Context, config_dir: str | None) -> None:
    """Start eval-hub-server as a background daemon.

    \b
    Examples:
      evalhub server start
      evalhub server start --config-dir /path/to/config
      evalhub --profile staging server start
    """
    pid = _live_pid()
    if pid is not None:
        raise click.ClickException(
            f"Server is already running (PID {pid}). "
            "Stop it first with: evalhub server stop"
        )

    binary = _find_server_binary()
    cfg_dir = Path(config_dir) if config_dir else _resolve_config_dir(ctx)
    _require_config(cfg_dir)

    port = _read_server_port(cfg_dir)
    tls = _is_tls_enabled(cfg_dir)
    scheme = _server_scheme(tls)
    cmd = [binary, "-local", "-configdir", str(cfg_dir)]

    SERVER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = LOG_FILE.open("w")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    finally:
        log_fh.close()

    if not _wait_for_healthy(port, _STARTUP_TIMEOUT, tls=tls):
        if proc.poll() is not None:
            output = LOG_FILE.read_text().strip()
            msg = f"Server crashed on startup (exit code {proc.returncode})."
            if output:
                msg += f"\nLog output:\n{output}"
            raise click.ClickException(msg)
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
    pid = _live_pid()
    if pid is None:
        click.echo("Server is not running.")
        return

    os.kill(pid, _GRACEFUL_SIGNAL)

    deadline = time.monotonic() + _STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            PID_FILE.unlink(missing_ok=True)
            click.echo("Server stopped.")
            return
        time.sleep(0.2)

    os.kill(pid, _FORCE_SIGNAL)
    PID_FILE.unlink(missing_ok=True)
    click.echo("Server force-killed.")


@server.command("status")
@click.pass_context
def server_status(ctx: click.Context) -> None:
    """Check if the background eval-hub-server is running.

    \b
    Examples:
      evalhub server status
    """
    pid = _live_pid()
    if pid is None:
        click.echo("Server is not running.")
        return

    click.echo(f"Server is running (PID {pid}).")

    cfg_dir = _resolve_config_dir(ctx)
    port = _read_server_port(cfg_dir)
    tls = _is_tls_enabled(cfg_dir)
    scheme = _server_scheme(tls)

    if _health_check(port, tls=tls):
        click.echo("  Health: healthy")
    else:
        click.echo("  Health: not responding")

    click.echo(f"  URL:    {scheme}://localhost:{port}")
    click.echo(f"  Logs:   {LOG_FILE}")
