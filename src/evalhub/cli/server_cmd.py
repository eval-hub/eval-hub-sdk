"""Server command group — eval-hub-server lifecycle management."""

from __future__ import annotations

import json
import ssl
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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

SIDECAR_STATE_DIR = cfg.DEFAULT_CONFIG_DIR / "sidecar"
SIDECAR_PID_FILE = SIDECAR_STATE_DIR / "pid"
SIDECAR_LOG_FILE = SIDECAR_STATE_DIR / "sidecar.log"

_STARTUP_TIMEOUT = 30.0
_STARTUP_POLL = 0.5
_STOP_TIMEOUT = 5.0
_DEFAULT_PORT = 8080
_DEFAULT_SIDECAR_PORT = 8082
_SIDECAR_STARTUP_TIMEOUT = 15.0
_SIDECAR_MIN_VERSION = "1.0.1"


def _build_default_server_config() -> dict[str, Any]:
    return {
        "service": {"port": _DEFAULT_PORT},
        "database": {
            "driver": "sqlite",
            "url": "file::eval_hub:?mode=memory&cache=shared",
        },
        "sidecar": {
            "local_mode": True,
            "base_url": f"http://localhost:{_DEFAULT_SIDECAR_PORT}",
        },
    }


def _require_server_version() -> None:
    """Raise if the installed eval-hub-server is missing or too old."""
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    try:
        ver = version("eval-hub-server")
    except PackageNotFoundError:
        raise click.ClickException(
            "The 'eval-hub-server' package is not installed.\n"
            "Install it with: pip install eval-hub-server"
        )
    if Version(ver) < Version(_SIDECAR_MIN_VERSION):
        raise click.ClickException(
            f"eval-hub-server {ver} is installed but >= {_SIDECAR_MIN_VERSION} "
            f"is required.\n"
            f"Upgrade with: pip install 'eval-hub-server>={_SIDECAR_MIN_VERSION}'"
        )


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


def _read_sidecar_settings(
    config_dir: Path,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Return (sidecar_enabled, sidecar_base_url, local_settings) from the server config."""
    default_url = f"http://localhost:{_DEFAULT_SIDECAR_PORT}"
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        return False, default_url, None
    try:
        data = yaml.safe_load(config_path.read_text())
        sidecar = data.get("sidecar") if isinstance(data, dict) else None
        if not sidecar or not isinstance(sidecar, dict):
            return False, default_url, None
        local_mode = bool(sidecar.get("local_mode", False))
        base_url = str(sidecar.get("base_url", default_url))
        local_raw = sidecar.get("local")
        local_settings = dict(local_raw) if isinstance(local_raw, dict) else None
        return local_mode, base_url, local_settings
    except (yaml.YAMLError, TypeError, ValueError, AttributeError):
        return False, default_url, None


def _fetch_health_info(port: int, *, tls: bool = False) -> dict[str, Any] | None:
    """Fetch the full JSON response from the health endpoint."""
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
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                if isinstance(data, dict):
                    return data

    except Exception:
        pass
    return None


def _health_check(port: int, *, tls: bool = False) -> bool:
    return _fetch_health_info(port, tls=tls) is not None


_STOP_HINT_SERVER = "evalhub server stop, or Ctrl-C if running in the foreground"
# server stop handles sidecar shutdown even if the server is not running
_STOP_HINT_SIDECAR = "evalhub server stop"


def _require_server_not_listening(port: int, *, tls: bool = False) -> None:
    """Raise if a server is already responding on the configured port."""
    if _health_check(port, tls=tls):
        scheme = "https" if tls else "http"
        raise click.ClickException(
            f"A server is already responding at {scheme}://localhost:{port}.\n"
            f"Stop it first with: {_STOP_HINT_SERVER}"
        )


def _poll_until(check: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    delay = _STARTUP_POLL
    while time.monotonic() < deadline:
        if check():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 2.0)
    return check()


def _wait_for_healthy(port: int, timeout: float, *, tls: bool = False) -> bool:
    return _poll_until(lambda: _health_check(port, tls=tls), timeout)


def _sidecar_health_check(base_url: str) -> bool:
    """Check if the sidecar is healthy via GET <base_url>/health."""
    url = f"{base_url}/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def _wait_for_sidecar_healthy(base_url: str, timeout: float) -> bool:
    return _poll_until(lambda: _sidecar_health_check(base_url), timeout)


def _resolve_config_dir(ctx: click.Context) -> tuple[Path, bool]:
    """Return (config_dir, user_provided) for the active profile."""
    data = cfg.load_config()
    profile = ctx.obj.get("profile")
    profile_data = cfg.get_profile(data, profile)
    user_provided = "server_config_file" in profile_data
    config_dir = cfg.resolve_component_config_dir(
        data,
        SERVER_STATE_DIR,
        profile=profile,
    )
    return config_dir, user_provided


def _resolve_sidecar_config_dir(ctx: click.Context) -> Path:
    """Return the sidecar config directory for the active profile."""
    data = cfg.load_config()
    profile = ctx.obj.get("profile")
    return cfg.resolve_component_config_dir(data, SIDECAR_STATE_DIR, profile=profile)


def _ensure_config(config_dir: Path) -> None:
    config_path = config_dir / "config.yaml"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = _build_default_server_config()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    click.echo(f"Using default server config at {config_path}.")


def _generate_sidecar_config(
    server_port: int,
    sidecar_base_url: str,
    config_dir: Path,
    local_settings: dict[str, Any] | None = None,
    *,
    server_tls: bool = False,
) -> Path:
    """Generate sidecar-server-config.json for the eval-runtime-sidecar binary."""
    config_dir.mkdir(parents=True, exist_ok=True)
    server_scheme = "https" if server_tls else "http"
    config: dict[str, Any] = {
        "base_url": sidecar_base_url,
        "local_mode": True,
        "eval_hub": {
            "base_url": f"{server_scheme}://localhost:{server_port}",
        },
    }
    if local_settings:
        config["local"] = local_settings
    config_path = config_dir / "sidecar-server-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def _start_sidecar(
    ctx: click.Context,
    server_port: int,
    sidecar_base_url: str,
    local_settings: dict[str, Any] | None = None,
    *,
    server_tls: bool = False,
) -> int:
    """Start the sidecar process and return its PID."""
    sidecar_binary = find_binary("eval-runtime-sidecar", "EVALHUB_SIDECAR_BIN")
    if _sidecar_health_check(sidecar_base_url):
        raise click.ClickException(
            f"A sidecar is already responding at {sidecar_base_url}.\n"
            f"Stop it first with: {_STOP_HINT_SIDECAR}"
        )
    sidecar_cfg_dir = _resolve_sidecar_config_dir(ctx)
    config_path = _generate_sidecar_config(
        server_port,
        sidecar_base_url,
        sidecar_cfg_dir,
        local_settings,
        server_tls=server_tls,
    )

    proc = spawn_background(
        [sidecar_binary, "-sidecarconfig", str(config_path)],
        SIDECAR_STATE_DIR,
        SIDECAR_LOG_FILE,
    )

    if not _wait_for_sidecar_healthy(sidecar_base_url, _SIDECAR_STARTUP_TIMEOUT):
        proc.terminate()
        try:
            proc.wait(timeout=_STOP_TIMEOUT)
        except Exception:
            proc.kill()
            proc.wait(timeout=2)
        raise click.ClickException(
            f"Sidecar did not become healthy within {_SIDECAR_STARTUP_TIMEOUT}s.\n"
            f"Check logs at: {SIDECAR_LOG_FILE}"
        )

    if proc.poll() is not None:
        raise click.ClickException(
            f"Sidecar exited on startup (exit code {proc.returncode}).\n"
            f"Check logs at: {SIDECAR_LOG_FILE}"
        )

    SIDECAR_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR_PID_FILE.write_text(str(proc.pid))
    return proc.pid


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


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
    _require_server_version()
    require_not_running(PID_FILE, "Server", _STOP_HINT_SERVER)
    require_not_running(SIDECAR_PID_FILE, "Sidecar", _STOP_HINT_SIDECAR)

    binary = find_binary("eval-hub-server", "EVALHUB_SERVER_BIN")
    cfg_dir, user_provided = _resolve_config_dir(ctx)
    if not user_provided:
        _ensure_config(cfg_dir)

    port, tls = _read_server_config(cfg_dir)
    _require_server_not_listening(port, tls=tls)

    sidecar_enabled, sidecar_base_url, local_settings = _read_sidecar_settings(cfg_dir)

    if sidecar_enabled:
        sidecar_pid = _start_sidecar(
            ctx, port, sidecar_base_url, local_settings, server_tls=tls
        )
        click.echo(f"Sidecar started (PID {sidecar_pid}).")

    try:
        run_foreground([binary, "-local", "-configdir", str(cfg_dir)], ctx)
    finally:
        if sidecar_enabled:
            stop_daemon(SIDECAR_PID_FILE, _STOP_TIMEOUT, "Sidecar")


@server.command("start")
@click.pass_context
def server_start(ctx: click.Context) -> None:
    """Start eval-hub-server as a background daemon.

    \b
    Examples:
      evalhub server start
      evalhub --profile staging server start
    """
    _require_server_version()
    require_not_running(PID_FILE, "Server", _STOP_HINT_SERVER)
    require_not_running(SIDECAR_PID_FILE, "Sidecar", _STOP_HINT_SIDECAR)

    binary = find_binary("eval-hub-server", "EVALHUB_SERVER_BIN")
    cfg_dir, user_provided = _resolve_config_dir(ctx)
    if not user_provided:
        _ensure_config(cfg_dir)

    port, tls = _read_server_config(cfg_dir)
    scheme = "https" if tls else "http"
    _require_server_not_listening(port, tls=tls)

    sidecar_enabled, sidecar_base_url, local_settings = _read_sidecar_settings(cfg_dir)

    sidecar_pid = None
    if sidecar_enabled:
        sidecar_pid = _start_sidecar(
            ctx, port, sidecar_base_url, local_settings, server_tls=tls
        )

    cmd = [binary, "-local", "-configdir", str(cfg_dir)]
    proc = spawn_background(cmd, SERVER_STATE_DIR, LOG_FILE)

    if not _wait_for_healthy(port, _STARTUP_TIMEOUT, tls=tls):
        if sidecar_enabled:
            stop_daemon(SIDECAR_PID_FILE, _STOP_TIMEOUT, "Sidecar")
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
    if sidecar_pid is not None:
        click.echo(f"  Sidecar: {sidecar_base_url} (PID {sidecar_pid})")
    click.echo(f"  Logs: {LOG_FILE}")


@server.command("stop")
def server_stop() -> None:
    """Stop the background eval-hub-server and sidecar.

    \b
    Examples:
      evalhub server stop
    """
    stop_daemon(PID_FILE, _STOP_TIMEOUT, "Server")
    stop_daemon(SIDECAR_PID_FILE, _STOP_TIMEOUT, "Sidecar")


@server.command("status")
@click.pass_context
def server_status(ctx: click.Context) -> None:
    """Check if eval-hub-server and sidecar are running.

    Works for both background (server start) and foreground (server run)
    by probing the health endpoint directly.

    \b
    Examples:
      evalhub server status
    """
    cfg_dir, _ = _resolve_config_dir(ctx)
    port, tls = _read_server_config(cfg_dir)
    scheme = "https" if tls else "http"

    pid = live_pid(PID_FILE)
    info = _fetch_health_info(port, tls=tls)

    if not info and pid is None:
        click.echo("Server is not running.")
    else:
        if pid is not None:
            click.echo(f"Server is running (PID {pid}).")
        else:
            click.echo("Server is running.")
        click.echo(f"  Health: {'healthy' if info else 'not responding'}")
        if info:
            if info.get("build"):
                click.echo(f"  Version: {info['build']}")
            if info.get("git_hash"):
                click.echo(f"  Commit:  {info['git_hash']}")
        click.echo(f"  URL:    {scheme}://localhost:{port}")
        if pid is not None:
            click.echo(f"  Logs:   {LOG_FILE}")

    sidecar_pid = live_pid(SIDECAR_PID_FILE)
    sidecar_enabled, sidecar_base_url, _ = _read_sidecar_settings(cfg_dir)

    if sidecar_pid is not None:
        parsed = urlparse(sidecar_base_url)
        sidecar_port = parsed.port or _DEFAULT_SIDECAR_PORT
        healthy = _sidecar_health_check(sidecar_base_url)
        click.echo(
            f"Sidecar is running "
            f"(PID {sidecar_pid}, port {sidecar_port}, mode: local)."
        )
        click.echo(f"  Health: {'healthy' if healthy else 'not responding'}")
        click.echo(f"  Logs:   {SIDECAR_LOG_FILE}")
    elif sidecar_enabled:
        click.echo("Sidecar is not running.")
