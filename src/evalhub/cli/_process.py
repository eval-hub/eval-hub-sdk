"""Shared process lifecycle helpers for CLI daemon commands."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

GRACEFUL_SIGNAL: signal.Signals = (
    signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM  # type: ignore[attr-defined]
)
FORCE_SIGNAL: signal.Signals = (
    signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
)


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if pid > 0 else None


def live_pid(pid_file: Path) -> int | None:
    pid = read_pid(pid_file)
    if pid is not None and not is_process_alive(pid):
        pid_file.unlink(missing_ok=True)
        return None
    return pid


def find_binary(name: str, env_var: str) -> str:
    env = os.environ.get(env_var)
    if env:
        if not Path(env).is_file():
            raise click.ClickException(
                f"{env_var} is set to '{env}' but that file does not exist."
            )
        return env
    found = shutil.which(name)
    if found:
        return found
    raise click.ClickException(
        f"Could not find the '{name}' binary.\n"
        f"Install it and ensure it is on your PATH, or set {env_var}."
    )


def graceful_stop(pid: int, pid_file: Path, timeout: float, label: str) -> None:
    os.kill(pid, GRACEFUL_SIGNAL)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            pid_file.unlink(missing_ok=True)
            click.echo(f"{label} stopped.")
            return
        time.sleep(0.2)
    os.kill(pid, FORCE_SIGNAL)
    pid_file.unlink(missing_ok=True)
    click.echo(f"{label} force-killed.")


def spawn_background(
    cmd: list[str], state_dir: Path, log_file: Path
) -> subprocess.Popen[bytes]:
    state_dir.mkdir(parents=True, exist_ok=True)
    log_fh = log_file.open("w")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    finally:
        log_fh.close()


# ---------------------------------------------------------------------------
# High-level daemon lifecycle helpers
# ---------------------------------------------------------------------------


def run_foreground(cmd: list[str], ctx: click.Context) -> None:
    """Run *cmd* in the foreground, forwarding stdio and exit code."""
    result = subprocess.run(
        cmd,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    ctx.exit(result.returncode)


def require_not_running(pid_file: Path, label: str, stop_hint: str) -> None:
    """Raise if a daemon tracked by *pid_file* is already alive."""
    pid = live_pid(pid_file)
    if pid is not None:
        raise click.ClickException(
            f"{label} is already running (PID {pid}). "
            f"Stop it first with: {stop_hint}"
        )


def stop_daemon(pid_file: Path, timeout: float, label: str) -> None:
    """Stop a daemon tracked by *pid_file*, or report that it is not running."""
    pid = live_pid(pid_file)
    if pid is None:
        click.echo(f"{label} is not running.")
        return
    graceful_stop(pid, pid_file, timeout, label)
