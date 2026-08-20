"""Unit tests for the EvalHub CLI server subcommand."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml
from click.testing import CliRunner
from evalhub.cli.main import main
from evalhub.cli.server_cmd import _require_server_version as _real_require_version

pytestmark = pytest.mark.unit


@pytest.fixture()
def config_file(tmp_path: Path) -> Iterator[Path]:
    """Provide a temporary config file path and isolate from env vars."""
    path = tmp_path / "config.yaml"
    saved_config = os.environ.get("EVALHUB_CONFIG")
    saved_token = os.environ.get("EVALHUB_TOKEN")
    os.environ["EVALHUB_CONFIG"] = str(path)
    os.environ.pop("EVALHUB_TOKEN", None)
    yield path
    if saved_config is not None:
        os.environ["EVALHUB_CONFIG"] = saved_config
    else:
        os.environ.pop("EVALHUB_CONFIG", None)
    if saved_token is not None:
        os.environ["EVALHUB_TOKEN"] = saved_token
    else:
        os.environ.pop("EVALHUB_TOKEN", None)


def _seed_profile(config_file: Path, profile: str = "default", **kwargs: str) -> None:
    """Write a profile into the config file."""
    data: dict[str, object] = {"active_profile": profile, "profiles": {profile: kwargs}}
    config_file.write_text(yaml.safe_dump(data))


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate_server_state(tmp_path: Path) -> Iterator[None]:
    """Isolate tests from real PID/log files, version checks, and port probes."""
    sidecar_dir = tmp_path / "sidecar_state"
    with patch("evalhub.cli.server_cmd.SIDECAR_STATE_DIR", sidecar_dir), patch(
        "evalhub.cli.server_cmd.SIDECAR_PID_FILE", sidecar_dir / "pid"
    ), patch(
        "evalhub.cli.server_cmd.SIDECAR_LOG_FILE", sidecar_dir / "sidecar.log"
    ), patch("evalhub.cli.server_cmd._require_server_version"), patch(
        "evalhub.cli.server_cmd._require_server_not_listening"
    ):
        yield


# ---------------------------------------------------------------------------
# Help / discoverability
# ---------------------------------------------------------------------------


def test_server_appears_in_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "server" in result.output


def test_server_subcommands_appear_in_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["server", "--help"])
    assert result.exit_code == 0
    for sub in ("run", "start", "stop", "status"):
        assert sub in result.output


def _setup_server_config(
    tmp_path: Path,
    config_file: Path,
    profile: str = "default",
    *,
    tls: bool = False,
) -> Path:
    """Create a minimal server config and register it in the CLI profile."""
    cfg_dir = tmp_path / profile
    cfg_dir.mkdir(parents=True, exist_ok=True)
    svc: dict[str, object] = {"port": 8080}
    if tls:
        svc["tls_cert_file"] = "/tmp/server.crt"
        svc["tls_key_file"] = "/tmp/server.key"
    server_yaml = cfg_dir / "config.yaml"
    server_yaml.write_text(yaml.safe_dump({"service": svc}))
    data = (
        yaml.safe_load(config_file.read_text())
        if config_file.exists()
        else {"active_profile": profile, "profiles": {profile: {}}}
    )
    data.setdefault("profiles", {}).setdefault(profile, {})["server_config_file"] = str(
        server_yaml
    )
    config_file.write_text(yaml.safe_dump(data))
    return cfg_dir


# ---------------------------------------------------------------------------
# _require_server_not_listening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subcmd", ["start", "run"])
@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_refuses_when_already_listening(
    mock_find: MagicMock,
    subcmd: str,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Both start and run refuse if the port already has a server responding."""
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)

    pid_file = tmp_path / "pid"
    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli.server_cmd._require_server_not_listening",
        side_effect=click.ClickException(
            "A server is already responding at http://localhost:8080.\n"
            "Stop it first with: evalhub server stop, "
            "or Ctrl-C if running in the foreground"
        ),
    ):
        result = runner.invoke(main, ["server", subcmd])

    assert result.exit_code != 0
    assert "already responding" in result.output
    assert "evalhub server stop" in result.output


# ---------------------------------------------------------------------------
# server run
# ---------------------------------------------------------------------------


@patch("evalhub.cli._process.subprocess.run")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_run_foreground(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)
    mock_run.return_value = MagicMock(returncode=0)

    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path):
        result = runner.invoke(main, ["server", "run"])

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/eval-hub-server"
    assert "-local" in cmd
    assert "-configdir" in cmd


@patch("evalhub.cli.server_cmd._wait_for_sidecar_healthy", return_value=True)
@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    side_effect=lambda name, _env: f"/usr/bin/{name}",
)
def test_server_run_generates_default_config_when_missing(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_run: MagicMock,
    mock_sidecar_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)

    mock_proc = MagicMock()
    mock_proc.pid = 54321
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    mock_run.return_value = MagicMock(returncode=0)

    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli._process.is_process_alive", return_value=False
    ):
        result = runner.invoke(main, ["server", "run"])

    assert result.exit_code == 0, result.output
    assert "Using default server config" in result.output

    generated = tmp_path / "default" / "config.yaml"
    assert generated.exists()
    loaded = yaml.safe_load(generated.read_text())
    assert loaded["service"]["port"] == 8080
    assert loaded["database"]["driver"] == "sqlite"
    assert loaded["sidecar"]["local_mode"] is True


def test_server_run_binary_not_found(
    runner: CliRunner,
    config_file: Path,
) -> None:
    from click import ClickException

    _seed_profile(config_file)

    with patch(
        "evalhub.cli.server_cmd.find_binary",
        side_effect=ClickException(
            "Could not find the 'eval-hub-server' binary.\n"
            "Install it and ensure it is on your PATH, or set EVALHUB_SERVER_BIN."
        ),
    ):
        result = runner.invoke(main, ["server", "run"])

    assert result.exit_code != 0
    assert "eval-hub-server" in result.output


# ---------------------------------------------------------------------------
# server start
# ---------------------------------------------------------------------------


@patch("evalhub.cli.server_cmd._wait_for_healthy", return_value=True)
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_start_launches_background(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)

    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    pid_file = tmp_path / "pid"
    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code == 0, result.output
    assert "12345" in result.output
    assert "http://localhost:8080" in result.output

    cmd = mock_popen.call_args[0][0]
    assert "-local" in cmd
    assert "-configdir" in cmd

    assert pid_file.exists()
    assert pid_file.read_text().strip() == "12345"


@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_start_already_running(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    pid_file = tmp_path / "pid"
    pid_file.write_text("99999")

    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code != 0
    assert "already running" in result.output
    assert "evalhub server stop" in result.output
    mock_popen.assert_not_called()


@patch("evalhub.cli.server_cmd._wait_for_healthy", return_value=False)
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_start_crash_on_startup(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)

    mock_proc = MagicMock()
    mock_proc.pid = 11111
    mock_proc.poll.return_value = 1
    mock_proc.returncode = 1
    mock_popen.return_value = mock_proc

    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code != 0
    assert "crashed on startup" in result.output


@patch("evalhub.cli.server_cmd._wait_for_healthy", return_value=False)
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_start_health_check_timeout(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)

    mock_proc = MagicMock()
    mock_proc.pid = 22222
    mock_proc.poll.return_value = None  # process still alive, just not healthy
    mock_popen.return_value = mock_proc

    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code != 0
    assert "not become healthy" in result.output


@patch("evalhub.cli.server_cmd._wait_for_healthy", return_value=True)
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_start_tls_uses_https_scheme(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file, tls=True)

    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    pid_file = tmp_path / "pid"
    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code == 0, result.output
    assert "https://localhost:8080" in result.output
    assert "http://localhost:8080" not in result.output
    mock_healthy.assert_called_once_with(8080, 30.0, tls=True)


@patch("evalhub.cli.server_cmd._wait_for_sidecar_healthy", return_value=True)
@patch("evalhub.cli.server_cmd._wait_for_healthy", return_value=True)
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    side_effect=lambda name, _env: f"/usr/bin/{name}",
)
def test_server_start_generates_default_config_when_missing(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_healthy: MagicMock,
    mock_sidecar_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)

    mock_proc_sidecar = MagicMock()
    mock_proc_sidecar.pid = 54321
    mock_proc_sidecar.poll.return_value = None

    mock_proc_server = MagicMock()
    mock_proc_server.pid = 12345
    mock_proc_server.poll.return_value = None

    mock_popen.side_effect = [mock_proc_sidecar, mock_proc_server]

    pid_file = tmp_path / "pid"
    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code == 0, result.output
    assert "Using default server config" in result.output
    assert "12345" in result.output

    generated = tmp_path / "default" / "config.yaml"
    assert generated.exists()
    loaded = yaml.safe_load(generated.read_text())
    assert loaded["service"]["port"] == 8080
    assert loaded["database"]["driver"] == "sqlite"
    assert loaded["sidecar"]["local_mode"] is True


# ---------------------------------------------------------------------------
# server stop
# ---------------------------------------------------------------------------


@patch("evalhub.cli._process.os.kill")
def test_server_stop_success(
    mock_kill: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")

    alive_calls = iter([True, False])

    with patch("evalhub.cli.server_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli._process.is_process_alive", side_effect=alive_calls
    ), patch("evalhub.cli._process.time.sleep"):
        result = runner.invoke(main, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert "stopped" in result.output
    assert not pid_file.exists()


def test_server_stop_not_running(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)

    with patch("evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"):
        result = runner.invoke(main, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert "not running" in result.output


@patch("evalhub.cli._process.os.kill")
def test_server_stop_force_kill(
    mock_kill: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")

    with patch("evalhub.cli.server_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ), patch("evalhub.cli._process.time.sleep"), patch(
        "evalhub.cli.server_cmd._STOP_TIMEOUT", 0
    ):
        result = runner.invoke(main, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert "force-killed" in result.output
    assert not pid_file.exists()


# ---------------------------------------------------------------------------
# server status
# ---------------------------------------------------------------------------


def test_server_status_not_running(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)

    with patch("evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd._fetch_health_info", return_value=None):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "not running" in result.output


def test_server_status_running_healthy(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")
    _setup_server_config(tmp_path, config_file)

    health_info = {"status": "healthy", "build": "0.4.4", "git_hash": "f758919"}

    with patch("evalhub.cli.server_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ), patch("evalhub.cli.server_cmd._fetch_health_info", return_value=health_info):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "12345" in result.output
    assert "healthy" in result.output
    assert "Version: 0.4.4" in result.output
    assert "Commit:  f758919" in result.output
    assert "http://localhost:8080" in result.output
    assert "Logs:" in result.output


def test_server_status_healthy_no_pid(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Status detects a foreground server via health endpoint even without a PID file."""
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)

    health_info = {"status": "healthy", "build": "1.0.0"}

    with patch("evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd._fetch_health_info", return_value=health_info):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "healthy" in result.output
    assert "Version: 1.0.0" in result.output
    assert "Commit:" not in result.output
    assert "http://localhost:8080" in result.output
    assert "PID" not in result.output
    assert "Logs:" not in result.output


def test_server_status_tls_uses_https_scheme(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")
    _setup_server_config(tmp_path, config_file, tls=True)

    health_info = {"status": "healthy"}

    with patch("evalhub.cli.server_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ), patch(
        "evalhub.cli.server_cmd._fetch_health_info", return_value=health_info
    ) as mock_fhi:
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "https://localhost:8080" in result.output
    assert "http://localhost:8080" not in result.output
    mock_fhi.assert_called_once_with(8080, tls=True)


def test_server_status_running_unhealthy(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")
    _setup_server_config(tmp_path, config_file)

    with patch("evalhub.cli.server_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ), patch("evalhub.cli.server_cmd._fetch_health_info", return_value=None):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "not responding" in result.output
    assert "Version:" not in result.output
    assert "Commit:" not in result.output


# ---------------------------------------------------------------------------
# config set/get/unset server_config_file
# ---------------------------------------------------------------------------


def _patch_store_dir(tmp_path: Path) -> Any:
    """Patch _FILE_KEY_STORE_DIRS so file keys write under tmp_path."""
    return patch(
        "evalhub.cli.config._FILE_KEY_STORE_DIRS",
        {"server_config_file": tmp_path / "server"},
    )


def test_config_set_server_config_file_copies_and_stores(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "myconfig.yaml"
    src.write_text(yaml.safe_dump({"service": {"port": 9090}}))

    with _patch_store_dir(tmp_path):
        result = runner.invoke(main, ["config", "set", "server_config_file", str(src)])

    assert result.exit_code == 0, result.output
    assert "server_config_file" in result.output

    dest = tmp_path / "server" / "default" / "config.yaml"
    assert dest.exists()
    loaded = yaml.safe_load(dest.read_text())
    assert loaded["service"]["port"] == 9090

    get_result = runner.invoke(main, ["config", "get", "server_config_file"])
    assert get_result.exit_code == 0
    assert str(dest) in get_result.output


def test_config_set_server_config_file_validates_yaml(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "bad.yaml"
    src.write_text(": :\n  - :\n  bad: [unterminated")

    result = runner.invoke(main, ["config", "set", "server_config_file", str(src)])

    assert result.exit_code != 0
    assert "Invalid YAML" in result.output


def test_config_set_server_config_file_rejects_non_mapping(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "list.yaml"
    src.write_text("- item1\n- item2\n")

    result = runner.invoke(main, ["config", "set", "server_config_file", str(src)])

    assert result.exit_code != 0
    assert "mapping" in result.output


def test_config_set_server_config_file_not_found(
    runner: CliRunner,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    result = runner.invoke(
        main, ["config", "set", "server_config_file", "/nonexistent/path.yaml"]
    )
    assert result.exit_code != 0
    assert "File not found" in result.output


def test_config_set_server_config_file_respects_profile(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    data = {
        "active_profile": "default",
        "profiles": {
            "default": {"base_url": "http://localhost:8080"},
            "staging": {"base_url": "https://staging.example.com"},
        },
    }
    config_file.write_text(yaml.safe_dump(data))

    src = tmp_path / "staging.yaml"
    src.write_text(yaml.safe_dump({"service": {"port": 8081}}))

    with _patch_store_dir(tmp_path):
        result = runner.invoke(
            main,
            [
                "--profile",
                "staging",
                "config",
                "set",
                "server_config_file",
                str(src),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "staging" in result.output
    assert (tmp_path / "server" / "staging" / "config.yaml").exists()
    assert not (tmp_path / "server" / "default" / "config.yaml").exists()


def test_config_get_server_config_file_unfold(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "myconfig.yaml"
    content = yaml.safe_dump(
        {"service": {"port": 9090}, "database": {"path": "data.db"}}
    )
    src.write_text(content)

    with _patch_store_dir(tmp_path):
        runner.invoke(main, ["config", "set", "server_config_file", str(src)])

    result = runner.invoke(main, ["config", "get", "server_config_file", "--unfold"])
    assert result.exit_code == 0, result.output
    assert "9090" in result.output
    assert "data.db" in result.output


def test_config_get_unfold_file_missing(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "myconfig.yaml"
    src.write_text(yaml.safe_dump({"key": "value"}))

    with _patch_store_dir(tmp_path):
        runner.invoke(main, ["config", "set", "server_config_file", str(src)])

    dest = tmp_path / "server" / "default" / "config.yaml"
    dest.unlink()

    result = runner.invoke(main, ["config", "get", "server_config_file", "--unfold"])
    assert result.exit_code != 0
    assert "File not found" in result.output


def test_config_get_unfold_non_file_key_errors(
    runner: CliRunner,
    config_file: Path,
) -> None:
    _seed_profile(config_file, base_url="http://localhost:8080")
    result = runner.invoke(main, ["config", "get", "base_url", "--unfold"])
    assert result.exit_code != 0
    assert "file-based" in result.output


def test_config_get_unfold_masks_sensitive_keys(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "myconfig.yaml"
    src.write_text(
        yaml.safe_dump({"base_url": "http://localhost", "token": "super-secret-tok"})
    )

    with _patch_store_dir(tmp_path):
        runner.invoke(main, ["config", "set", "server_config_file", str(src)])

    result = runner.invoke(main, ["config", "get", "server_config_file", "--unfold"])
    assert result.exit_code == 0, result.output
    assert "super-secret-tok" not in result.output
    assert "sup***ok" in result.output
    assert "http://localhost" in result.output


def test_config_get_unfold_with_unmask(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "myconfig.yaml"
    src.write_text(
        yaml.safe_dump({"base_url": "http://localhost", "token": "super-secret-tok"})
    )

    with _patch_store_dir(tmp_path):
        runner.invoke(main, ["config", "set", "server_config_file", str(src)])

    result = runner.invoke(
        main, ["config", "get", "server_config_file", "--unfold", "--unmask"]
    )
    assert result.exit_code == 0, result.output
    assert "super-secret-tok" in result.output


def test_config_unset_server_config_file_deletes_stored_copy(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "myconfig.yaml"
    src.write_text(yaml.safe_dump({"key": "value"}))

    with _patch_store_dir(tmp_path):
        runner.invoke(main, ["config", "set", "server_config_file", str(src)])

        dest = tmp_path / "server" / "default" / "config.yaml"
        assert dest.exists()

        result = runner.invoke(main, ["config", "unset", "server_config_file"])

    assert result.exit_code == 0, result.output
    assert "Unset" in result.output
    assert not dest.exists()
    assert not (tmp_path / "server" / "default").exists()

    get_result = runner.invoke(main, ["config", "get", "server_config_file"])
    assert get_result.exit_code != 0


def test_config_list_shows_server_config_file(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "myconfig.yaml"
    src.write_text(yaml.safe_dump({"key": "value"}))

    with _patch_store_dir(tmp_path):
        runner.invoke(main, ["config", "set", "server_config_file", str(src)])

    result = runner.invoke(main, ["config", "list"])

    assert result.exit_code == 0, result.output
    assert "server_config_file" in result.output


def test_config_set_then_unfold_roundtrip(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)
    src = tmp_path / "myconfig.yaml"
    original = {"service": {"port": 7070, "host": "0.0.0.0"}}
    src.write_text(yaml.safe_dump(original))

    with _patch_store_dir(tmp_path):
        set_result = runner.invoke(
            main, ["config", "set", "server_config_file", str(src)]
        )
    assert set_result.exit_code == 0, set_result.output

    unfold_result = runner.invoke(
        main, ["config", "get", "server_config_file", "--unfold"]
    )
    assert unfold_result.exit_code == 0, unfold_result.output
    loaded = yaml.safe_load(unfold_result.output)
    assert loaded == original


# ---------------------------------------------------------------------------
# _ensure_config
# ---------------------------------------------------------------------------


def test_ensure_config_writes_default(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _ensure_config

    config_dir = tmp_path / "profile"
    _ensure_config(config_dir)

    config_path = config_dir / "config.yaml"
    assert config_path.exists()

    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["service"]["port"] == 8080
    assert loaded["database"]["driver"] == "sqlite"
    assert loaded["database"]["url"] == "file::eval_hub:?mode=memory&cache=shared"
    assert loaded["sidecar"]["local_mode"] is True
    assert loaded["sidecar"]["base_url"] == "http://localhost:8082"


def test_ensure_config_overwrites_existing(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _ensure_config

    config_dir = tmp_path / "profile"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump({"service": {"port": 9999}}))

    _ensure_config(config_dir)

    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["service"]["port"] == 8080
    assert loaded["sidecar"]["local_mode"] is True


# ---------------------------------------------------------------------------
# _require_server_version
# ---------------------------------------------------------------------------


def test_require_server_version_above_min() -> None:
    with patch("importlib.metadata.version", return_value="2.0.0"):
        _real_require_version()


def test_require_server_version_at_min() -> None:
    with patch("importlib.metadata.version", return_value="1.0.1"):
        _real_require_version()


def test_require_server_version_below_min() -> None:
    with patch("importlib.metadata.version", return_value="1.0.0"):
        with pytest.raises(click.ClickException, match="1.0.1"):
            _real_require_version()


def test_require_server_version_not_installed() -> None:
    from importlib.metadata import PackageNotFoundError

    with patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError("eval-hub-server"),
    ):
        with pytest.raises(click.ClickException, match="not installed"):
            _real_require_version()


# ---------------------------------------------------------------------------
# _build_default_server_config
# ---------------------------------------------------------------------------


def test_build_default_server_config() -> None:
    from evalhub.cli.server_cmd import _build_default_server_config

    config = _build_default_server_config()
    assert config["service"]["port"] == 8080
    assert config["database"]["driver"] == "sqlite"
    assert config["sidecar"]["local_mode"] is True
    assert config["sidecar"]["base_url"] == "http://localhost:8082"


# ---------------------------------------------------------------------------
# _generate_sidecar_config
# ---------------------------------------------------------------------------


def test_generate_sidecar_config(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _generate_sidecar_config

    config_dir = tmp_path / "sidecar" / "default"
    result_path = _generate_sidecar_config(8080, "http://localhost:8082", config_dir)

    assert result_path == config_dir / "sidecar-server-config.json"
    assert result_path.exists()

    loaded = json.loads(result_path.read_text())
    assert loaded["base_url"] == "http://localhost:8082"
    assert loaded["local_mode"] is True
    assert "local" not in loaded
    assert loaded["eval_hub"]["base_url"] == "http://localhost:8080"


def test_generate_sidecar_config_custom_ports(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _generate_sidecar_config

    config_dir = tmp_path / "sidecar" / "staging"
    result_path = _generate_sidecar_config(9090, "http://localhost:9092", config_dir)

    loaded = json.loads(result_path.read_text())
    assert loaded["base_url"] == "http://localhost:9092"
    assert loaded["eval_hub"]["base_url"] == "http://localhost:9090"


def test_generate_sidecar_config_with_local_settings(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _generate_sidecar_config

    config_dir = tmp_path / "sidecar" / "default"
    local_settings = {
        "job_cache_sweep_interval": "1h30m",
        "job_cache_entry_ttl": "2h",
    }
    result_path = _generate_sidecar_config(
        8080, "http://localhost:8082", config_dir, local_settings
    )

    loaded = json.loads(result_path.read_text())
    assert loaded["base_url"] == "http://localhost:8082"
    assert loaded["local_mode"] is True
    assert loaded["local"] == {
        "job_cache_sweep_interval": "1h30m",
        "job_cache_entry_ttl": "2h",
    }
    assert loaded["eval_hub"]["base_url"] == "http://localhost:8080"


def test_generate_sidecar_config_without_local_settings(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _generate_sidecar_config

    config_dir = tmp_path / "sidecar" / "default"
    result_path = _generate_sidecar_config(
        8080, "http://localhost:8082", config_dir, None
    )

    loaded = json.loads(result_path.read_text())
    assert "local" not in loaded


# ---------------------------------------------------------------------------
# _read_sidecar_settings
# ---------------------------------------------------------------------------


def test_read_sidecar_settings_with_sidecar(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _read_sidecar_settings

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "service": {"port": 8080},
                "sidecar": {
                    "local_mode": True,
                    "base_url": "http://localhost:8082",
                },
            }
        )
    )

    enabled, base_url, local_settings = _read_sidecar_settings(config_dir)
    assert enabled is True
    assert base_url == "http://localhost:8082"
    assert local_settings is None


def test_read_sidecar_settings_without_sidecar(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _read_sidecar_settings

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(yaml.safe_dump({"service": {"port": 8080}}))

    enabled, base_url, local_settings = _read_sidecar_settings(config_dir)
    assert enabled is False
    assert base_url == "http://localhost:8082"
    assert local_settings is None


def test_read_sidecar_settings_local_mode_false(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _read_sidecar_settings

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "service": {"port": 8080},
                "sidecar": {
                    "local_mode": False,
                    "base_url": "http://localhost:8082",
                },
            }
        )
    )

    enabled, base_url, local_settings = _read_sidecar_settings(config_dir)
    assert enabled is False
    assert local_settings is None


def test_read_sidecar_settings_no_config_file(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _read_sidecar_settings

    enabled, base_url, local_settings = _read_sidecar_settings(tmp_path / "nonexistent")
    assert enabled is False
    assert local_settings is None


def test_read_sidecar_settings_with_local_section(tmp_path: Path) -> None:
    from evalhub.cli.server_cmd import _read_sidecar_settings

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "service": {"port": 8080},
                "sidecar": {
                    "local_mode": True,
                    "base_url": "http://localhost:8082",
                    "local": {
                        "job_cache_sweep_interval": "1h30m",
                        "job_cache_entry_ttl": "2h",
                    },
                },
            }
        )
    )

    enabled, base_url, local_settings = _read_sidecar_settings(config_dir)
    assert enabled is True
    assert base_url == "http://localhost:8082"
    assert local_settings == {
        "job_cache_sweep_interval": "1h30m",
        "job_cache_entry_ttl": "2h",
    }


# ---------------------------------------------------------------------------
# Sidecar helpers for lifecycle tests
# ---------------------------------------------------------------------------


def _setup_server_config_with_sidecar(
    tmp_path: Path,
    config_file: Path,
    profile: str = "default",
    *,
    sidecar_base_url: str = "http://localhost:8082",
) -> Path:
    """Create a server config with sidecar section and register it."""
    cfg_dir = tmp_path / profile
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "service": {"port": 8080},
        "database": {
            "driver": "sqlite",
            "url": "file::eval_hub:?mode=memory&cache=shared",
        },
        "sidecar": {
            "local_mode": True,
            "base_url": sidecar_base_url,
        },
    }
    server_yaml = cfg_dir / "config.yaml"
    server_yaml.write_text(yaml.safe_dump(config))
    data = (
        yaml.safe_load(config_file.read_text())
        if config_file.exists()
        else {"active_profile": profile, "profiles": {profile: {}}}
    )
    data.setdefault("profiles", {}).setdefault(profile, {})["server_config_file"] = str(
        server_yaml
    )
    config_file.write_text(yaml.safe_dump(data))
    return cfg_dir


# ---------------------------------------------------------------------------
# server start — sidecar lifecycle
# ---------------------------------------------------------------------------


@patch("evalhub.cli.server_cmd._wait_for_sidecar_healthy", return_value=True)
@patch("evalhub.cli.server_cmd._wait_for_healthy", return_value=True)
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    side_effect=lambda name, _env: f"/usr/bin/{name}",
)
def test_server_start_with_sidecar(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_healthy: MagicMock,
    mock_sidecar_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Both server and sidecar started when config has sidecar section."""
    _seed_profile(config_file)
    _setup_server_config_with_sidecar(tmp_path, config_file)

    mock_proc_sidecar = MagicMock()
    mock_proc_sidecar.pid = 54321
    mock_proc_sidecar.poll.return_value = None

    mock_proc_server = MagicMock()
    mock_proc_server.pid = 12345
    mock_proc_server.poll.return_value = None

    mock_popen.side_effect = [mock_proc_sidecar, mock_proc_server]

    pid_file = tmp_path / "pid"
    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code == 0, result.output
    assert "12345" in result.output
    assert "Sidecar" in result.output
    assert "54321" in result.output
    assert "http://localhost:8082" in result.output

    assert mock_popen.call_count == 2
    sidecar_cmd = mock_popen.call_args_list[0][0][0]
    assert sidecar_cmd[0] == "/usr/bin/eval-runtime-sidecar"
    assert "-sidecarconfig" in sidecar_cmd

    sidecar_config_path = Path(sidecar_cmd[sidecar_cmd.index("-sidecarconfig") + 1])
    assert sidecar_config_path.name == "sidecar-server-config.json"
    loaded = json.loads(sidecar_config_path.read_text())
    assert loaded["local_mode"] is True
    assert loaded["eval_hub"]["base_url"] == "http://localhost:8080"


@patch("evalhub.cli.server_cmd._wait_for_healthy", return_value=True)
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_start_user_config_without_sidecar(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Sidecar skipped when user config has no sidecar section."""
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)

    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    pid_file = tmp_path / "pid"
    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code == 0, result.output
    assert "Sidecar" not in result.output
    assert mock_popen.call_count == 1


@patch("evalhub.cli.server_cmd._wait_for_sidecar_healthy", return_value=False)
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    side_effect=lambda name, _env: f"/usr/bin/{name}",
)
def test_server_start_sidecar_fails_no_server(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_sidecar_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Server NOT started when sidecar fails health check."""
    _seed_profile(config_file)
    _setup_server_config_with_sidecar(tmp_path, config_file)

    mock_proc = MagicMock()
    mock_proc.pid = 54321
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    pid_file = tmp_path / "pid"
    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code != 0
    assert "Sidecar did not become healthy" in result.output
    assert mock_popen.call_count == 1
    mock_proc.terminate.assert_called_once()


@patch("evalhub.cli.server_cmd._wait_for_sidecar_healthy", return_value=True)
@patch("evalhub.cli.server_cmd._wait_for_healthy", return_value=False)
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    side_effect=lambda name, _env: f"/usr/bin/{name}",
)
def test_server_start_server_fails_sidecar_cleanup(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_healthy: MagicMock,
    mock_sidecar_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Sidecar cleaned up when server fails to start."""
    _seed_profile(config_file)
    _setup_server_config_with_sidecar(tmp_path, config_file)

    mock_proc_sidecar = MagicMock()
    mock_proc_sidecar.pid = 54321

    mock_proc_server = MagicMock()
    mock_proc_server.pid = 12345
    mock_proc_server.poll.return_value = 1
    mock_proc_server.returncode = 1

    mock_popen.side_effect = [mock_proc_sidecar, mock_proc_server]

    sidecar_pid_file = tmp_path / "sidecar_state" / "pid"

    pid_file = tmp_path / "pid"
    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=False
    ):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code != 0
    assert "crashed on startup" in result.output
    assert not sidecar_pid_file.exists()


@pytest.mark.parametrize("subcmd", ["start", "run"])
def test_server_old_version_error(
    subcmd: str,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Error raised when eval-hub-server version is too old."""
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)

    pid_file = tmp_path / "pid"
    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli.server_cmd._require_server_version",
        side_effect=_real_require_version,
    ), patch("importlib.metadata.version", return_value="1.0.0"):
        result = runner.invoke(main, ["server", subcmd])

    assert result.exit_code != 0
    assert "1.0.1" in result.output
    assert "Upgrade" in result.output


# ---------------------------------------------------------------------------
# server stop — sidecar lifecycle
# ---------------------------------------------------------------------------


@patch("evalhub.cli._process.os.kill")
def test_server_stop_both_processes(
    mock_kill: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Both server and sidecar are stopped."""
    _seed_profile(config_file)
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")

    sidecar_pid_file = tmp_path / "sidecar_state" / "pid"
    sidecar_pid_file.parent.mkdir(parents=True, exist_ok=True)
    sidecar_pid_file.write_text("54321")

    alive_calls = iter([True, False, True, False])

    with patch("evalhub.cli.server_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli._process.is_process_alive", side_effect=alive_calls
    ), patch("evalhub.cli._process.time.sleep"):
        result = runner.invoke(main, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert "Server stopped" in result.output
    assert "Sidecar stopped" in result.output
    assert not pid_file.exists()
    assert not sidecar_pid_file.exists()


# ---------------------------------------------------------------------------
# server run — sidecar lifecycle
# ---------------------------------------------------------------------------


@patch("evalhub.cli.server_cmd._wait_for_sidecar_healthy", return_value=True)
@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli._process.subprocess.Popen")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    side_effect=lambda name, _env: f"/usr/bin/{name}",
)
def test_server_run_with_sidecar_cleanup(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_run: MagicMock,
    mock_sidecar_healthy: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Sidecar started in background and cleaned up when server exits."""
    _seed_profile(config_file)
    _setup_server_config_with_sidecar(tmp_path, config_file)

    mock_proc = MagicMock()
    mock_proc.pid = 54321
    mock_popen.return_value = mock_proc

    mock_run.return_value = MagicMock(returncode=0)

    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli._process.is_process_alive", return_value=False
    ):
        result = runner.invoke(main, ["server", "run"])

    assert result.exit_code == 0, result.output
    assert "Sidecar started" in result.output
    assert "54321" in result.output

    sidecar_cmd = mock_popen.call_args[0][0]
    assert sidecar_cmd[0] == "/usr/bin/eval-runtime-sidecar"
    assert "-sidecarconfig" in sidecar_cmd


@patch("evalhub.cli._process.subprocess.run")
@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_run_user_config_without_sidecar(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Sidecar skipped when user config has no sidecar section."""
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)
    mock_run.return_value = MagicMock(returncode=0)

    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path):
        result = runner.invoke(main, ["server", "run"])

    assert result.exit_code == 0, result.output
    assert "Sidecar" not in result.output
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# server status — sidecar reporting
# ---------------------------------------------------------------------------


@patch("evalhub.cli.server_cmd._sidecar_health_check", return_value=True)
def test_server_status_with_sidecar(
    mock_sidecar_health: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Status reports both server and sidecar."""
    _seed_profile(config_file)
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")
    _setup_server_config_with_sidecar(tmp_path, config_file)

    sidecar_pid_file = tmp_path / "sidecar_state" / "pid"
    sidecar_pid_file.parent.mkdir(parents=True, exist_ok=True)
    sidecar_pid_file.write_text("54321")

    health_info = {"status": "healthy", "build": "2.0.0"}

    with patch("evalhub.cli.server_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ), patch("evalhub.cli.server_cmd._fetch_health_info", return_value=health_info):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "Server is running (PID 12345)" in result.output
    assert "Sidecar is running" in result.output
    assert "54321" in result.output
    assert "8082" in result.output
    assert "mode: local" in result.output


def test_server_status_sidecar_enabled_not_running(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Status shows sidecar not running when enabled but no PID."""
    _seed_profile(config_file)
    _setup_server_config_with_sidecar(tmp_path, config_file)

    with patch("evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd._fetch_health_info", return_value=None):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "Server is not running" in result.output
    assert "Sidecar is not running" in result.output


def test_server_status_sidecar_not_enabled_hidden(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    """Status hides sidecar info when sidecar is not configured."""
    _seed_profile(config_file)
    _setup_server_config(tmp_path, config_file)

    with patch("evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd._fetch_health_info", return_value=None):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "Server is not running" in result.output
    assert "Sidecar" not in result.output
