"""Unit tests for the EvalHub CLI server subcommand."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from evalhub.cli.main import main

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


# ---------------------------------------------------------------------------
# Shared helpers for lifecycle tests
# ---------------------------------------------------------------------------


def _patch_server_state(tmp_path: Path) -> tuple[Any, Any, Any]:
    """Context manager that patches all server state dir paths to tmp_path."""
    return (
        patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path),
        patch("evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"),
        patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"),
    )


def _apply_patches(*patches: Any) -> list[Any]:
    """Enter multiple patches; return list to exit later. Not a context manager."""
    started = []
    for p in patches:
        p.start()
        started.append(p)
    return started


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


@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_run_no_config_errors(
    mock_find: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)

    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path):
        result = runner.invoke(main, ["server", "run"])

    assert result.exit_code != 0
    assert "No server config found" in result.output


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


@patch(
    "evalhub.cli.server_cmd.find_binary",
    return_value="/usr/bin/eval-hub-server",
)
def test_server_start_no_config_errors(
    mock_find: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    _seed_profile(config_file)

    with patch("evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path), patch(
        "evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"):
        result = runner.invoke(main, ["server", "start"])

    assert result.exit_code != 0
    assert "No server config found" in result.output


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
    ), patch("evalhub.cli.server_cmd._health_check", return_value=False):
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

    with patch("evalhub.cli.server_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ), patch("evalhub.cli.server_cmd._health_check", return_value=True):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "12345" in result.output
    assert "healthy" in result.output
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

    with patch("evalhub.cli.server_cmd.PID_FILE", tmp_path / "pid"), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd._health_check", return_value=True):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "healthy" in result.output
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

    with patch("evalhub.cli.server_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.server_cmd.SERVER_STATE_DIR", tmp_path
    ), patch("evalhub.cli.server_cmd.LOG_FILE", tmp_path / "server.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ), patch("evalhub.cli.server_cmd._health_check", return_value=True) as mock_hc:
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "https://localhost:8080" in result.output
    assert "http://localhost:8080" not in result.output
    mock_hc.assert_called_once_with(8080, tls=True)


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
    ), patch("evalhub.cli.server_cmd._health_check", return_value=False):
        result = runner.invoke(main, ["server", "status"])

    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "not responding" in result.output


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


def test_config_get_unmask_and_unfold_mutually_exclusive(
    runner: CliRunner,
    config_file: Path,
) -> None:
    _seed_profile(config_file, token="secret")
    result = runner.invoke(main, ["config", "get", "token", "--unmask", "--unfold"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


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
