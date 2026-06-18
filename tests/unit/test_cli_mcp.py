"""Unit tests for the EvalHub CLI mcp subcommand."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from evalhub.cli.config import load_config, save_config, set_value
from evalhub.cli.main import main


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


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Python MCP server path (evalhub mcp, no subcommand)
# ---------------------------------------------------------------------------


def test_mcp_appears_in_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "mcp" in result.output


def test_mcp_subcommands_appear_in_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["mcp", "--help"])
    assert result.exit_code == 0
    for sub in ("run", "start", "stop", "status"):
        assert sub in result.output


def test_mcp_missing_package(runner: CliRunner, config_file: Path) -> None:
    with patch.dict("sys.modules", {"mcp": None}):
        result = runner.invoke(main, ["mcp"])
    assert result.exit_code != 0
    assert "pip install" in result.output
    assert "eval-hub-sdk[mcp]" in result.output


@patch("asyncio.run")
@patch("evalhub.mcp.server.set_client")
@patch("evalhub.client.evalhub.AsyncEvalHubClient")
def test_mcp_resolves_from_profile(
    mock_client_cls: MagicMock,
    mock_set_client: MagicMock,
    mock_asyncio_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
) -> None:
    data = load_config()
    set_value(data, "base_url", "https://evalhub.example.com")
    set_value(data, "token", "profile-token")
    set_value(data, "tenant", "my-namespace")
    save_config(data)

    mock_client_cls.return_value = MagicMock()

    result = runner.invoke(main, ["mcp"])
    assert result.exit_code == 0, result.output

    mock_client_cls.assert_called_once_with(
        base_url="https://evalhub.example.com",
        auth_token="profile-token",
        tenant="my-namespace",
        insecure=False,
        timeout=30.0,
    )
    mock_set_client.assert_called_once()


@patch("asyncio.run")
@patch("evalhub.mcp.server.set_client")
@patch("evalhub.client.evalhub.AsyncEvalHubClient")
def test_mcp_cli_flags_override_profile(
    mock_client_cls: MagicMock,
    mock_set_client: MagicMock,
    mock_asyncio_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
) -> None:
    data = load_config()
    set_value(data, "base_url", "https://profile-url.example.com")
    set_value(data, "token", "profile-token")
    set_value(data, "tenant", "profile-ns")
    save_config(data)

    mock_client_cls.return_value = MagicMock()

    result = runner.invoke(
        main,
        ["--base-url", "https://flag-url.example.com", "--token", "flag-token", "mcp"],
    )
    assert result.exit_code == 0, result.output

    mock_client_cls.assert_called_once_with(
        base_url="https://flag-url.example.com",
        auth_token="flag-token",
        tenant="profile-ns",
        insecure=False,
        timeout=30.0,
    )
    mock_set_client.assert_called_once()


@patch("asyncio.run")
@patch("evalhub.mcp.server.set_client")
@patch("evalhub.client.evalhub.AsyncEvalHubClient")
def test_mcp_no_subcommand_with_tenant(
    mock_client_cls: MagicMock,
    mock_set_client: MagicMock,
    mock_asyncio_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
) -> None:
    data = load_config()
    set_value(data, "base_url", "https://evalhub.example.com")
    set_value(data, "token", "t")
    set_value(data, "tenant", "profile-ns")
    save_config(data)

    mock_client_cls.return_value = MagicMock()

    result = runner.invoke(main, ["mcp", "--tenant", "cli-ns"])
    assert result.exit_code == 0, result.output

    mock_client_cls.assert_called_once()
    call_kwargs = mock_client_cls.call_args[1]
    assert call_kwargs["tenant"] == "cli-ns"


# ---------------------------------------------------------------------------
# Go binary subcommands
# ---------------------------------------------------------------------------


@patch("evalhub.cli.mcp_cmd.subprocess.run")
@patch("evalhub.cli.mcp_cmd._find_mcp_binary", return_value="/usr/bin/evalhub-mcp")
def test_mcp_run_stdio(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
) -> None:
    mock_run.return_value = MagicMock(returncode=0)

    result = runner.invoke(main, ["mcp", "run"])
    assert result.exit_code == 0, result.output

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/evalhub-mcp"


@patch("evalhub.cli.mcp_cmd.subprocess.run")
@patch("evalhub.cli.mcp_cmd._find_mcp_binary", return_value="/usr/bin/evalhub-mcp")
def test_mcp_run_with_passthrough_args(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
) -> None:
    mock_run.return_value = MagicMock(returncode=0)

    result = runner.invoke(
        main, ["mcp", "run", "--", "--transport", "http", "--port", "8888"]
    )
    assert result.exit_code == 0, result.output

    cmd = mock_run.call_args[0][0]
    assert cmd == ["/usr/bin/evalhub-mcp", "--transport", "http", "--port", "8888"]


@patch("evalhub.cli.mcp_cmd._find_mcp_binary")
def test_mcp_run_binary_not_found(
    mock_find: MagicMock,
    runner: CliRunner,
    config_file: Path,
) -> None:
    from click import ClickException

    mock_find.side_effect = ClickException(
        "Could not find the 'evalhub-mcp' binary.\n"
        "Install it and ensure it is on your PATH, or set EVALHUB_MCP_BIN."
    )

    result = runner.invoke(main, ["mcp", "run"])
    assert result.exit_code != 0
    assert "evalhub-mcp" in result.output


@patch("evalhub.cli.mcp_cmd.time.sleep")
@patch("evalhub.cli.mcp_cmd.subprocess.Popen")
@patch("evalhub.cli.mcp_cmd._find_mcp_binary", return_value="/usr/bin/evalhub-mcp")
def test_mcp_start_launches_background(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_sleep: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    with patch("evalhub.cli.mcp_cmd.MCP_STATE_DIR", tmp_path), patch(
        "evalhub.cli.mcp_cmd.PID_FILE", tmp_path / "pid"
    ), patch("evalhub.cli.mcp_cmd.LOG_FILE", tmp_path / "log"):
        result = runner.invoke(main, ["mcp", "start"])

    assert result.exit_code == 0, result.output
    assert "12345" in result.output

    cmd = mock_popen.call_args[0][0]
    assert "--transport" in cmd
    assert "http" in cmd

    pid_content = (tmp_path / "pid").read_text().strip()
    assert pid_content == "12345"


@patch("evalhub.cli.mcp_cmd.time.sleep")
@patch("evalhub.cli.mcp_cmd.subprocess.Popen")
@patch("evalhub.cli.mcp_cmd._find_mcp_binary", return_value="/usr/bin/evalhub-mcp")
def test_mcp_start_already_running(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_sleep: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    pid_file = tmp_path / "pid"
    pid_file.write_text("99999")

    with patch("evalhub.cli.mcp_cmd.MCP_STATE_DIR", tmp_path), patch(
        "evalhub.cli.mcp_cmd.PID_FILE", pid_file
    ), patch("evalhub.cli.mcp_cmd.LOG_FILE", tmp_path / "log"), patch(
        "evalhub.cli.mcp_cmd._is_process_alive", return_value=True
    ):
        result = runner.invoke(main, ["mcp", "start"])

    assert result.exit_code != 0
    assert "already running" in result.output
    assert "evalhub mcp stop" in result.output
    mock_popen.assert_not_called()


@patch("evalhub.cli.mcp_cmd.os.kill")
def test_mcp_stop(
    mock_kill: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")

    alive_calls = iter([True, False])

    with patch("evalhub.cli.mcp_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.mcp_cmd._is_process_alive", side_effect=alive_calls
    ), patch("evalhub.cli.mcp_cmd.time.sleep"):
        result = runner.invoke(main, ["mcp", "stop"])

    assert result.exit_code == 0, result.output
    assert "stopped" in result.output
    assert not pid_file.exists()


def test_mcp_status_not_running(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    with patch("evalhub.cli.mcp_cmd.PID_FILE", tmp_path / "pid"):
        result = runner.invoke(main, ["mcp", "status"])

    assert result.exit_code == 0, result.output
    assert "not running" in result.output


def test_mcp_status_running(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")

    with patch("evalhub.cli.mcp_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.mcp_cmd._is_process_alive", return_value=True
    ):
        result = runner.invoke(main, ["mcp", "status"])

    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "12345" in result.output
