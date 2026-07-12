"""Unit tests for the EvalHub CLI mcp subcommand."""

from __future__ import annotations

import json
import os
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from evalhub.cli.main import main
from evalhub.cli.mcp_cmd import GENERATED_CONFIG, _fetch_server_info

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


def _seed_profile(config_file: Path, profile: str = "default", **kwargs: str) -> None:
    """Write a profile into the config file."""
    data: dict[str, object] = {"active_profile": profile, "profiles": {profile: kwargs}}
    config_file.write_text(yaml.safe_dump(data))


def _write_mcp_config(cfg_dir: Path, **overrides: object) -> Path:
    """Write an MCP config.yaml into *cfg_dir* and return the directory."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "transport": "http",
        "host": "localhost",
        "port": 3001,
    }
    data.update(overrides)
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(data))
    return cfg_dir


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_mcp_appears_in_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "mcp" in result.output


def test_mcp_subcommands_appear_in_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["mcp", "--help"])
    assert result.exit_code == 0
    for sub in ("run", "start", "stop", "status"):
        assert sub in result.output


# ---------------------------------------------------------------------------
# No MCP config.yaml — generated from root profile only
# ---------------------------------------------------------------------------


@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_mcp_run_no_mcp_config_uses_root_profile(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    """When no MCP config.yaml exists, mcp-config.yaml is generated from root profile."""
    _seed_profile(
        config_file,
        base_url="https://prod.example.com",
        token="prod-token",
        tenant="team-a",
    )
    cfg_dir = tmp_path / "mcp" / "default"
    mock_run.return_value = MagicMock(returncode=0)

    with patch("evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir):
        result = runner.invoke(main, ["mcp", "run"])
    assert result.exit_code == 0, result.output

    generated = yaml.safe_load((cfg_dir / GENERATED_CONFIG).read_text())
    assert generated["base_url"] == "https://prod.example.com"
    assert generated["token"] == "prod-token"
    assert generated["tenant"] == "team-a"


@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_mcp_run_no_config_at_all_defaults_to_stdio(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    """When neither MCP config nor root profile keys exist, transport defaults to stdio."""
    cfg_dir = tmp_path / "mcp" / "default"
    mock_run.return_value = MagicMock(returncode=0)

    with patch("evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir):
        result = runner.invoke(main, ["mcp", "run"])
    assert result.exit_code == 0, result.output

    generated = yaml.safe_load((cfg_dir / GENERATED_CONFIG).read_text()) or {}
    assert generated == {"transport": "stdio"}


# ---------------------------------------------------------------------------
# Go binary subcommands
# ---------------------------------------------------------------------------


@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_mcp_run(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    cfg_dir = _write_mcp_config(tmp_path / "mcp" / "default", transport="stdio")

    with patch("evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir):
        result = runner.invoke(main, ["mcp", "run"])
    assert result.exit_code == 0, result.output

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/evalhub-mcp"
    assert "--config" in cmd
    assert str(cfg_dir / GENERATED_CONFIG) in cmd


@patch("evalhub.cli.mcp_cmd.find_binary")
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
@patch("evalhub.cli._process.subprocess.Popen")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
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

    cfg_dir = _write_mcp_config(
        tmp_path / "mcp" / "default", transport="http", host="localhost", port=3001
    )

    with patch(
        "evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir
    ), patch("evalhub.cli.mcp_cmd.MCP_STATE_DIR", tmp_path), patch(
        "evalhub.cli.mcp_cmd.PID_FILE", tmp_path / "pid"
    ), patch("evalhub.cli.mcp_cmd.LOG_FILE", tmp_path / "mcp.log"):
        result = runner.invoke(main, ["mcp", "start"])

    assert result.exit_code == 0, result.output
    assert "12345" in result.output
    assert "Transport: http" in result.output
    assert "http://localhost:3001" in result.output

    cmd = mock_popen.call_args[0][0]
    assert "--config" in cmd
    assert str(cfg_dir / GENERATED_CONFIG) in cmd

    pid_content = (tmp_path / "pid").read_text().strip()
    assert pid_content == "12345"


@patch("evalhub.cli.mcp_cmd.time.sleep")
@patch("evalhub.cli._process.subprocess.Popen")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
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
    ), patch("evalhub.cli.mcp_cmd.LOG_FILE", tmp_path / "mcp.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ):
        result = runner.invoke(main, ["mcp", "start"])

    assert result.exit_code != 0
    assert "already running" in result.output
    assert "evalhub mcp stop" in result.output
    mock_popen.assert_not_called()


@patch("evalhub.cli.mcp_cmd.time.sleep")
@patch("evalhub.cli._process.subprocess.Popen")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_mcp_start_defaults_to_http_when_no_transport(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_sleep: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    """When no transport is configured, mcp start defaults to http."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    cfg_dir = _write_mcp_config(
        tmp_path / "mcp" / "default", host="localhost", port=3001
    )
    mcp_cfg = cfg_dir / "config.yaml"
    data = yaml.safe_load(mcp_cfg.read_text())
    del data["transport"]
    mcp_cfg.write_text(yaml.safe_dump(data))

    with patch(
        "evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir
    ), patch("evalhub.cli.mcp_cmd.MCP_STATE_DIR", tmp_path), patch(
        "evalhub.cli.mcp_cmd.PID_FILE", tmp_path / "pid"
    ), patch("evalhub.cli.mcp_cmd.LOG_FILE", tmp_path / "mcp.log"):
        result = runner.invoke(main, ["mcp", "start"])

    assert result.exit_code == 0, result.output
    assert "Transport: http" in result.output

    generated = yaml.safe_load((cfg_dir / GENERATED_CONFIG).read_text())
    assert generated["transport"] == "http"


@patch("evalhub.cli.mcp_cmd.time.sleep")
@patch("evalhub.cli._process.subprocess.Popen")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_mcp_start_rejects_stdio_transport(
    mock_find: MagicMock,
    mock_popen: MagicMock,
    mock_sleep: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    cfg_dir = _write_mcp_config(tmp_path / "mcp" / "default", transport="stdio")

    with patch(
        "evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir
    ), patch("evalhub.cli.mcp_cmd.MCP_STATE_DIR", tmp_path), patch(
        "evalhub.cli.mcp_cmd.PID_FILE", tmp_path / "pid"
    ), patch("evalhub.cli.mcp_cmd.LOG_FILE", tmp_path / "mcp.log"):
        result = runner.invoke(main, ["mcp", "start"])

    assert result.exit_code != 0
    assert "stdio" in result.output
    assert "evalhub mcp run" in result.output
    mock_popen.assert_not_called()


@patch("evalhub.cli._process.os.kill")
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
        "evalhub.cli._process.is_process_alive", side_effect=alive_calls
    ), patch("evalhub.cli._process.time.sleep"):
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


def test_mcp_status_running_with_server_info(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")

    cfg_dir = _write_mcp_config(
        tmp_path / "mcp" / "default", host="localhost", port=3001
    )

    server_info = {"name": "evalhub-mcp", "version": "1.2.3"}

    with patch("evalhub.cli.mcp_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir
    ), patch("evalhub.cli.mcp_cmd.LOG_FILE", tmp_path / "mcp.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ), patch("evalhub.cli.mcp_cmd._fetch_server_info", return_value=server_info):
        result = runner.invoke(main, ["mcp", "status"])

    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "12345" in result.output
    assert "evalhub-mcp" in result.output
    assert "1.2.3" in result.output
    assert "http://localhost:3001" in result.output


def test_mcp_status_running_server_info_unavailable(
    runner: CliRunner,
    tmp_path: Path,
    config_file: Path,
) -> None:
    pid_file = tmp_path / "pid"
    pid_file.write_text("12345")

    cfg_dir = _write_mcp_config(
        tmp_path / "mcp" / "default", host="127.0.0.1", port=4000
    )
    (cfg_dir / "mcp-config.yaml").write_text(
        yaml.safe_dump({"host": "127.0.0.1", "port": 4000})
    )

    with patch("evalhub.cli.mcp_cmd.PID_FILE", pid_file), patch(
        "evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir
    ), patch("evalhub.cli.mcp_cmd.LOG_FILE", tmp_path / "mcp.log"), patch(
        "evalhub.cli._process.is_process_alive", return_value=True
    ), patch("evalhub.cli.mcp_cmd._fetch_server_info", return_value=None):
        result = runner.invoke(main, ["mcp", "status"])

    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "12345" in result.output
    assert "Name:" not in result.output
    assert "http://127.0.0.1:4000" in result.output


# ---------------------------------------------------------------------------
# Merge behavior — root profile fallback + MCP override
# ---------------------------------------------------------------------------


@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_merge_inherits_connection_keys_from_profile(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    """Connection keys absent from MCP config are sourced from root profile."""
    _seed_profile(
        config_file,
        base_url="https://prod.example.com",
        token="prod-token",
        tenant="team-a",
        insecure="true",
    )
    cfg_dir = _write_mcp_config(
        tmp_path / "mcp" / "default", transport="stdio", host="localhost", port=3001
    )
    mock_run.return_value = MagicMock(returncode=0)

    with patch("evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir):
        result = runner.invoke(main, ["mcp", "run"])
    assert result.exit_code == 0, result.output

    generated = yaml.safe_load((cfg_dir / GENERATED_CONFIG).read_text())
    assert generated["base_url"] == "https://prod.example.com"
    assert generated["token"] == "prod-token"
    assert generated["tenant"] == "team-a"
    assert generated["insecure"] is True
    assert generated["transport"] == "stdio"
    assert generated["host"] == "localhost"
    assert generated["port"] == 3001


@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_merge_mcp_config_overrides_profile(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    """MCP config keys take precedence over root profile keys."""
    _seed_profile(
        config_file,
        base_url="https://prod.example.com",
        token="prod-token",
        tenant="team-a",
    )
    cfg_dir = _write_mcp_config(
        tmp_path / "mcp" / "default",
        base_url="http://localhost:8080",
        token="dev-token",
        transport="stdio",
    )
    mock_run.return_value = MagicMock(returncode=0)

    with patch("evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir):
        result = runner.invoke(main, ["mcp", "run"])
    assert result.exit_code == 0, result.output

    generated = yaml.safe_load((cfg_dir / GENERATED_CONFIG).read_text())
    assert generated["base_url"] == "http://localhost:8080"
    assert generated["token"] == "dev-token"
    assert generated["tenant"] == "team-a"


@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_merge_respects_profile_flag(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    """--profile flag selects which root profile to merge from."""
    data = {
        "active_profile": "default",
        "profiles": {
            "default": {
                "base_url": "http://localhost:8080",
                "token": "default-tok",
                "tenant": "default-t",
            },
            "prod": {
                "base_url": "https://prod.example.com",
                "token": "prod-tok",
                "tenant": "prod-t",
            },
        },
    }
    config_file.write_text(yaml.safe_dump(data))
    cfg_dir = _write_mcp_config(tmp_path / "mcp" / "prod", transport="stdio")
    mock_run.return_value = MagicMock(returncode=0)

    with patch("evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir):
        result = runner.invoke(main, ["--profile", "prod", "mcp", "run"])
    assert result.exit_code == 0, result.output

    generated = yaml.safe_load((cfg_dir / GENERATED_CONFIG).read_text())
    assert generated["base_url"] == "https://prod.example.com"
    assert generated["token"] == "prod-tok"
    assert generated["tenant"] == "prod-t"


# ---------------------------------------------------------------------------
# Merge error handling
# ---------------------------------------------------------------------------


@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_merge_rejects_malformed_mcp_config(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    """A MCP config.yaml with invalid YAML produces a clear error."""
    cfg_dir = tmp_path / "mcp" / "default"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(": invalid: yaml: [")

    with patch("evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir):
        result = runner.invoke(main, ["mcp", "run"])

    assert result.exit_code != 0
    assert "Failed to parse MCP config" in result.output


@patch("evalhub.cli._process.subprocess.run")
@patch("evalhub.cli.mcp_cmd.find_binary", return_value="/usr/bin/evalhub-mcp")
def test_merge_rejects_invalid_port(
    mock_find: MagicMock,
    mock_run: MagicMock,
    runner: CliRunner,
    config_file: Path,
    tmp_path: Path,
) -> None:
    """A non-numeric port in MCP config produces a clear error."""
    cfg_dir = _write_mcp_config(tmp_path / "mcp" / "default", port="not-a-number")

    with patch("evalhub.cli.config.resolve_component_config_dir", return_value=cfg_dir):
        result = runner.invoke(main, ["mcp", "run"])

    assert result.exit_code != 0
    assert "Invalid port value" in result.output


# ---------------------------------------------------------------------------
# _fetch_server_info tests
# ---------------------------------------------------------------------------


def _mock_urlopen_response(body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _initialize_result(version: str = "1.0.0") -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-03-26",
            "serverInfo": {"name": "evalhub-mcp", "version": version},
            "capabilities": {},
        },
    }


def test_fetch_server_info_success_sse() -> None:
    sse_body = (
        "event: message\ndata: " + json.dumps(_initialize_result("1.0.0")) + "\n"
    ).encode()

    with patch(
        "evalhub.cli.mcp_cmd.urllib.request.urlopen",
        return_value=_mock_urlopen_response(sse_body),
    ):
        info = _fetch_server_info("localhost", 3001)

    assert info == {"name": "evalhub-mcp", "version": "1.0.0"}


def test_fetch_server_info_success_plain_json() -> None:
    plain_body = json.dumps(_initialize_result("2.0.0")).encode()

    with patch(
        "evalhub.cli.mcp_cmd.urllib.request.urlopen",
        return_value=_mock_urlopen_response(plain_body),
    ):
        info = _fetch_server_info("localhost", 3001)

    assert info == {"name": "evalhub-mcp", "version": "2.0.0"}


def test_fetch_server_info_connection_refused() -> None:
    with patch(
        "evalhub.cli.mcp_cmd.urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        assert _fetch_server_info("localhost", 9999) is None


def test_fetch_server_info_bad_json() -> None:
    with patch(
        "evalhub.cli.mcp_cmd.urllib.request.urlopen",
        return_value=_mock_urlopen_response(b"not json"),
    ):
        assert _fetch_server_info() is None
