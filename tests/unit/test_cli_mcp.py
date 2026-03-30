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


def test_mcp_appears_in_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "mcp" in result.output


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
