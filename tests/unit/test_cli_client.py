"""Unit tests for EvalHub CLI client helper and error handling."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import click
import httpx
import pytest
from click.testing import CliRunner
from evalhub.cli.client import create_client, get_client, handle_api_errors
from evalhub.cli.config import (
    load_config,
    missing_required_keys,
    save_config,
    set_value,
)
from evalhub.cli.main import main
from evalhub.client import ClientError, JobNotFoundError


@pytest.fixture()
def config_file(tmp_path: Path) -> Iterator[Path]:
    """Provide a temporary config file path and set EVALHUB_CONFIG."""
    path = tmp_path / "config.yaml"
    os.environ["EVALHUB_CONFIG"] = str(path)
    yield path
    os.environ.pop("EVALHUB_CONFIG", None)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# --- create_client tests ---


class TestCreateClient:
    def test_defaults_when_no_config(self, config_file: Path) -> None:
        client = create_client()
        assert client.base_url == "http://localhost:8080"
        assert client.auth_token is None
        assert client.tenant is None
        client.close()

    def test_localhost_profile_empty_tenant(self, config_file: Path) -> None:
        """A localhost profile sets tenant to empty — no X-Tenant header, no missing-key warning."""
        data = {
            "active_profile": "default",
            "profiles": {
                "default": {
                    "base_url": "http://localhost:8080",
                    "token": "dev-token",
                    "tenant": "",
                }
            },
        }
        save_config(data)

        # config.yaml renders as "tenant: ''\n" which YAML loads back as ""
        raw = config_file.read_text()
        assert "tenant: ''" in raw

        assert missing_required_keys(data) == []

        client = create_client()
        assert client.base_url == "http://localhost:8080"
        assert client.tenant == ""
        client.close()

    def test_reads_from_profile(self, config_file: Path) -> None:
        data = load_config()
        set_value(data, "base_url", "https://evalhub.example.com")
        set_value(data, "token", "profile-token")
        set_value(data, "tenant", "my-namespace")
        save_config(data)

        client = create_client()
        assert client.base_url == "https://evalhub.example.com"
        assert client.auth_token == "profile-token"
        assert client.tenant == "my-namespace"
        client.close()

    def test_explicit_flags_override_profile(self, config_file: Path) -> None:
        data = load_config()
        set_value(data, "base_url", "https://profile-url.example.com")
        set_value(data, "token", "profile-token")
        save_config(data)

        client = create_client(
            base_url="https://flag-url.example.com", token="flag-token"
        )
        assert client.base_url == "https://flag-url.example.com"
        assert client.auth_token == "flag-token"
        client.close()

    def test_named_profile(self, config_file: Path) -> None:
        data = load_config()
        set_value(data, "base_url", "https://prod.example.com", profile="prod")
        set_value(data, "token", "prod-token", profile="prod")
        set_value(data, "tenant", "prod-ns", profile="prod")
        save_config(data)

        client = create_client(profile="prod")
        assert client.base_url == "https://prod.example.com"
        assert client.auth_token == "prod-token"
        assert client.tenant == "prod-ns"
        client.close()

    def test_insecure_flag(self, config_file: Path) -> None:
        data = load_config()
        set_value(data, "insecure", "true")
        save_config(data)

        client = create_client()
        # insecure mode disables TLS verification — check the client was created
        assert client.base_url == "http://localhost:8080"
        client.close()

    def test_custom_timeout(self, config_file: Path) -> None:
        data = load_config()
        set_value(data, "timeout", "60")
        save_config(data)

        client = create_client()
        assert client.base_url == "http://localhost:8080"
        client.close()


# --- get_client tests ---


class TestGetClient:
    def test_creates_client_on_first_access(self, config_file: Path) -> None:
        ctx = click.Context(
            click.Command("test"),
            obj={
                "profile": None,
                "base_url": None,
                "token": None,
            },
        )
        client = get_client(ctx)
        assert client is not None
        assert "client" in ctx.obj
        client.close()

    def test_returns_same_client_on_second_access(self, config_file: Path) -> None:
        ctx = click.Context(
            click.Command("test"),
            obj={
                "profile": None,
                "base_url": None,
                "token": None,
            },
        )
        client1 = get_client(ctx)
        client2 = get_client(ctx)
        assert client1 is client2
        client1.close()

    def test_respects_base_url_override(self, config_file: Path) -> None:
        ctx = click.Context(
            click.Command("test"),
            obj={
                "profile": None,
                "base_url": "https://override.example.com",
                "token": None,
            },
        )
        client = get_client(ctx)
        assert client.base_url == "https://override.example.com"
        client.close()

    def test_respects_token_override(self, config_file: Path) -> None:
        ctx = click.Context(
            click.Command("test"),
            obj={
                "profile": None,
                "base_url": None,
                "token": "override-token",
            },
        )
        client = get_client(ctx)
        assert client.auth_token == "override-token"
        client.close()


# --- handle_api_errors decorator tests ---


class TestHandleApiErrors:
    def test_passes_through_on_success(self) -> None:
        @handle_api_errors
        def ok() -> str:
            return "success"

        assert ok() == "success"

    def test_catches_client_error(self) -> None:
        @handle_api_errors
        def fail() -> None:
            raise ClientError("something went wrong")

        with pytest.raises(click.ClickException, match="something went wrong"):
            fail()

    def test_catches_job_not_found_error(self) -> None:
        @handle_api_errors
        def fail() -> None:
            raise JobNotFoundError("job-123")

        with pytest.raises(click.ClickException, match="job-123"):
            fail()

    def test_catches_http_status_error_with_json_detail(self) -> None:
        response = MagicMock()
        response.status_code = 404
        response.json.return_value = {"detail": "Job not found"}
        response.text = '{"detail": "Job not found"}'
        request = MagicMock()

        @handle_api_errors
        def fail() -> None:
            raise httpx.HTTPStatusError("error", request=request, response=response)

        with pytest.raises(
            click.ClickException, match="Server error \\(404\\): Job not found"
        ):
            fail()

    def test_catches_http_status_error_with_plain_text(self) -> None:
        response = MagicMock()
        response.status_code = 500
        response.json.side_effect = ValueError("not json")
        response.text = "Internal Server Error"
        request = MagicMock()

        @handle_api_errors
        def fail() -> None:
            raise httpx.HTTPStatusError("error", request=request, response=response)

        with pytest.raises(
            click.ClickException, match="Server error \\(500\\): Internal Server Error"
        ):
            fail()

    def test_catches_request_error(self) -> None:
        request = MagicMock()
        request.__str__ = lambda s: "GET http://localhost:8080"  # type: ignore[method-assign,misc,assignment]

        @handle_api_errors
        def fail() -> None:
            raise httpx.ConnectError("Connection refused", request=request)

        with pytest.raises(click.ClickException, match="Connection error"):
            fail()


# --- CLI integration: --base-url and --token flags ---


class TestCliFlags:
    def test_help_shows_base_url_and_token(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert "--base-url" in result.output
        assert "--token" in result.output

    def test_base_url_env_var(self, runner: CliRunner, config_file: Path) -> None:
        """EVALHUB_BASE_URL env var is accepted."""
        result = runner.invoke(
            main, ["version"], env={"EVALHUB_BASE_URL": "http://test:9090"}
        )
        assert result.exit_code == 0

    def test_token_env_var(self, runner: CliRunner, config_file: Path) -> None:
        """EVALHUB_TOKEN env var is accepted."""
        result = runner.invoke(main, ["version"], env={"EVALHUB_TOKEN": "test-token"})
        assert result.exit_code == 0
