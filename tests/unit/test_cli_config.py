"""Unit tests for EvalHub CLI config and profile management."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from evalhub.cli.config import (
    DEFAULT_PROFILE,
    REQUIRED_KEYS,
    get_active_profile,
    get_profile,
    get_value,
    is_known_key,
    load_config,
    missing_required_keys,
    save_config,
    set_active_profile,
    set_value,
)
from evalhub.cli.main import main


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


# --- config module unit tests ---


class TestLoadConfig:
    def test_returns_default_structure_when_file_missing(
        self, config_file: Path
    ) -> None:
        data = load_config()
        assert data["active_profile"] == DEFAULT_PROFILE
        assert data["profiles"] == {}

    def test_loads_existing_config(self, config_file: Path) -> None:
        config_file.write_text(
            yaml.safe_dump(
                {
                    "active_profile": "prod",
                    "profiles": {"prod": {"base_url": "https://example.com"}},
                }
            )
        )
        data = load_config()
        assert data["active_profile"] == "prod"
        assert data["profiles"]["prod"]["base_url"] == "https://example.com"

    def test_fills_missing_keys(self, config_file: Path) -> None:
        config_file.write_text(yaml.safe_dump({}))
        data = load_config()
        assert data["active_profile"] == DEFAULT_PROFILE
        assert data["profiles"] == {}


class TestSaveConfig:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "config.yaml"
        save_config({"active_profile": "default", "profiles": {}}, path=path)
        assert path.exists()

    def test_sets_safe_permissions(self, config_file: Path) -> None:
        save_config({"active_profile": "default", "profiles": {}})
        mode = config_file.stat().st_mode
        assert mode & stat.S_IRUSR
        assert mode & stat.S_IWUSR
        assert not (mode & stat.S_IRGRP)
        assert not (mode & stat.S_IROTH)

    def test_roundtrip(self, config_file: Path) -> None:
        original = {
            "active_profile": "dev",
            "profiles": {"dev": {"base_url": "http://localhost:9090", "token": "abc"}},
        }
        save_config(original)
        loaded = load_config()
        assert loaded == original


class TestProfileOperations:
    def test_get_active_profile_default(self) -> None:
        assert get_active_profile({}) == DEFAULT_PROFILE

    def test_get_active_profile_custom(self) -> None:
        assert get_active_profile({"active_profile": "staging"}) == "staging"

    def test_set_active_profile(self) -> None:
        data = {"active_profile": "default", "profiles": {}}
        set_active_profile(data, "prod")
        assert data["active_profile"] == "prod"

    def test_get_profile_missing(self) -> None:
        data = {"active_profile": "default", "profiles": {}}
        assert get_profile(data) == {}

    def test_get_profile_existing(self) -> None:
        data = {
            "active_profile": "default",
            "profiles": {"default": {"base_url": "http://localhost:8080"}},
        }
        assert get_profile(data) == {"base_url": "http://localhost:8080"}

    def test_get_profile_explicit_name(self) -> None:
        data = {
            "active_profile": "default",
            "profiles": {"prod": {"base_url": "https://prod.example.com"}},
        }
        assert get_profile(data, "prod") == {"base_url": "https://prod.example.com"}

    def test_set_value_creates_profile(self) -> None:
        data: dict[str, Any] = {"active_profile": "default", "profiles": {}}
        set_value(data, "base_url", "http://localhost:8080")
        assert data["profiles"]["default"]["base_url"] == "http://localhost:8080"

    def test_set_value_explicit_profile(self) -> None:
        data: dict[str, Any] = {"active_profile": "default", "profiles": {}}
        set_value(data, "token", "secret", profile="staging")
        assert data["profiles"]["staging"]["token"] == "secret"

    def test_get_value_existing(self) -> None:
        data = {
            "active_profile": "default",
            "profiles": {"default": {"base_url": "http://localhost:8080"}},
        }
        assert get_value(data, "base_url") == "http://localhost:8080"

    def test_get_value_missing(self) -> None:
        data = {"active_profile": "default", "profiles": {}}
        assert get_value(data, "nonexistent") is None


class TestRequiredKeys:
    def test_all_missing_on_empty_profile(self) -> None:
        data = {"active_profile": "default", "profiles": {}}
        missing = missing_required_keys(data)
        assert set(missing) == set(REQUIRED_KEYS)

    def test_none_missing_when_all_set(self) -> None:
        data = {
            "active_profile": "default",
            "profiles": {
                "default": {
                    "base_url": "http://localhost",
                    "token": "t",
                    "tenant": "ns",
                }
            },
        }
        assert missing_required_keys(data) == []

    def test_partial_missing(self) -> None:
        data = {
            "active_profile": "default",
            "profiles": {"default": {"base_url": "http://localhost"}},
        }
        missing = missing_required_keys(data)
        assert "token" in missing
        assert "tenant" in missing
        assert "base_url" not in missing

    def test_is_known_key(self) -> None:
        assert is_known_key("base_url")
        assert is_known_key("token")
        assert is_known_key("tenant")
        assert is_known_key("timeout")
        assert not is_known_key("foobar")


# --- CLI integration tests ---


class TestConfigSetCommand:
    def test_set_value(self, runner: CliRunner, config_file: Path) -> None:
        result = runner.invoke(
            main, ["config", "set", "base_url", "http://myhost:8080"]
        )
        assert result.exit_code == 0
        assert "Set 'base_url' in profile 'default'" in result.output
        data = load_config()
        assert data["profiles"]["default"]["base_url"] == "http://myhost:8080"

    def test_set_with_profile_flag(self, runner: CliRunner, config_file: Path) -> None:
        result = runner.invoke(
            main, ["--profile", "staging", "config", "set", "token", "mytoken"]
        )
        assert result.exit_code == 0
        assert "profile 'staging'" in result.output
        data = load_config()
        assert data["profiles"]["staging"]["token"] == "mytoken"

    def test_set_unknown_key_warns(self, runner: CliRunner, config_file: Path) -> None:
        result = runner.invoke(main, ["config", "set", "foobar", "baz"])
        assert result.exit_code == 0
        assert "not a recognised config key" in result.output
        # Value is still set despite the warning
        data = load_config()
        assert data["profiles"]["default"]["foobar"] == "baz"

    def test_set_known_key_no_warning(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        result = runner.invoke(main, ["config", "set", "base_url", "http://host:8080"])
        assert result.exit_code == 0
        assert "not a recognised" not in result.output


class TestConfigGetCommand:
    def test_get_existing_value(self, runner: CliRunner, config_file: Path) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://myhost:8080"])
        result = runner.invoke(main, ["config", "get", "base_url"])
        assert result.exit_code == 0
        assert result.output.strip() == "http://myhost:8080"

    def test_get_missing_key(self, runner: CliRunner, config_file: Path) -> None:
        result = runner.invoke(main, ["config", "get", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_get_with_profile_flag(self, runner: CliRunner, config_file: Path) -> None:
        runner.invoke(
            main,
            [
                "--profile",
                "prod",
                "config",
                "set",
                "base_url",
                "https://prod.example.com",
            ],
        )
        result = runner.invoke(main, ["--profile", "prod", "config", "get", "base_url"])
        assert result.exit_code == 0
        assert result.output.strip() == "https://prod.example.com"


class TestConfigListCommand:
    def test_list_empty_profile(self, runner: CliRunner, config_file: Path) -> None:
        result = runner.invoke(main, ["config", "list"])
        assert result.exit_code == 0
        assert "no configuration values" in result.output
        assert "Missing required keys:" in result.output
        assert "base_url" in result.output
        assert "token" in result.output
        assert "tenant" in result.output

    def test_list_populated_profile(self, runner: CliRunner, config_file: Path) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://myhost:8080"])
        runner.invoke(main, ["config", "set", "token", "abc123"])
        result = runner.invoke(main, ["config", "list"])
        assert result.exit_code == 0
        assert "base_url: http://myhost:8080" in result.output
        assert "token: abc123" in result.output
        assert "Profile: default" in result.output
        assert "Missing required keys:" in result.output
        assert "tenant" in result.output

    def test_list_complete_profile_no_missing(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://myhost:8080"])
        runner.invoke(main, ["config", "set", "token", "abc123"])
        runner.invoke(main, ["config", "set", "tenant", "my-namespace"])
        result = runner.invoke(main, ["config", "list"])
        assert result.exit_code == 0
        assert "Missing required keys" not in result.output

    def test_list_localhost_profile_empty_tenant_no_missing(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://localhost:8080"])
        runner.invoke(main, ["config", "set", "token", "dev-token"])
        runner.invoke(main, ["config", "set", "tenant", ""])
        result = runner.invoke(main, ["config", "list"])
        assert result.exit_code == 0
        assert "Missing required keys" not in result.output


class TestConfigUseCommand:
    def test_use_switches_active_profile(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        # Create the profile first
        runner.invoke(
            main, ["--profile", "prod", "config", "set", "base_url", "https://prod:443"]
        )
        result = runner.invoke(main, ["config", "use", "prod"])
        assert result.exit_code == 0
        assert "Active profile set to 'prod'" in result.output
        data = load_config()
        assert data["active_profile"] == "prod"

    def test_use_nonexistent_profile_errors(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        result = runner.invoke(main, ["config", "use", "nonexistent"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_use_then_set_uses_new_profile(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        # Create the profile first, then switch to it
        runner.invoke(
            main,
            [
                "--profile",
                "staging",
                "config",
                "set",
                "base_url",
                "https://staging.example.com",
            ],
        )
        runner.invoke(main, ["config", "use", "staging"])
        # Update a value in the now-active profile
        runner.invoke(main, ["config", "set", "token", "abc"])
        data = load_config()
        assert data["profiles"]["staging"]["base_url"] == "https://staging.example.com"
        assert data["profiles"]["staging"]["token"] == "abc"


class TestProfileOverride:
    def test_profile_flag_overrides_active(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        # Set in default profile
        runner.invoke(main, ["config", "set", "base_url", "http://default:8080"])
        # Set in prod profile
        runner.invoke(
            main, ["--profile", "prod", "config", "set", "base_url", "https://prod:443"]
        )
        # Get from prod via --profile
        result = runner.invoke(main, ["--profile", "prod", "config", "get", "base_url"])
        assert result.output.strip() == "https://prod:443"
        # Get from default (no --profile)
        result = runner.invoke(main, ["config", "get", "base_url"])
        assert result.output.strip() == "http://default:8080"

    def test_profile_env_var(self, runner: CliRunner, config_file: Path) -> None:
        runner.invoke(
            main,
            ["--profile", "env-test", "config", "set", "base_url", "http://env:8080"],
        )
        result = runner.invoke(
            main, ["config", "get", "base_url"], env={"EVALHUB_PROFILE": "env-test"}
        )
        assert result.output.strip() == "http://env:8080"


class TestFilePermissions:
    def test_config_file_not_world_readable(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "token", "secret"])
        mode = config_file.stat().st_mode
        # Should be 0600 (owner read/write only)
        assert not (mode & stat.S_IRGRP), "Group read should be off"
        assert not (mode & stat.S_IWGRP), "Group write should be off"
        assert not (mode & stat.S_IROTH), "Other read should be off"
        assert not (mode & stat.S_IWOTH), "Other write should be off"
