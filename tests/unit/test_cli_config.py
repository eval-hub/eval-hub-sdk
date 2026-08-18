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
    FILE_KEYS,
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    SENSITIVE_KEYS,
    _validate_path_within,
    create_profile,
    delete_profile,
    validate_profile_name,
    get_active_profile,
    get_profile,
    get_value,
    is_known_key,
    load_config,
    mask_mapping,
    mask_value,
    missing_required_keys,
    parse_bool,
    save_config,
    set_active_profile,
    set_value,
    unset_value,
)
from evalhub.cli.main import main

pytestmark = pytest.mark.unit


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
        assert "token: ***" in result.output
        assert "token: abc123" not in result.output
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


class TestConfigListAllCommand:
    def test_list_all_no_profiles(self, runner: CliRunner, config_file: Path) -> None:
        result = runner.invoke(main, ["config", "list", "--all"])
        assert result.exit_code == 0
        assert "No profiles configured." in result.output

    def test_list_all_shows_every_profile(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://localhost:8080"])
        runner.invoke(main, ["config", "set", "token", "default-token-val"])
        runner.invoke(
            main, ["--profile", "prod", "config", "set", "base_url", "https://prod:443"]
        )
        runner.invoke(
            main, ["--profile", "prod", "config", "set", "token", "prod-token-val"]
        )
        result = runner.invoke(main, ["config", "list", "--all"])
        assert result.exit_code == 0
        assert "Profile: default *" in result.output
        assert "Profile: prod" in result.output
        assert "base_url: http://localhost:8080" in result.output
        assert "base_url: https://prod:443" in result.output

    def test_list_all_masks_tokens(self, runner: CliRunner, config_file: Path) -> None:
        runner.invoke(main, ["config", "set", "token", "secret-default-token"])
        runner.invoke(
            main, ["--profile", "prod", "config", "set", "token", "secret-prod-token"]
        )
        result = runner.invoke(main, ["config", "list", "--all"])
        assert result.exit_code == 0
        assert "secret-default-token" not in result.output
        assert "secret-prod-token" not in result.output
        assert "token: sec***en" in result.output

    def test_list_all_marks_active_profile(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://localhost:8080"])
        runner.invoke(
            main, ["--profile", "prod", "config", "set", "base_url", "https://prod:443"]
        )
        runner.invoke(main, ["config", "use", "prod"])
        result = runner.invoke(main, ["config", "list", "--all"])
        assert result.exit_code == 0
        assert "Profile: prod *" in result.output
        assert "Profile: default\n" in result.output

    def test_list_all_shows_missing_keys(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://localhost:8080"])
        runner.invoke(main, ["config", "set", "token", "tok12345"])
        runner.invoke(main, ["config", "set", "tenant", "ns"])
        runner.invoke(
            main, ["--profile", "prod", "config", "set", "base_url", "https://prod:443"]
        )
        result = runner.invoke(main, ["config", "list", "--all"])
        assert result.exit_code == 0
        # default is complete
        assert "Missing required keys" in result.output  # prod is incomplete


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


class TestMaskValue:
    def test_long_value_shows_prefix_and_suffix(self) -> None:
        assert mask_value("abcdefghij") == "abc***ij"

    def test_exactly_min_length(self) -> None:
        assert mask_value("12345678") == "123***78"

    def test_short_value_fully_masked(self) -> None:
        assert mask_value("short") == "***"

    def test_empty_string(self) -> None:
        assert mask_value("") == "***"

    def test_single_char(self) -> None:
        assert mask_value("x") == "***"

    def test_sensitive_keys_contains_token(self) -> None:
        assert "token" in SENSITIVE_KEYS


class TestMaskMapping:
    def test_masks_sensitive_keys(self) -> None:
        mapping = {"base_url": "http://localhost", "token": "super-secret-tok"}
        result = mask_mapping(mapping)
        assert result["base_url"] == "http://localhost"
        assert result["token"] == "sup***ok"

    def test_leaves_non_sensitive_keys_unchanged(self) -> None:
        mapping = {"host": "localhost", "port": 3001}
        assert mask_mapping(mapping) == mapping

    def test_empty_mapping(self) -> None:
        assert mask_mapping({}) == {}


class TestConfigMasking:
    def test_config_get_token_masked_by_default(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "token", "my-secret-token-value"])
        result = runner.invoke(main, ["config", "get", "token"])
        assert result.exit_code == 0
        assert "my-secret-token-value" not in result.output
        assert "my-***ue" in result.output

    def test_config_get_token_unmask_flag(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "token", "my-secret-token-value"])
        result = runner.invoke(main, ["config", "get", "token", "--unmask"])
        assert result.exit_code == 0
        assert result.output.strip() == "my-secret-token-value"

    def test_config_get_non_sensitive_key_not_masked(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://localhost:8080"])
        result = runner.invoke(main, ["config", "get", "base_url"])
        assert result.exit_code == 0
        assert result.output.strip() == "http://localhost:8080"

    def test_config_list_masks_token(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "token", "longtoken123"])
        result = runner.invoke(main, ["config", "list"])
        assert result.exit_code == 0
        assert "token: lon***23" in result.output
        assert "longtoken123" not in result.output


class TestMcpConfigFileKey:
    def test_mcp_config_file_in_optional_keys(self) -> None:
        assert "mcp_config_file" in OPTIONAL_KEYS

    def test_mcp_config_file_in_file_keys(self) -> None:
        assert "mcp_config_file" in FILE_KEYS


class TestParseBool:
    def test_true_values(self) -> None:
        for val in ("true", "True", "TRUE", "1", "yes", "Yes"):
            assert parse_bool(val) is True

    def test_false_values(self) -> None:
        for val in ("false", "False", "0", "no", "anything", ""):
            assert parse_bool(val) is False

    def test_none_returns_default_false(self) -> None:
        assert parse_bool(None) is False

    def test_none_returns_custom_default(self) -> None:
        assert parse_bool(None, default=True) is True


class TestUnsetValue:
    def test_removes_existing_key(self) -> None:
        data: dict[str, Any] = {
            "active_profile": "default",
            "profiles": {"default": {"base_url": "http://localhost", "token": "t"}},
        }
        assert unset_value(data, "token") is True
        assert "token" not in data["profiles"]["default"]
        assert "base_url" in data["profiles"]["default"]

    def test_noop_when_key_missing(self) -> None:
        data: dict[str, Any] = {
            "active_profile": "default",
            "profiles": {"default": {"base_url": "http://localhost"}},
        }
        assert unset_value(data, "nonexistent") is False
        assert data["profiles"]["default"] == {"base_url": "http://localhost"}

    def test_noop_when_profile_missing(self) -> None:
        data: dict[str, Any] = {
            "active_profile": "default",
            "profiles": {"other": {"base_url": "http://localhost"}},
        }
        assert unset_value(data, "base_url") is False
        assert data["profiles"] == {"other": {"base_url": "http://localhost"}}

    def test_noop_when_no_profiles_key(self) -> None:
        data: dict[str, Any] = {"active_profile": "default"}
        assert unset_value(data, "base_url") is False
        assert "profiles" not in data

    def test_explicit_profile(self) -> None:
        data: dict[str, Any] = {
            "active_profile": "default",
            "profiles": {
                "default": {"base_url": "http://localhost"},
                "prod": {"base_url": "https://prod", "token": "secret"},
            },
        }
        assert unset_value(data, "token", profile="prod") is True
        assert "token" not in data["profiles"]["prod"]
        assert data["profiles"]["default"] == {"base_url": "http://localhost"}


class TestConfigUnsetCommand:
    def test_unset_existing_key(self, runner: CliRunner, config_file: Path) -> None:
        assert (
            runner.invoke(
                main, ["config", "set", "base_url", "http://localhost:8080"]
            ).exit_code
            == 0
        )
        assert (
            runner.invoke(main, ["config", "set", "token", "my-token"]).exit_code == 0
        )
        result = runner.invoke(main, ["config", "unset", "token"])
        assert result.exit_code == 0
        assert "Unset 'token' from profile 'default'" in result.output
        data = load_config()
        assert "token" not in data["profiles"]["default"]
        assert data["profiles"]["default"]["base_url"] == "http://localhost:8080"

    def test_unset_missing_key_errors(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        assert (
            runner.invoke(
                main, ["config", "set", "base_url", "http://localhost:8080"]
            ).exit_code
            == 0
        )
        result = runner.invoke(main, ["config", "unset", "nonexistent"])
        assert result.exit_code != 0
        assert "Key 'nonexistent' not found" in result.output

    def test_unset_with_profile_flag(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        assert (
            runner.invoke(
                main, ["--profile", "prod", "config", "set", "token", "prod-token"]
            ).exit_code
            == 0
        )
        result = runner.invoke(main, ["--profile", "prod", "config", "unset", "token"])
        assert result.exit_code == 0
        assert "profile 'prod'" in result.output
        data = load_config()
        assert "token" not in data["profiles"]["prod"]


# --- create / delete profile unit tests ---


class TestCreateProfile:
    def test_creates_empty_profile(self) -> None:
        data: dict[str, Any] = {"active_profile": "default", "profiles": {}}
        create_profile(data, "staging")
        assert "staging" in data["profiles"]
        assert data["profiles"]["staging"] == {}

    def test_errors_on_duplicate(self) -> None:
        data: dict[str, Any] = {
            "active_profile": "default",
            "profiles": {"staging": {"base_url": "http://localhost"}},
        }
        with pytest.raises(Exception, match="already exists"):
            create_profile(data, "staging")

    def test_does_not_change_active_profile(self) -> None:
        data: dict[str, Any] = {"active_profile": "default", "profiles": {}}
        create_profile(data, "new")
        assert data["active_profile"] == "default"

    def test_rejects_traversal_name(self) -> None:
        data: dict[str, Any] = {"active_profile": "default", "profiles": {}}
        with pytest.raises(Exception, match="Invalid profile name"):
            create_profile(data, "../escape")


class TestDeleteProfile:
    def test_deletes_existing_profile(self) -> None:
        data: dict[str, Any] = {
            "active_profile": "default",
            "profiles": {
                "default": {"base_url": "http://localhost"},
                "staging": {"base_url": "http://staging"},
            },
        }
        delete_profile(data, "staging")
        assert "staging" not in data["profiles"]
        assert "default" in data["profiles"]

    def test_errors_on_active_profile(self) -> None:
        data: dict[str, Any] = {
            "active_profile": "default",
            "profiles": {"default": {}},
        }
        with pytest.raises(Exception, match="Cannot delete the active profile"):
            delete_profile(data, "default")

    def test_errors_on_nonexistent_profile(self) -> None:
        data: dict[str, Any] = {"active_profile": "default", "profiles": {}}
        with pytest.raises(Exception, match="does not exist"):
            delete_profile(data, "ghost")

    def test_rejects_traversal_name(self) -> None:
        data: dict[str, Any] = {
            "active_profile": "default",
            "profiles": {"default": {}},
        }
        with pytest.raises(Exception, match="Invalid profile name"):
            delete_profile(data, "../escape")


class TestValidateProfileName:
    @pytest.mark.parametrize("name", ["default", "prod", "my-profile", "v1.2", "a"])
    def test_accepts_valid_names(self, name: str) -> None:
        validate_profile_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "..",
            "../etc",
            "foo/../bar",
            "/absolute",
            "has/slash",
            "has space",
            ".leading-dot",
            "-leading-dash",
            "trailing-dot.",
            "trailing-dash-",
        ],
    )
    def test_rejects_unsafe_names(self, name: str) -> None:
        with pytest.raises(Exception, match="Invalid profile name"):
            validate_profile_name(name)


class TestValidatePathWithin:
    def test_accepts_path_within_base(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        _validate_path_within(base / "child", base)

    def test_rejects_path_escaping_base(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        with pytest.raises(Exception, match="escapes base directory"):
            _validate_path_within(base / ".." / "outside", base)



# --- config create / delete CLI tests ---


class TestConfigCreateCommand:
    def test_create_new_profile(self, runner: CliRunner, config_file: Path) -> None:
        result = runner.invoke(main, ["config", "create", "staging"])
        assert result.exit_code == 0
        assert "Created profile 'staging'" in result.output
        data = load_config()
        assert "staging" in data["profiles"]
        assert data["profiles"]["staging"] == {}

    def test_create_with_activate(self, runner: CliRunner, config_file: Path) -> None:
        result = runner.invoke(main, ["config", "create", "prod", "--activate"])
        assert result.exit_code == 0
        assert "(active)" in result.output
        data = load_config()
        assert data["active_profile"] == "prod"

    def test_create_without_activate_keeps_active(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        result = runner.invoke(main, ["config", "create", "staging"])
        assert result.exit_code == 0
        data = load_config()
        assert data["active_profile"] == DEFAULT_PROFILE

    def test_create_duplicate_errors(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "create", "staging"])
        result = runner.invoke(main, ["config", "create", "staging"])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_create_then_set_values(self, runner: CliRunner, config_file: Path) -> None:
        runner.invoke(main, ["config", "create", "staging", "--activate"])
        runner.invoke(main, ["config", "set", "base_url", "http://staging:8080"])
        data = load_config()
        assert data["profiles"]["staging"]["base_url"] == "http://staging:8080"

    def test_create_then_use(self, runner: CliRunner, config_file: Path) -> None:
        runner.invoke(main, ["config", "create", "staging"])
        result = runner.invoke(main, ["config", "use", "staging"])
        assert result.exit_code == 0
        assert "Active profile set to 'staging'" in result.output


class TestConfigDeleteCommand:
    def test_delete_inactive_profile(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "create", "staging"])
        result = runner.invoke(main, ["config", "delete", "staging"])
        assert result.exit_code == 0
        assert "Deleted profile 'staging'" in result.output
        data = load_config()
        assert "staging" not in data["profiles"]

    def test_delete_active_profile_errors(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "create", "staging", "--activate"])
        result = runner.invoke(main, ["config", "delete", "staging"])
        assert result.exit_code != 0
        assert "Cannot delete the active profile" in result.output

    def test_delete_nonexistent_profile_errors(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        result = runner.invoke(main, ["config", "delete", "ghost"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_delete_preserves_other_profiles(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://default:8080"])
        runner.invoke(main, ["config", "create", "staging"])
        runner.invoke(
            main,
            [
                "--profile",
                "staging",
                "config",
                "set",
                "base_url",
                "http://staging:8080",
            ],
        )
        runner.invoke(main, ["config", "delete", "staging"])
        data = load_config()
        assert "default" in data["profiles"]
        assert data["profiles"]["default"]["base_url"] == "http://default:8080"


class TestConfigListHint:
    def test_hint_shown_when_multiple_profiles(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://default:8080"])
        runner.invoke(main, ["config", "create", "staging"])
        result = runner.invoke(main, ["config", "list"])
        assert result.exit_code == 0
        assert "(use --all to see all profiles)" in result.output

    def test_hint_not_shown_for_single_profile(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://default:8080"])
        result = runner.invoke(main, ["config", "list"])
        assert result.exit_code == 0
        assert "--all" not in result.output

    def test_hint_not_shown_with_all_flag(
        self, runner: CliRunner, config_file: Path
    ) -> None:
        runner.invoke(main, ["config", "set", "base_url", "http://default:8080"])
        runner.invoke(main, ["config", "create", "staging"])
        result = runner.invoke(main, ["config", "list", "--all"])
        assert result.exit_code == 0
        assert "(use --all to see all profiles)" not in result.output
