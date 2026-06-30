"""EvalHub CLI configuration and profile management.

Config is stored at ~/.config/evalhub/config.yaml with structure:

    active_profile: default
    profiles:
      default:
        base_url: http://localhost:8080
        token: ...
        tenant: ''  # empty tenant for localhost
      prod:
        base_url: https://evalhub.example.com
        token: ...
        tenant: 'team-a'
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

import click
import yaml

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "evalhub"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"

REQUIRED_KEYS = ("base_url", "token", "tenant")
OPTIONAL_KEYS = (
    "provider",
    "insecure",
    "timeout",
    "mcp_transport",
    "mcp_host",
    "mcp_port",
    "server_config_file",
)
KNOWN_KEYS = set(REQUIRED_KEYS) | set(OPTIONAL_KEYS)
SENSITIVE_KEYS = frozenset({"token"})
FILE_KEYS = frozenset({"server_config_file"})

DEFAULT_PROFILE = "default"


def mask_value(
    value: str, *, prefix_len: int = 3, suffix_len: int = 2, min_len: int = 8
) -> str:
    """Mask a sensitive value, showing only a prefix and suffix."""
    if len(value) < min_len:
        return "***"
    return f"{value[:prefix_len]}***{value[-suffix_len:]}"


def _config_path() -> Path:
    """Return the config file path, respecting EVALHUB_CONFIG env var."""
    env = os.environ.get("EVALHUB_CONFIG")
    if env:
        return Path(env)
    return DEFAULT_CONFIG_FILE


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the config file. Returns empty structure if file does not exist."""
    p = path or _config_path()
    if not p.exists():
        return {"active_profile": DEFAULT_PROFILE, "profiles": {}}
    with p.open("r") as f:
        data = yaml.safe_load(f) or {}
    if "active_profile" not in data:
        data["active_profile"] = DEFAULT_PROFILE
    if "profiles" not in data:
        data["profiles"] = {}
    return data


def save_config(data: dict[str, Any], path: Path | None = None) -> None:
    """Save config to disk with safe permissions (0600)."""
    p = path or _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    p.chmod(stat.S_IRUSR | stat.S_IWUSR)


def get_active_profile(data: dict[str, Any]) -> str:
    """Return the active profile name."""
    active = data.get("active_profile", DEFAULT_PROFILE)
    if not isinstance(active, str):
        return DEFAULT_PROFILE
    return active


def get_profile(data: dict[str, Any], profile: str | None = None) -> dict[str, Any]:
    """Return the settings dict for a profile (empty dict if it doesn't exist yet)."""
    name = profile or get_active_profile(data)
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        return {}
    result = profiles.get(name, {})
    if not isinstance(result, dict):
        return {}
    return result


def set_value(
    data: dict[str, Any], key: str, value: str, profile: str | None = None
) -> dict[str, Any]:
    """Set a key in a profile. Creates the profile if it doesn't exist."""
    name = profile or get_active_profile(data)
    profiles = data.setdefault("profiles", {})
    prof = profiles.setdefault(name, {})
    prof[key] = value
    return data


def get_value(data: dict[str, Any], key: str, profile: str | None = None) -> str | None:
    """Get a single value from a profile."""
    prof = get_profile(data, profile)
    return prof.get(key)


def unset_value(data: dict[str, Any], key: str, profile: str | None = None) -> bool:
    """Remove a key from a profile. Returns True if the key was present."""
    name = profile or get_active_profile(data)
    profiles = data.get("profiles")
    if profiles is None:
        return False
    prof = profiles.get(name)
    if prof is None:
        return False
    return prof.pop(key, None) is not None


def missing_required_keys(
    data: dict[str, Any], profile: str | None = None
) -> list[str]:
    """Return required keys not yet set in the profile."""
    prof = get_profile(data, profile)
    return [k for k in REQUIRED_KEYS if k not in prof]


def is_known_key(key: str) -> bool:
    """Check whether a key is a recognised config key."""
    return key in KNOWN_KEYS


def is_file_key(key: str) -> bool:
    """Check whether a key references an external file."""
    return key in FILE_KEYS


_FILE_KEY_STORE_DIRS: dict[str, Path] = {
    "server_config_file": DEFAULT_CONFIG_DIR / "server",
}


def validate_config_file(path: Path) -> None:
    """Validate that *path* exists and contains a YAML mapping.

    Raises ``click.ClickException`` on any validation failure.
    """
    if not path.is_file():
        raise click.ClickException(f"File not found: {path}")
    try:
        with path.open() as f:
            parsed = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise click.ClickException(f"Invalid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise click.ClickException(
            f"Config file must contain a YAML mapping, got {type(parsed).__name__}"
        )


def store_file_key(key: str, src: Path, profile_name: str) -> str:
    """Copy *src* into the profile-specific storage dir for *key*.

    Returns the absolute path of the stored copy.
    """
    base = _FILE_KEY_STORE_DIRS[key]
    dest_dir = base / profile_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "config.yaml"
    shutil.copy2(str(src), str(dest))
    dest.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return str(dest)


def remove_file_key(key: str, profile_name: str) -> None:
    """Delete the stored file for *key* and clean up the directory if empty."""
    base = _FILE_KEY_STORE_DIRS[key]
    dest_dir = base / profile_name
    dest = dest_dir / "config.yaml"
    if dest.exists():
        dest.unlink()
    try:
        dest_dir.rmdir()
    except OSError:
        pass


def set_active_profile(data: dict[str, Any], profile: str) -> dict[str, Any]:
    """Switch the active profile."""
    data["active_profile"] = profile
    return data


def parse_bool(value: Any, *, default: bool = False) -> bool:
    """Parse a config value as a boolean."""
    if value is None:
        return default
    return str(value).lower() in ("true", "1", "yes")


def build_mcp_config(
    profile: dict[str, Any], *, default_transport: str = "http"
) -> dict[str, Any]:
    """Build the Go MCP binary config dict from a CLI profile."""
    try:
        port = int(profile.get("mcp_port", 3001))
    except (TypeError, ValueError):
        port = 3001
    return {
        "base_url": profile.get("base_url", "http://localhost:8080"),
        "token": profile.get("token", ""),
        "tenant": profile.get("tenant", ""),
        "insecure": parse_bool(profile.get("insecure")),
        "transport": profile.get("mcp_transport", default_transport),
        "host": profile.get("mcp_host", "localhost"),
        "port": port,
    }
