"""EvalHub CLI configuration and profile management.

Config is stored at ~/.config/evalhub/config.yaml with structure:

    active_profile: default
    profiles:
      default:
        base_url: http://localhost:8080
        token: ...
      prod:
        base_url: https://evalhub.example.com
        token: ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "evalhub"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"

REQUIRED_KEYS = ("base_url", "token", "tenant")
OPTIONAL_KEYS = ("provider", "insecure", "timeout")
KNOWN_KEYS = set(REQUIRED_KEYS) | set(OPTIONAL_KEYS)

DEFAULT_PROFILE = "default"


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


def missing_required_keys(
    data: dict[str, Any], profile: str | None = None
) -> list[str]:
    """Return required keys not yet set in the profile."""
    prof = get_profile(data, profile)
    return [k for k in REQUIRED_KEYS if k not in prof]


def is_known_key(key: str) -> bool:
    """Check whether a key is a recognised config key."""
    return key in KNOWN_KEYS


def set_active_profile(data: dict[str, Any], profile: str) -> dict[str, Any]:
    """Switch the active profile."""
    data["active_profile"] = profile
    return data
