"""Typed settings for adapter runtime configuration.

This module centralizes adapter configuration (env vars, defaults, validation).

It is intentionally small and dependency-light (pydantic-settings) so that:
- adapters don't scatter `os.getenv()` calls across entrypoints
- behavior is consistent across providers
- local development has a clear "mode" switch

The job spec is mounted in Kubernetes at `/meta/job.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AdapterSettings(BaseSettings):
    """Settings for adapter execution environment."""

    # We intentionally do not use an env prefix to keep compatibility with the
    # existing env var names used in POCs and docs.
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Execution mode affects defaults only (env vars always win).
    mode: Literal["k8s", "local"] = Field(
        default="local", validation_alias="EVALHUB_MODE"
    )

    # Job spec configuration
    job_spec_path: Path | None = Field(
        default=None, validation_alias="EVALHUB_JOB_SPEC_PATH"
    )

    # OCI registry configuration
    registry_url: str | None = Field(default=None, validation_alias="REGISTRY_URL")
    registry_username: str | None = Field(
        default=None, validation_alias="REGISTRY_USERNAME"
    )
    registry_password: str | None = Field(
        default=None, validation_alias="REGISTRY_PASSWORD"
    )
    registry_insecure: bool = Field(default=False, validation_alias="REGISTRY_INSECURE")

    # Authentication configuration (for Kubernetes ServiceAccount tokens)
    auth_token_path: Path | None = Field(
        default=None, validation_alias="EVALHUB_AUTH_TOKEN_PATH"
    )
    ca_bundle_path: Path | None = Field(
        default=None, validation_alias="EVALHUB_CA_BUNDLE_PATH"
    )

    @classmethod
    def from_env(cls) -> Self:
        """Load settings from environment variables.

        This is equivalent to `AdapterSettings()` but makes explicit that
        values are being read from the environment.
        """
        return cls()

    @property
    def resolved_job_spec_path(self) -> Path:
        """Resolve job spec path using mode defaults."""
        if self.job_spec_path is not None:
            return self.job_spec_path
        return Path("/meta/job.json") if self.mode == "k8s" else Path("meta/job.json")

    @property
    def resolved_auth_token_path(self) -> Path | None:
        """Resolve ServiceAccount token path with auto-detection.

        Returns the path to the ServiceAccount token if running in Kubernetes,
        or None if not available (local mode).
        """
        if self.auth_token_path is not None:
            return self.auth_token_path

        # Auto-detect Kubernetes ServiceAccount token
        default_token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        if default_token_path.exists():
            return default_token_path

        return None

    @property
    def resolved_ca_bundle_path(self) -> Path | None:
        """Resolve CA bundle path with auto-detection.

        Tries multiple common locations for service CA bundles in order:
        1. OpenShift service-ca (injected via annotation)
        2. Kubernetes ServiceAccount CA

        Returns None if no CA bundle is found (local mode).
        """
        if self.ca_bundle_path is not None:
            return self.ca_bundle_path

        # Try common CA bundle locations
        ca_paths = [
            Path("/etc/pki/ca-trust/source/anchors/service-ca.crt"),  # OpenShift
            Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"),  # Kubernetes
        ]

        for path in ca_paths:
            if path.exists():
                return path

        return None

    def validate_runtime(self) -> None:
        """Validate that required settings are available for adapter runtime."""
        if not self.resolved_job_spec_path.exists():
            raise FileNotFoundError(
                f"Job spec file not found at {self.resolved_job_spec_path}. "
                "Set EVALHUB_JOB_SPEC_PATH (or EVALHUB_MODE=k8s for /meta/job.json)."
            )

        # For the current adapter callbacks implementation, registry is required.
        if not self.registry_url:
            raise ValueError("REGISTRY_URL environment variable is required")
