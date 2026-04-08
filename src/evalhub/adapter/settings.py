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
from typing import Annotated, Self

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from evalhub.adapter.config import (
    DEFAULT_JOB_SPEC_PATH_K8S,
    DEFAULT_JOB_SPEC_PATH_LOCAL,
    JOB_SPEC_PATH_ENV,
    EvalHubMode,
    MlflowBackend,
)


class AdapterSettings(BaseSettings):
    """Settings for adapter execution environment."""

    # We intentionally do not use an env prefix to keep compatibility with the
    # existing env var names used in POCs and docs.
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Execution mode affects defaults only (env vars always win).
    mode: Annotated[
        EvalHubMode,
        BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v),
    ] = Field(default=EvalHubMode.LOCAL, validation_alias="EVALHUB_MODE")

    # Job spec configuration
    job_spec_path: Path | None = Field(
        default=None, validation_alias="EVALHUB_JOB_SPEC_PATH"
    )

    # OCI registry configuration
    oci_auth_config_path: Path | None = Field(
        default=None, validation_alias="OCI_AUTH_CONFIG_PATH"
    )
    oci_insecure: bool = Field(default=False, validation_alias="OCI_INSECURE")

    # Optional TLS CA for HTTPS to eval-hub (e.g. verify TLS to the runtime sidecar).
    # Forwarded by ``DefaultCallbacks.from_adapter``. Upstream bearer auth is injected by the
    # sidecar in Kubernetes job pods, not from the adapter process.
    ca_bundle_path: Path | None = Field(
        default=None, validation_alias="EVALHUB_CA_BUNDLE_PATH"
    )

    # Connection to evalhub
    # (separate of OCI, as local EH might be localhost but OCI registry a real one)
    evalhub_insecure: bool = Field(default=False, validation_alias="EVALHUB_INSECURE")

    # MLflow backend selection
    mlflow_backend: Annotated[
        MlflowBackend,
        BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v),
    ] = Field(default=MlflowBackend.ODH, validation_alias="EVALHUB_MLFLOW_BACKEND")

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
        return (
            Path(DEFAULT_JOB_SPEC_PATH_K8S)
            if self.mode == EvalHubMode.K8S
            else Path(DEFAULT_JOB_SPEC_PATH_LOCAL)
        )

    def validate_runtime(self) -> None:
        """Validate that required settings are available for adapter runtime."""
        if not self.resolved_job_spec_path.exists():
            raise FileNotFoundError(
                f"Job spec file not found at {self.resolved_job_spec_path}. "
                f"Set {JOB_SPEC_PATH_ENV} (or EVALHUB_MODE=k8s for {DEFAULT_JOB_SPEC_PATH_K8S})."
            )
