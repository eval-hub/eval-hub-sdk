"""Helpers for resolving model auth from environment.

In Kubernetes the auth files live under a well-known mount
(``/var/run/secrets/model``).  In local mode, when ``job_spec.model.auth``
is present, the directory is derived from its ``secret_ref`` field which
points to a user-provided filesystem path that may contain ``api-key``,
``hf-token``, and ``ca_cert`` files.

The job spec is discovered automatically from ``EVALHUB_JOB_SPEC_PATH``
(or the mode-dependent default) — callers do not need to pass it in.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import EvalHubMode, get_job_spec_path
from .models.job import JobSpec
from .settings import AdapterSettings

logger = logging.getLogger(__name__)
_MODEL_AUTH_DIR = Path("/var/run/secrets/model")


def _resolve_auth_dir() -> Path:
    try:
        job_spec = JobSpec.from_file(get_job_spec_path())
    except (FileNotFoundError, ValueError) as exc:
        logger.debug("Job spec not available: %s", exc)
        return _MODEL_AUTH_DIR

    settings = AdapterSettings.from_env()
    if settings.mode == EvalHubMode.LOCAL and job_spec.model.auth is not None:
        ref = Path(job_spec.model.auth.secret_ref)
        if ref.is_dir():
            return ref
    return _MODEL_AUTH_DIR


def _read_key_from_dir(auth_dir: Path, key_name: str) -> str | None:
    if not key_name or "/" in key_name or "\\" in key_name or key_name in (".", ".."):
        return None
    path = auth_dir / key_name
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read model auth key %s", key_name, exc_info=exc)
        return None
    return value or None


def read_model_auth_key(key_name: str) -> str | None:
    """Read a specific key from the model auth secret directory.

    The auth directory is resolved automatically from the job spec
    (via ``EVALHUB_JOB_SPEC_PATH``) in local mode, or from the default
    Kubernetes mount path.
    """
    cleaned = key_name.strip()
    if not cleaned:
        return None
    return _read_key_from_dir(_resolve_auth_dir(), cleaned)


@dataclass
class ModelCredentials:
    """Resolved model credentials.

    In Kubernetes ``api_key`` holds a ref token (e.g. ``"api-key:ref"``)
    resolved by the sidecar proxy.  In local mode ``api_key`` is the
    actual credential — the adapter sets the ``Authorization`` header
    directly.

    ``hf_token`` is the Hugging Face token from the ``hf-token`` file.
    ``ca_cert_path`` points to the ``ca_cert`` file when present.
    """

    api_key: str | None = field(default=None, repr=False)
    hf_token: str | None = field(default=None, repr=False)
    ca_cert_path: Path | None = field(default=None)


def resolve_model_credentials() -> ModelCredentials:
    """Resolve model authentication from the auth secret directory.

    Reads ``api-key``, ``hf-token``, and checks for ``ca_cert``.
    The auth directory is resolved automatically from the job spec.
    """
    auth_dir = _resolve_auth_dir()
    ca_cert = auth_dir / "ca_cert"
    return ModelCredentials(
        api_key=_read_key_from_dir(auth_dir, "api-key"),
        hf_token=_read_key_from_dir(auth_dir, "hf-token"),
        ca_cert_path=ca_cert if ca_cert.is_file() else None,
    )
