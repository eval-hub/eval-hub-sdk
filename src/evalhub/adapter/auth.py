"""Helpers for resolving model auth from environment."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
_MODEL_AUTH_DIR = Path("/var/run/secrets/model")

_ENV_LOCAL_MODEL_API_KEY = "EVALHUB_LOCAL_MODEL_API_KEY"
_ENV_LOCAL_MODEL_CA_CERT_PATH = "EVALHUB_LOCAL_MODEL_CA_CERT_PATH"
_ENV_LOCAL_MODEL_HF_TOKEN = "EVALHUB_LOCAL_MODEL_HF_TOKEN"


def read_model_auth_key(key_name: str) -> str | None:
    """Read a specific key from the mounted model auth secret."""
    cleaned = key_name.strip()
    if not cleaned:
        return None
    path = _MODEL_AUTH_DIR / cleaned
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Failed to read model auth key %s", cleaned, exc_info=exc)
        return None
    return value or None


def _read_model_auth_path(key_name: str) -> str | None:
    """Return the path to a mounted model auth key file, if it exists."""
    cleaned = key_name.strip()
    if not cleaned:
        return None
    path = _MODEL_AUTH_DIR / cleaned
    if not path.is_file():
        return None
    return str(path)


@dataclass
class ModelCredentials:
    """Resolved model credentials.

    In K8s mode, api_key holds the ref token (e.g. 'api-key:ref') that
    the sidecar proxy resolves to the real credential. In local mode,
    api_key holds the real credential read from EVALHUB_LOCAL_MODEL_API_KEY.

    ca_cert_path is the filesystem path to the CA certificate PEM used
    for TLS verification against the model endpoint.

    hf_token is the HuggingFace token for gated dataset/tokenizer access.
    """

    api_key: str | None = field(default=None, repr=False)
    ca_cert_path: str | None = field(default=None)
    hf_token: str | None = field(default=None, repr=False)


def resolve_model_credentials() -> ModelCredentials:
    """Resolve model authentication from the pod environment.

    Uses a fallback pattern: K8s-mounted secret files take precedence,
    EVALHUB_LOCAL_MODEL_* env vars are the local-mode fallback.
    """
    return ModelCredentials(
        api_key=read_model_auth_key("api-key")
        or os.environ.get(_ENV_LOCAL_MODEL_API_KEY),
        ca_cert_path=_read_model_auth_path("ca_cert")
        or os.environ.get(_ENV_LOCAL_MODEL_CA_CERT_PATH),
        hf_token=read_model_auth_key("hf-token")
        or os.environ.get(_ENV_LOCAL_MODEL_HF_TOKEN),
    )
