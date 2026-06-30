"""Helpers for resolving model auth from environment."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
_MODEL_AUTH_DIR = Path("/var/run/secrets/model")


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


@dataclass
class ModelCredentials:
    """Resolved model credentials from the pod environment.

    api_key holds the ref token (e.g. 'api-key:ref') that the sidecar
    proxy resolves to the real credential. CA cert and SA token injection
    are handled transparently by the sidecar — not the adapter.
    """

    api_key: str | None = field(default=None, repr=False)


def resolve_model_credentials() -> ModelCredentials:
    """Resolve model authentication from the pod environment.

    Reads the api-key ref token from the mounted model auth secret.
    CA cert and SA token injection are handled by the sidecar proxy.
    """
    return ModelCredentials(api_key=read_model_auth_key("api-key"))
