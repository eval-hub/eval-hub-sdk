"""Pytest configuration for eval-hub-sdk tests."""

from typing import Any


def pytest_addoption(parser: Any) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--e2e-debug",
        action="store_true",
        default=False,
        help="Enable DEBUG logging for E2E test fixtures",
    )
