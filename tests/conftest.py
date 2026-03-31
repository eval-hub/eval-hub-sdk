"""Pytest configuration for eval-hub-sdk tests."""

import logging
from typing import Any

import pytest


def pytest_addoption(parser: Any) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run only E2E tests",
    )
    parser.addoption(
        "--e2e-debug",
        action="store_true",
        default=False,
        help="Enable DEBUG logging for E2E test fixtures",
    )


def pytest_configure(config: Any) -> None:
    """Register custom markers and configure logging."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test (run with --e2e flag)"
    )
    if config.getoption("--e2e-debug", default=False):
        e2e_logger = logging.getLogger("tests.e2e.conftest")
        e2e_logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
        e2e_logger.addHandler(handler)


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    e2e_flag = config.getoption("--e2e")

    skip_e2e = pytest.mark.skip(reason="E2E tests require --e2e flag")
    skip_non_e2e = pytest.mark.skip(
        reason="Non-E2E tests are skipped when --e2e flag is used"
    )

    for item in items:
        is_e2e = "e2e" in item.keywords

        if e2e_flag:
            # When --e2e flag is provided, skip non-e2e tests
            if not is_e2e:
                item.add_marker(skip_non_e2e)
        else:
            # When --e2e flag is NOT provided, skip e2e tests
            if is_e2e:
                item.add_marker(skip_e2e)
