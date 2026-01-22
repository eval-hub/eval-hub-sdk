"""Pytest configuration for eval-hub-sdk tests."""

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


def pytest_configure(config: Any) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test (run with --e2e flag)"
    )


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
