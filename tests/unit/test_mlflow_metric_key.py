"""MLflow metric key sanitization for REST API rules."""

import pytest
from evalhub.adapter.mlflow import sanitize_metric_key_for_api

pytestmark = pytest.mark.unit


def test_sanitize_lm_eval_style_comma() -> None:
    assert sanitize_metric_key_for_api("acc,none") == "acc_none"


def test_sanitize_preserves_allowed_chars() -> None:
    assert (
        sanitize_metric_key_for_api("exact_match,strict-match")
        == "exact_match_strict-match"
    )


def test_sanitize_empty_fallback() -> None:
    assert sanitize_metric_key_for_api(",,,") == "metric"


def test_sanitize_simple_name_unchanged() -> None:
    assert sanitize_metric_key_for_api("accuracy") == "accuracy"
