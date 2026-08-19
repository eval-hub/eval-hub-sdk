"""Unit tests for the configure_telemetry() helper."""

from __future__ import annotations

import os
from unittest.mock import patch

import evalhub.adapter.telemetry as telemetry_mod
import pytest
from evalhub.adapter.telemetry import configure_telemetry
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_global_state() -> None:  # type: ignore[misc]
    """Reset the module-level _provider_installed and the global TracerProvider."""
    saved_provider = trace_api._TRACER_PROVIDER
    saved_done = trace_api._TRACER_PROVIDER_SET_ONCE._done
    saved_installed = telemetry_mod._provider_installed

    trace_api._TRACER_PROVIDER = None
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False
    telemetry_mod._provider_installed = None

    yield

    if telemetry_mod._provider_installed is not None:
        try:
            telemetry_mod._provider_installed.shutdown()
        except Exception:
            pass
    telemetry_mod._provider_installed = saved_installed
    trace_api._TRACER_PROVIDER = saved_provider
    trace_api._TRACER_PROVIDER_SET_ONCE._done = saved_done


class TestNoopWhenUnconfigured:
    def test_returns_false_without_endpoint(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert configure_telemetry() is False

    def test_no_provider_installed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configure_telemetry()
        assert telemetry_mod._provider_installed is None


class TestProviderInstalled:
    def test_returns_true_with_endpoint_arg(self) -> None:
        result = configure_telemetry(endpoint="http://localhost:4317")
        assert result is True

    def test_returns_true_with_env_var(self) -> None:
        with patch.dict(
            os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"}
        ):
            result = configure_telemetry()
        assert result is True

    def test_installs_tracer_provider(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        provider = trace_api.get_tracer_provider()
        assert isinstance(provider, TracerProvider)

    def test_service_name_from_arg(self) -> None:
        configure_telemetry(service_name="my-adapter", endpoint="http://localhost:4317")
        provider = trace_api.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.name"] == "my-adapter"

    def test_service_name_from_env(self) -> None:
        with patch.dict(os.environ, {"OTEL_SERVICE_NAME": "env-adapter"}):
            configure_telemetry(endpoint="http://localhost:4317")
        provider = trace_api.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.name"] == "env-adapter"

    def test_default_service_name(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        provider = trace_api.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.name"] == "evalhub-adapter"

    def test_arg_overrides_env_service_name(self) -> None:
        with patch.dict(os.environ, {"OTEL_SERVICE_NAME": "env-name"}):
            configure_telemetry(
                service_name="arg-name", endpoint="http://localhost:4317"
            )
        provider = trace_api.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.name"] == "arg-name"


class TestIdempotent:
    def test_second_call_returns_true_immediately(self) -> None:
        assert configure_telemetry(endpoint="http://localhost:4317") is True
        first_provider = telemetry_mod._provider_installed
        assert configure_telemetry(endpoint="http://other:4317") is True
        assert telemetry_mod._provider_installed is first_provider


class TestShutdown:
    def test_shutdown_clears_installed(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        assert telemetry_mod._provider_installed is not None
        telemetry_mod._shutdown_provider()
        assert telemetry_mod._provider_installed is None
