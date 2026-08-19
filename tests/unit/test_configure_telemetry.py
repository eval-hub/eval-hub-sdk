"""Unit tests for the configure_telemetry() helper."""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import evalhub.adapter.telemetry as telemetry_mod
import pytest
from evalhub.adapter.telemetry import configure_telemetry
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

pytest.importorskip(
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    reason="opentelemetry-exporter-otlp-proto-grpc not installed",
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_global_state() -> None:  # type: ignore[misc]
    """Reset module-level state and the global TracerProvider for each test."""
    saved_provider = trace_api._TRACER_PROVIDER
    saved_done = trace_api._TRACER_PROVIDER_SET_ONCE._done
    saved_installed = telemetry_mod._provider_installed
    saved_owns = telemetry_mod._owns_provider

    trace_api._TRACER_PROVIDER = None
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False
    telemetry_mod._provider_installed = None
    telemetry_mod._owns_provider = False

    yield

    if telemetry_mod._provider_installed is not None and telemetry_mod._owns_provider:
        try:
            telemetry_mod._provider_installed.shutdown()
        except Exception:
            pass
    telemetry_mod._provider_installed = saved_installed
    telemetry_mod._owns_provider = saved_owns
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

    def test_owns_provider_flag_set(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        assert telemetry_mod._owns_provider is True

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


class TestExistingGlobalProvider:
    """configure_telemetry reuses an already-active SDK TracerProvider."""

    def test_reuses_preinstalled_provider(self) -> None:
        from opentelemetry.sdk.resources import Resource

        external = TracerProvider(
            sampler=ALWAYS_ON,
            resource=Resource.create({"service.name": "external-svc"}),
        )
        trace_api.set_tracer_provider(external)

        result = configure_telemetry(endpoint="http://localhost:4317")

        assert result is True
        assert telemetry_mod._provider_installed is external
        assert telemetry_mod._owns_provider is False

    def test_does_not_shut_down_reused_provider(self) -> None:
        from opentelemetry.sdk.resources import Resource

        external = TracerProvider(
            sampler=ALWAYS_ON,
            resource=Resource.create({"service.name": "external-svc"}),
        )
        trace_api.set_tracer_provider(external)

        configure_telemetry(endpoint="http://localhost:4317")
        telemetry_mod._shutdown_provider()

        assert telemetry_mod._provider_installed is None
        assert telemetry_mod._owns_provider is False
        # The external provider should still be the active global — we did
        # not shut it down because we don't own it.
        assert trace_api.get_tracer_provider() is external

        # Clean up so fixture restore doesn't double-shutdown.
        external.shutdown()


class TestConcurrentInitialization:
    """Multiple threads calling configure_telemetry see consistent results."""

    def test_concurrent_calls_produce_single_provider(self) -> None:
        results: list[bool] = []
        barrier = threading.Barrier(4)

        def call() -> None:
            barrier.wait()
            results.append(configure_telemetry(endpoint="http://localhost:4317"))

        threads = [threading.Thread(target=call) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is True for r in results)
        assert telemetry_mod._provider_installed is not None
        # Only one provider should be the active global.
        active = trace_api.get_tracer_provider()
        assert active is telemetry_mod._provider_installed


class TestShutdown:
    def test_shutdown_clears_state(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        assert telemetry_mod._provider_installed is not None
        assert telemetry_mod._owns_provider is True
        telemetry_mod._shutdown_provider()
        assert telemetry_mod._provider_installed is None
        assert telemetry_mod._owns_provider is False

    def test_shutdown_is_noop_when_not_owner(self) -> None:
        telemetry_mod._provider_installed = TracerProvider(sampler=ALWAYS_ON)
        telemetry_mod._owns_provider = False
        telemetry_mod._shutdown_provider()
        assert telemetry_mod._provider_installed is None
        assert telemetry_mod._owns_provider is False
