"""Unit tests for the OTEL log pipeline (severity, trace correlation, job context)."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import evalhub.adapter.telemetry as telemetry_mod
import pytest
from evalhub.adapter.telemetry import (
    _JobContextFilter,
    clear_log_job_context,
    configure_telemetry,
    set_log_job_context,
)
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

pytest.importorskip(
    "opentelemetry.sdk._logs",
    reason="opentelemetry-sdk logs module not available",
)
pytest.importorskip(
    "opentelemetry.exporter.otlp.proto.grpc._log_exporter",
    reason="opentelemetry-exporter-otlp-proto-grpc log exporter not installed",
)

from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler  # noqa: E402
from opentelemetry.sdk._logs.export import (  # noqa: E402
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_state() -> Generator[None, None, None]:
    """Reset module-level state and the global TracerProvider for each test."""
    saved_provider = trace_api._TRACER_PROVIDER
    saved_done = trace_api._TRACER_PROVIDER_SET_ONCE._done
    saved_installed = telemetry_mod._provider_installed
    saved_owns = telemetry_mod._owns_provider
    saved_log = telemetry_mod._log_provider_installed
    saved_log_owns = telemetry_mod._owns_log_provider
    saved_log_handler = telemetry_mod._log_handler_installed

    trace_api._TRACER_PROVIDER = None
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False
    telemetry_mod._provider_installed = None
    telemetry_mod._owns_provider = False
    telemetry_mod._log_provider_installed = None
    telemetry_mod._owns_log_provider = False
    telemetry_mod._log_handler_installed = None

    yield

    # Remove any OTEL LoggingHandlers that tests may have added to root logger
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, LoggingHandler)]

    if (
        telemetry_mod._log_provider_installed is not None
        and telemetry_mod._owns_log_provider
    ):
        try:
            telemetry_mod._log_provider_installed.shutdown()
        except Exception:
            pass
    if telemetry_mod._provider_installed is not None and telemetry_mod._owns_provider:
        try:
            telemetry_mod._provider_installed.shutdown()
        except Exception:
            pass

    telemetry_mod._provider_installed = saved_installed
    telemetry_mod._owns_provider = saved_owns
    telemetry_mod._log_provider_installed = saved_log
    telemetry_mod._owns_log_provider = saved_log_owns
    telemetry_mod._log_handler_installed = saved_log_handler
    trace_api._TRACER_PROVIDER = saved_provider
    trace_api._TRACER_PROVIDER_SET_ONCE._done = saved_done

    clear_log_job_context()


@pytest.fixture()
def log_exporter() -> Generator[InMemoryLogExporter, None, None]:
    """Set up a LoggerProvider with in-memory exporter and attach to root logger.

    Returns the exporter so tests can inspect captured log records.
    """
    from opentelemetry.sdk.resources import Resource

    exporter = InMemoryLogExporter()
    resource = Resource.create({"service.name": "test-adapter"})
    provider = LoggerProvider(resource=resource)
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))

    handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)
    handler.addFilter(_JobContextFilter())
    root = logging.getLogger()
    root.addHandler(handler)

    yield exporter

    root.removeHandler(handler)
    provider.shutdown()


@pytest.fixture()
def span_exporter() -> InMemorySpanExporter:
    """Install a TracerProvider with in-memory span exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)
    return exporter


# ---------------------------------------------------------------------------
# Tests: Log pipeline is installed by configure_telemetry
# ---------------------------------------------------------------------------


class TestLogPipelineInstalled:
    def test_log_provider_installed_with_endpoint(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        assert telemetry_mod._log_provider_installed is not None
        assert telemetry_mod._owns_log_provider is True

    def test_no_log_provider_without_endpoint(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configure_telemetry()
        assert telemetry_mod._log_provider_installed is None

    def test_logging_handler_added_to_root_logger(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        root = logging.getLogger()
        otel_handlers = [h for h in root.handlers if isinstance(h, LoggingHandler)]
        assert len(otel_handlers) >= 1

    def test_logging_handler_has_job_context_filter(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        root = logging.getLogger()
        otel_handlers = [h for h in root.handlers if isinstance(h, LoggingHandler)]
        assert otel_handlers
        filters = otel_handlers[0].filters
        assert any(isinstance(f, _JobContextFilter) for f in filters)


# ---------------------------------------------------------------------------
# Tests: Severity mapping
# ---------------------------------------------------------------------------


class TestSeverityMapping:
    """The OTEL LoggingHandler maps Python log levels to severity_text."""

    @pytest.mark.parametrize(
        "py_level, expected_severity_text",
        [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARN"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "FATAL"),
        ],
    )
    def test_severity_text_set_correctly(
        self,
        log_exporter: InMemoryLogExporter,
        py_level: int,
        expected_severity_text: str,
    ) -> None:
        test_logger = logging.getLogger("test.severity")
        test_logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

        test_logger.log(py_level, "severity test at level %s", py_level)

        records = log_exporter.get_finished_logs()
        matching = [
            r
            for r in records
            if r.log_record.body is not None
            and "severity test" in str(r.log_record.body)
        ]
        assert matching, f"No log record found for level {py_level}"
        lr = matching[-1].log_record
        assert lr.severity_text == expected_severity_text

    @pytest.mark.parametrize(
        "py_level, min_severity_number",
        [
            (logging.DEBUG, 1),
            (logging.INFO, 9),
            (logging.WARNING, 13),
            (logging.ERROR, 17),
            (logging.CRITICAL, 21),
        ],
    )
    def test_severity_number_set(
        self,
        log_exporter: InMemoryLogExporter,
        py_level: int,
        min_severity_number: int,
    ) -> None:
        test_logger = logging.getLogger("test.severity_num")
        test_logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

        test_logger.log(py_level, "num test %s", py_level)

        records = log_exporter.get_finished_logs()
        matching = [
            r
            for r in records
            if r.log_record.body is not None and "num test" in str(r.log_record.body)
        ]
        assert matching
        lr = matching[-1].log_record
        sn = lr.severity_number
        assert sn is not None
        assert sn.value >= min_severity_number


# ---------------------------------------------------------------------------
# Tests: Trace correlation (trace_id / span_id in log records)
# ---------------------------------------------------------------------------


class TestTraceCorrelation:
    def test_log_inside_span_carries_trace_id(
        self,
        log_exporter: InMemoryLogExporter,
        span_exporter: InMemorySpanExporter,
    ) -> None:
        tracer = trace_api.get_tracer("test")

        with tracer.start_as_current_span("test-span") as span:
            expected_trace_id = span.get_span_context().trace_id
            expected_span_id = span.get_span_context().span_id
            logging.getLogger("test.trace").info("inside span")

        records = log_exporter.get_finished_logs()
        matching = [
            r
            for r in records
            if r.log_record.body is not None and "inside span" in str(r.log_record.body)
        ]
        assert matching
        lr = matching[-1].log_record
        assert lr.trace_id == expected_trace_id
        assert lr.span_id == expected_span_id

    def test_log_outside_span_has_zero_trace_id(
        self,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        logging.getLogger("test.notrace").info("outside span")

        records = log_exporter.get_finished_logs()
        matching = [
            r
            for r in records
            if r.log_record.body is not None
            and "outside span" in str(r.log_record.body)
        ]
        assert matching
        lr = matching[-1].log_record
        assert lr.trace_id == 0


# ---------------------------------------------------------------------------
# Tests: Job context injection
# ---------------------------------------------------------------------------


class TestJobContext:
    def test_set_log_job_context_injects_attributes(
        self,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        set_log_job_context(
            job_id="job-42",
            benchmark_id="mmlu",
            provider_id="lighteval",
            model_id="llama-3",
        )

        logging.getLogger("test.job_ctx").info("context test")

        records = log_exporter.get_finished_logs()
        matching = [
            r
            for r in records
            if r.log_record.body is not None
            and "context test" in str(r.log_record.body)
        ]
        assert matching
        attrs = dict(matching[-1].log_record.attributes or {})
        assert attrs["evalhub.job_id"] == "job-42"
        assert attrs["evalhub.benchmark_id"] == "mmlu"
        assert attrs["evalhub.provider_id"] == "lighteval"
        assert attrs["evalhub.model_id"] == "llama-3"

    def test_clear_log_job_context_removes_attributes(
        self,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        set_log_job_context(job_id="job-1")
        clear_log_job_context()

        logging.getLogger("test.cleared").info("cleared test")

        records = log_exporter.get_finished_logs()
        matching = [
            r
            for r in records
            if r.log_record.body is not None
            and "cleared test" in str(r.log_record.body)
        ]
        assert matching
        attrs = dict(matching[-1].log_record.attributes or {})
        assert "evalhub.job_id" not in attrs

    def test_partial_context_only_sets_provided_keys(
        self,
        log_exporter: InMemoryLogExporter,
    ) -> None:
        set_log_job_context(job_id="job-99")

        logging.getLogger("test.partial").info("partial test")

        records = log_exporter.get_finished_logs()
        matching = [
            r
            for r in records
            if r.log_record.body is not None
            and "partial test" in str(r.log_record.body)
        ]
        assert matching
        attrs = dict(matching[-1].log_record.attributes or {})
        assert attrs["evalhub.job_id"] == "job-99"
        assert "evalhub.benchmark_id" not in attrs

    def test_from_job_spec_sets_log_context(self) -> None:
        from evalhub.adapter.telemetry import EvalTracer, _job_log_context

        spec = MagicMock()
        spec.id = "job-77"
        spec.provider_id = "inspect"
        spec.benchmark_id = "petri-sycophancy"
        spec.model.name = "gpt-4o"

        EvalTracer.from_job_spec(spec)

        ctx = _job_log_context.get({})
        assert ctx["evalhub.job_id"] == "job-77"
        assert ctx["evalhub.benchmark_id"] == "petri-sycophancy"
        assert ctx["evalhub.provider_id"] == "inspect"
        assert ctx["evalhub.model_id"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Tests: Shutdown cleans up log provider
# ---------------------------------------------------------------------------


class TestLogShutdown:
    def test_shutdown_clears_log_provider(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        assert telemetry_mod._log_provider_installed is not None
        assert telemetry_mod._owns_log_provider is True

        telemetry_mod._shutdown_provider()

        assert telemetry_mod._log_provider_installed is None
        assert telemetry_mod._owns_log_provider is False

    def test_shutdown_clears_both_providers(self) -> None:
        configure_telemetry(endpoint="http://localhost:4317")
        assert telemetry_mod._provider_installed is not None
        assert telemetry_mod._log_provider_installed is not None

        telemetry_mod._shutdown_provider()

        assert telemetry_mod._provider_installed is None
        assert telemetry_mod._log_provider_installed is None


# ---------------------------------------------------------------------------
# Tests: _JobContextFilter standalone
# ---------------------------------------------------------------------------


class TestJobContextFilter:
    def test_filter_always_returns_true(self) -> None:
        f = _JobContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert f.filter(record) is True

    def test_filter_sets_attributes_on_record(self) -> None:
        set_log_job_context(job_id="j1", benchmark_id="b1")

        f = _JobContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)

        assert getattr(record, "evalhub.job_id") == "j1"
        assert getattr(record, "evalhub.benchmark_id") == "b1"

    def test_filter_no_attributes_when_context_empty(self) -> None:
        clear_log_job_context()

        f = _JobContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)

        assert not hasattr(record, "evalhub.job_id")
