"""Unit tests for the adapter OTEL span instrumentation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from evalhub.adapter.telemetry import EvalTracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import StatusCode

pytestmark = pytest.mark.unit

VALID_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.fixture(autouse=True)
def _otel_provider(
    request: pytest.FixtureRequest,
) -> InMemorySpanExporter | None:
    """Install a fresh TracerProvider + InMemorySpanExporter for each test.

    Tests decorated with ``@pytest.mark.no_otel_provider`` skip the provider
    setup so the global no-op tracer is exercised instead.
    """
    from opentelemetry import trace as trace_api

    saved_provider = trace_api._TRACER_PROVIDER
    saved_done = trace_api._TRACER_PROVIDER_SET_ONCE._done

    trace_api._TRACER_PROVIDER = None
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False

    if "no_otel_provider" in {m.name for m in request.node.iter_markers()}:

        def _restore() -> None:
            trace_api._TRACER_PROVIDER = saved_provider
            trace_api._TRACER_PROVIDER_SET_ONCE._done = saved_done

        request.addfinalizer(_restore)
        return None

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)

    def _cleanup() -> None:
        provider.shutdown()
        trace_api._TRACER_PROVIDER = saved_provider
        trace_api._TRACER_PROVIDER_SET_ONCE._done = saved_done

    request.addfinalizer(_cleanup)
    return exporter


@pytest.fixture()
def exporter(_otel_provider: InMemorySpanExporter | None) -> InMemorySpanExporter:
    assert _otel_provider is not None, "This test requires the OTEL provider"
    return _otel_provider


@pytest.fixture()
def tracer() -> EvalTracer:
    return EvalTracer()


@pytest.fixture()
def job_tracer() -> EvalTracer:
    """EvalTracer pre-populated via a minimal mock JobSpec."""
    from unittest.mock import MagicMock

    spec = MagicMock()
    spec.id = "job-42"
    spec.provider_id = "lm_evaluation_harness"
    spec.benchmark_id = "mmlu"
    spec.model.name = "llama-3"
    return EvalTracer.from_job_spec(spec)


def _span_names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


class TestAllFiveSpanTypesExported:
    def test_full_lifecycle(
        self, exporter: InMemorySpanExporter, job_tracer: EvalTracer
    ) -> None:
        with job_tracer.evaluation_run():
            with job_tracer.dataset_load(source="hf://mmlu", size_samples=500):
                pass
            with job_tracer.inference_batch(model_id="llama-3", batch_size=32):
                pass
            with job_tracer.scoring(benchmark="mmlu", scorer="accuracy"):
                pass
            with job_tracer.result_log():
                pass

        names = _span_names(exporter)
        assert "evalhub.evaluation.run" in names
        assert "evalhub.evaluation.dataset_load" in names
        assert "evalhub.evaluation.inference" in names
        assert "evalhub.evaluation.scoring" in names
        assert "evalhub.evaluation.result_log" in names
        assert len(names) == 5


class TestSpanParentChildHierarchy:
    def test_children_share_root_parent(
        self, exporter: InMemorySpanExporter, job_tracer: EvalTracer
    ) -> None:
        with job_tracer.evaluation_run():
            with job_tracer.dataset_load():
                pass
            with job_tracer.inference_batch(batch_size=1):
                pass
            with job_tracer.scoring():
                pass
            with job_tracer.result_log():
                pass

        spans = {s.name: s for s in exporter.get_finished_spans()}
        root = spans["evalhub.evaluation.run"]
        root_ctx = root.context

        for child_name in (
            "evalhub.evaluation.dataset_load",
            "evalhub.evaluation.inference",
            "evalhub.evaluation.scoring",
            "evalhub.evaluation.result_log",
        ):
            child = spans[child_name]
            assert child.parent is not None, f"{child_name} has no parent"
            assert (
                child.parent.span_id == root_ctx.span_id
            ), f"{child_name} parent_span_id mismatch"


class TestRootSpanAttributes:
    def test_job_identity_attributes(
        self, exporter: InMemorySpanExporter, job_tracer: EvalTracer
    ) -> None:
        with job_tracer.evaluation_run():
            pass

        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name == "evalhub.evaluation.run")
        attrs = dict(root.attributes or {})

        assert attrs["evalhub.job_id"] == "job-42"
        assert attrs["evalhub.provider"] == "lm_evaluation_harness"
        assert attrs["evalhub.collection"] == "mmlu"
        assert attrs["evalhub.model_id"] == "llama-3"


class TestDatasetLoadSpanAttributes:
    def test_source_and_size(
        self, exporter: InMemorySpanExporter, tracer: EvalTracer
    ) -> None:
        with tracer.evaluation_run():
            with tracer.dataset_load(source="hf://mmlu", size_samples=1000):
                pass

        spans = exporter.get_finished_spans()
        ds = next(s for s in spans if s.name == "evalhub.evaluation.dataset_load")
        attrs = dict(ds.attributes or {})
        assert attrs["dataset.source"] == "hf://mmlu"
        assert attrs["dataset.size_samples"] == 1000


class TestInferenceSpanAttributes:
    def test_batch_attributes(
        self, exporter: InMemorySpanExporter, tracer: EvalTracer
    ) -> None:
        with tracer.evaluation_run():
            with tracer.inference_batch(
                model_id="llama-3",
                batch_size=32,
                tokens_input=4096,
                tokens_output=512,
            ):
                pass

        spans = exporter.get_finished_spans()
        inf = next(s for s in spans if s.name == "evalhub.evaluation.inference")
        attrs = dict(inf.attributes or {})
        assert attrs["model.id"] == "llama-3"
        assert attrs["batch.size"] == 32
        assert attrs["tokens.input"] == 4096
        assert attrs["tokens.output"] == 512


class TestScoringSpanAttributes:
    def test_benchmark_and_scorer(
        self, exporter: InMemorySpanExporter, tracer: EvalTracer
    ) -> None:
        with tracer.evaluation_run():
            with tracer.scoring(benchmark="mmlu", scorer="accuracy"):
                pass

        spans = exporter.get_finished_spans()
        sc = next(s for s in spans if s.name == "evalhub.evaluation.scoring")
        attrs = dict(sc.attributes or {})
        assert attrs["benchmark.name"] == "mmlu"
        assert attrs["scorer.type"] == "accuracy"


class TestW3cTraceparentExtraction:
    def test_parent_span_id_from_env(
        self, exporter: InMemorySpanExporter, tracer: EvalTracer
    ) -> None:
        with patch.dict(os.environ, {"TRACEPARENT": VALID_TRACEPARENT}):
            with tracer.evaluation_run():
                pass

        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name == "evalhub.evaluation.run")

        expected_trace_id = int("0af7651916cd43dd8448eb211c80319c", 16)
        expected_parent_span_id = int("b7ad6b7169203331", 16)

        assert root.context.trace_id == expected_trace_id
        assert root.parent is not None
        assert root.parent.span_id == expected_parent_span_id


@pytest.mark.no_otel_provider
class TestNoOtelConfiguredNoop:
    def test_noop_no_errors(self) -> None:
        tracer = EvalTracer()
        with tracer.evaluation_run():
            with tracer.dataset_load(source="x"):
                pass
            with tracer.inference_batch(batch_size=1):
                pass
            with tracer.scoring():
                pass
            with tracer.result_log():
                pass


class TestSpanErrorRecording:
    def test_exception_sets_error_status(
        self, exporter: InMemorySpanExporter, tracer: EvalTracer
    ) -> None:
        with pytest.raises(ValueError, match="boom"):
            with tracer.evaluation_run():
                with tracer.dataset_load():
                    raise ValueError("boom")

        spans = exporter.get_finished_spans()
        ds = next(s for s in spans if s.name == "evalhub.evaluation.dataset_load")
        assert ds.status.status_code == StatusCode.ERROR
        assert "boom" in (ds.status.description or "")

        events = ds.events
        assert any(e.name == "exception" for e in events)

        root = next(s for s in spans if s.name == "evalhub.evaluation.run")
        assert root.status.status_code == StatusCode.ERROR


class TestMultipleInferenceBatches:
    def test_three_batches_produce_three_spans(
        self, exporter: InMemorySpanExporter, tracer: EvalTracer
    ) -> None:
        with tracer.evaluation_run():
            for size in (16, 32, 64):
                with tracer.inference_batch(batch_size=size):
                    pass

        spans = exporter.get_finished_spans()
        inf_spans = [s for s in spans if s.name == "evalhub.evaluation.inference"]
        assert len(inf_spans) == 3
        sizes = sorted(int(s.attributes["batch.size"]) for s in inf_spans)
        assert sizes == [16, 32, 64]
