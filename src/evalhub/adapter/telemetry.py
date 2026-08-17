"""OpenTelemetry span instrumentation for the adapter framework.

Provides ``EvalTracer``, which exposes context-manager methods for the five
evaluation span types defined by the EvalHub observability contract:

- ``evalhub.evaluation.run`` (root)
- ``evalhub.evaluation.dataset_load``
- ``evalhub.evaluation.inference``
- ``evalhub.evaluation.scoring``
- ``evalhub.evaluation.result_log``

When no ``TracerProvider`` is configured (i.e. OTEL environment variables are
absent), the underlying ``trace.get_tracer()`` returns a no-op tracer and all
context managers become zero-cost no-ops.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract

if TYPE_CHECKING:
    from .models.job import JobSpec

logger = logging.getLogger(__name__)

_TRACER_NAME = "evalhub.adapter"


class _EnvCarrier(dict[str, str]):
    """Read-only carrier that extracts W3C TraceContext headers from env vars.

    Implements ``Mapping[str, str]`` so the default OTEL ``DefaultGetter``
    can call ``.get(key)`` on it.  The EvalHub sidecar (or Kubernetes
    downward API) injects ``TRACEPARENT`` and optionally ``TRACESTATE``
    into the pod environment.
    """

    def __init__(self) -> None:
        data: dict[str, str] = {}
        for header in ("traceparent", "tracestate"):
            val = os.environ.get(header.upper())
            if val:
                data[header] = val
        super().__init__(data)


class EvalTracer:
    """Thin wrapper around an OTEL ``Tracer`` scoped to one evaluation job.

    This class is an **instrumentation library** and intentionally does *not*
    create or configure a ``TracerProvider``.  The host application (or the
    Kubernetes sidecar / OTEL auto-configuration) must set up a
    ``TracerProvider`` with the desired ``Resource``, exporter, and span
    processor *before* any spans are created.  When no provider is configured,
    ``trace.get_tracer()`` returns a no-op tracer and all context managers
    become zero-cost no-ops.

    Typical usage via ``DefaultCallbacks``::

        callbacks = DefaultCallbacks.from_adapter(adapter)

        with callbacks.tracer.evaluation_run():
            with callbacks.tracer.dataset_load(source="hf://mmlu", size_samples=500):
                ...
            for batch in batches:
                with callbacks.tracer.inference_batch(
                    model_id="llama-3", batch_size=32
                ):
                    ...
            with callbacks.tracer.scoring(benchmark="mmlu", scorer="accuracy"):
                ...
            with callbacks.tracer.result_log():
                callbacks.report_results(results)
    """

    def __init__(self) -> None:
        self._tracer = trace.get_tracer(_TRACER_NAME, schema_url=None)
        self._job_id: str | None = None
        self._provider: str | None = None
        self._collection: str | None = None
        self._model_id: str | None = None

    @classmethod
    def from_job_spec(cls, job_spec: JobSpec) -> EvalTracer:
        """Create a tracer pre-populated with job identity attributes."""
        tracer = cls()
        tracer._job_id = job_spec.id
        tracer._provider = job_spec.provider_id
        tracer._collection = job_spec.benchmark_id
        tracer._model_id = job_spec.model.name if job_spec.model else None
        return tracer

    @staticmethod
    def extract_context() -> otel_context.Context | None:
        """Extract W3C TraceContext from environment variables.

        Returns a ``Context`` carrying the remote span context injected by
        the EvalHub sidecar/server, or ``None`` if no traceparent is present.
        """
        ctx = extract(carrier=_EnvCarrier())
        span = trace.get_current_span(ctx)
        if span.get_span_context().trace_id == 0:
            return None
        return ctx

    @contextmanager
    def evaluation_run(self) -> Generator[trace.Span, None, None]:
        """Root span wrapping the full evaluation execution."""
        parent_ctx = self.extract_context()
        attrs: dict[str, str] = {}
        if self._job_id is not None:
            attrs["evalhub.job_id"] = self._job_id
        if self._provider is not None:
            attrs["evalhub.provider"] = self._provider
        if self._collection is not None:
            attrs["evalhub.collection"] = self._collection
        if self._model_id is not None:
            attrs["evalhub.model_id"] = self._model_id

        ctx = parent_ctx if parent_ctx is not None else otel_context.get_current()
        with self._tracer.start_as_current_span(
            "evalhub.evaluation.run",
            context=ctx,
            attributes=attrs,
        ) as span:
            yield span

    @contextmanager
    def dataset_load(
        self,
        source: str | None = None,
        size_samples: int | None = None,
    ) -> Generator[trace.Span, None, None]:
        """Child span for the dataset loading phase."""
        attrs: dict[str, str | int] = {}
        if source is not None:
            attrs["dataset.source"] = source
        if size_samples is not None:
            attrs["dataset.size_samples"] = size_samples

        with self._tracer.start_as_current_span(
            "evalhub.evaluation.dataset_load",
            attributes=attrs,
        ) as span:
            yield span

    @contextmanager
    def inference_batch(
        self,
        model_id: str | None = None,
        batch_size: int | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
    ) -> Generator[trace.Span, None, None]:
        """Child span for a single inference batch."""
        attrs: dict[str, str | int] = {}
        if model_id is not None:
            attrs["model.id"] = model_id
        if batch_size is not None:
            attrs["batch.size"] = batch_size
        if tokens_input is not None:
            attrs["tokens.input"] = tokens_input
        if tokens_output is not None:
            attrs["tokens.output"] = tokens_output

        with self._tracer.start_as_current_span(
            "evalhub.evaluation.inference",
            attributes=attrs,
        ) as span:
            yield span

    @contextmanager
    def scoring(
        self,
        benchmark: str | None = None,
        scorer: str | None = None,
    ) -> Generator[trace.Span, None, None]:
        """Child span for the scoring / post-processing phase."""
        attrs: dict[str, str] = {}
        if benchmark is not None:
            attrs["benchmark.name"] = benchmark
        if scorer is not None:
            attrs["scorer.type"] = scorer

        with self._tracer.start_as_current_span(
            "evalhub.evaluation.scoring",
            attributes=attrs,
        ) as span:
            yield span

    @contextmanager
    def result_log(self) -> Generator[trace.Span, None, None]:
        """Child span for result logging to MLflow and EvalHub service."""
        with self._tracer.start_as_current_span(
            "evalhub.evaluation.result_log",
        ) as span:
            yield span
