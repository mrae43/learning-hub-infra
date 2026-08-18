"""Tests for the stage-span decorator (:mod:`core.telemetry`).

The five RAG stage seams use :func:`stage_span` so every request's stages are
traceable (issue #286). These tests pin the decorator's contract: a named span
per call, error status + recorded exception on failure, async support, and
inert behaviour when no tracer provider is configured.
"""

import asyncio

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from core.telemetry import stage_span


def test_stage_span_records_named_span_on_success(
    in_memory_tracing: InMemorySpanExporter,
) -> None:
    """A successful seam call records one span carrying the stage name."""

    @stage_span("retrieve")
    def seam() -> str:
        return "result"

    assert seam() == "result"

    spans = in_memory_tracing.get_finished_spans()
    assert [s.name for s in spans] == ["retrieve"]


def test_stage_span_records_error_and_re_raises(
    in_memory_tracing: InMemorySpanExporter,
) -> None:
    """A failing seam call records the exception and propagates unchanged."""

    @stage_span("generate")
    def seam() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        seam()

    spans = in_memory_tracing.get_finished_spans()
    assert [s.name for s in spans] == ["generate"]
    assert spans[0].status.status_code == StatusCode.ERROR
    assert "exception" in [event.name for event in spans[0].events]


def test_stage_span_wraps_async_seams(in_memory_tracing: InMemorySpanExporter) -> None:
    """Async seam functions are wrapped in the same named span."""

    @stage_span("embed")
    async def seam() -> int:
        return 42

    assert asyncio.run(seam()) == 42

    spans = in_memory_tracing.get_finished_spans()
    assert [s.name for s in spans] == ["embed"]


def test_stage_span_is_inert_without_a_configured_provider() -> None:
    """Without a configured provider the span is a no-op and the call succeeds."""
    previous = otel_trace._TRACER_PROVIDER
    otel_trace._TRACER_PROVIDER = None
    try:

        @stage_span("embed")
        def seam() -> str:
            return "ok"

        assert seam() == "ok"
        current = otel_trace.get_current_span()
        assert not current.get_span_context().is_valid
    finally:
        otel_trace._TRACER_PROVIDER = previous
