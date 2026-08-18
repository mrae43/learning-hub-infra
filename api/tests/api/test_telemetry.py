"""Tests for the api's OTLP export wiring (:mod:`api.telemetry`).

``configure_telemetry`` is the switch that keeps the five RAG stage spans
(issue #286) inert by default: without an ``otel_exporter_otlp_endpoint`` the
process exports nothing; with one, a global ``TracerProvider`` is installed
that exports every stage span.
"""

from fastapi.testclient import TestClient
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from api.telemetry import _otlp_traces_endpoint, configure_telemetry, shutdown_telemetry
from core.config.settings import Settings
from core.telemetry import stage_span


def test_otlp_traces_endpoint_appends_the_signal_path() -> None:
    """A base collector URL gains the OTLP/HTTP traces path.

    ``OTLPSpanExporter`` treats an explicit ``endpoint`` as the full URL, so
    the configured base URL must carry ``/v1/traces`` itself — mirroring the
    SDK's ``OTEL_EXPORTER_OTLP_ENDPOINT`` env-var behaviour (issue #289).
    """
    assert (
        _otlp_traces_endpoint("http://otel-collector:4318")
        == "http://otel-collector:4318/v1/traces"
    )
    assert (
        _otlp_traces_endpoint("http://otel-collector:4318/")
        == "http://otel-collector:4318/v1/traces"
    )


def test_configure_telemetry_stays_inert_without_an_endpoint() -> None:
    """Without an endpoint the provider is untouched and no exporter is set up."""
    previous = otel_trace._TRACER_PROVIDER
    try:
        result = configure_telemetry(Settings(otel_exporter_otlp_endpoint=None))

        assert result is False
        assert otel_trace._TRACER_PROVIDER is previous
    finally:
        otel_trace._TRACER_PROVIDER = previous


def test_configure_telemetry_installs_provider_and_exports_stage_spans() -> None:
    """With an endpoint the provider carries the service name and exports spans."""
    previous = otel_trace._TRACER_PROVIDER
    exporter = InMemorySpanExporter()
    provider: TracerProvider | None = None
    try:
        result = configure_telemetry(
            Settings(
                otel_exporter_otlp_endpoint="http://collector:4318",
                otel_service_name="test-api",
            ),
            exporter=exporter,
        )

        assert result is True
        installed = otel_trace._TRACER_PROVIDER
        assert isinstance(installed, TracerProvider)
        provider = installed
        assert provider.resource.attributes["service.name"] == "test-api"

        @stage_span("embed")
        def seam() -> None:
            return None

        seam()
        provider.force_flush()

        assert [s.name for s in exporter.get_finished_spans()] == ["embed"]
    finally:
        otel_trace._TRACER_PROVIDER = previous
        if provider is not None:
            provider.shutdown()


def test_create_app_adds_request_root_span(
    client: TestClient,
    in_memory_tracing: InMemorySpanExporter,
) -> None:
    """Each request opens a server root span so stage spans share one trace."""
    response = client.get("/health")

    assert response.status_code == 200
    spans = in_memory_tracing.get_finished_spans()
    assert [s.name for s in spans] == ["GET /health"]
    assert spans[0].kind == SpanKind.SERVER
    attributes = spans[0].attributes
    assert attributes is not None
    assert attributes["http.request.method"] == "GET"
    assert attributes["url.path"] == "/health"


def test_shutdown_telemetry_is_a_noop_without_an_exporter() -> None:
    """Shutdown never raises when no exporter was installed."""
    shutdown_telemetry()
