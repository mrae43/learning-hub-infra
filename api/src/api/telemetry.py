"""OTLP/HTTP export wiring for the api process (issue #286).

The api is the single export point for the five RAG stage spans
(embed / retrieve / rerank / generate / web-search). When
``Settings.otel_exporter_otlp_endpoint`` is configured, :func:`configure_telemetry`
installs a global ``TracerProvider`` that batch-exports spans over OTLP/HTTP to
that collector. Otherwise it leaves the OpenTelemetry default no-op provider in
place, so the stage spans created by :mod:`core.telemetry` are dropped — the
instrumentation is inert unless explicitly enabled by configuration.

:class:`TraceRequestMiddleware` opens one root span per request so the stage
spans nest under a single per-request trace instead of minting one orphaned
root per stage, and :func:`shutdown_telemetry` flushes the final batch on
graceful process shutdown.
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind
from starlette.types import ASGIApp, Receive, Scope, Send

from core.config.settings import Settings
from core.telemetry import TRACER_NAME

_provider: TracerProvider | None = None
"""The provider this module installed; shut down on process exit."""


def configure_telemetry(
    settings: Settings,
    exporter: SpanExporter | None = None,
) -> bool:
    """Install the global OTLP/HTTP tracer provider when configured.

    Args:
        settings: Application settings. Export is enabled only when
            ``otel_exporter_otlp_endpoint`` is set; ``otel_service_name``
            seeds the ``service.name`` resource attribute on exported spans.
        exporter: Optional span exporter override. Defaults to an
            ``OTLPSpanExporter`` pointed at the configured endpoint; tests
            pass an in-memory exporter to exercise the wiring without network.

    Returns:
        True when the OTLP exporter was installed (export enabled); False when
        the instrumentation was left inert (no endpoint configured).
    """
    global _provider
    endpoint = settings.otel_exporter_otlp_endpoint
    if not endpoint:
        return False
    resource = Resource.create({"service.name": settings.otel_service_name})
    _provider = TracerProvider(resource=resource)
    _provider.add_span_processor(
        BatchSpanProcessor(exporter or OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(_provider)
    return True


def shutdown_telemetry() -> None:
    """Flush and stop the api's span exporter on process shutdown.

    ``BatchSpanProcessor`` exports asynchronously every few seconds; without
    a shutdown the final request's spans could be dropped when the process
    exits. A no-op when no exporter was installed.
    """
    provider = _provider
    if provider is not None:
        provider.shutdown()


class TraceRequestMiddleware:
    """Start one root span per request so stage spans share a trace.

    Each request opens a server-kind span (``GET /health``, ``POST /query``,
    ...) under which the five stage spans nest, so a collector renders a
    single per-request trace instead of five orphaned roots. Inert by default:
    without a configured provider the span is a no-op and the request is
    unaffected.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI application.

        Args:
            app: The ASGI application this middleware sits in front of.
        """
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one request, spanning its full duration."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        path = scope["path"]
        method = scope["method"]
        with trace.get_tracer(TRACER_NAME).start_as_current_span(
            f"{method} {path}",
            kind=SpanKind.SERVER,
        ) as span:
            span.set_attribute("http.request.method", method)
            span.set_attribute("url.path", path)
            await self._app(scope, receive, send)


__all__ = ["TraceRequestMiddleware", "configure_telemetry", "shutdown_telemetry"]
