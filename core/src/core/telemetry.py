"""Manual OpenTelemetry stage spans for the hand-rolled RAG pipeline.

The five RAG stages — embed, retrieve, rerank, generate, web-search — are
wrapped at their existing seams with :func:`stage_span` so every request's
stages are traceable (issue #286). No pipeline behaviour changes: the spans
record the call, its duration, and its outcome; failures propagate unchanged.

The instrumentation is **inert by default**: spans flow through whatever
tracer provider the process has configured, and ``opentelemetry`` supplies a
no-op tracer until a provider is installed. Library packages never configure
the SDK here — the exporter is owned by the api process, the single export
point (``api.telemetry.configure_telemetry``). The tracer is resolved per call
so a provider installed after import time (the api does this at app creation)
takes effect immediately.
"""

from collections.abc import Callable
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Literal, ParamSpec, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

P = ParamSpec("P")
T = TypeVar("T")

StageName = Literal["embed", "retrieve", "rerank", "generate", "web-search"]
"""The five RAG stage names the stage spans carry (issue #286)."""

TRACER_NAME = "learning_hub"
"""Instrumenting scope shared by the stage spans and the api's request root span."""


def _mark_span_failed(span: Span, exc: Exception) -> None:
    """Record an exception on a span and set its status to ERROR.

    Args:
        span: The span that observed the failure.
        exc: The exception that escaped the seam call.
    """
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def stage_span(name: StageName) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate a pipeline seam so each call runs inside a span named ``name``.

    Wraps the function's body in a manual span: the span carries the stage
    name, the stage's duration, and — on failure — the recorded exception with
    an error status. The wrapped call's behaviour is unchanged; sync and async
    functions are both supported.

    Args:
        name: The stage name the span carries.

    Returns:
        A decorator that wraps the seam function.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                with trace.get_tracer(TRACER_NAME).start_as_current_span(name) as span:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        _mark_span_failed(span, exc)
                        raise

            return cast(Callable[P, T], async_wrapper)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with trace.get_tracer(TRACER_NAME).start_as_current_span(name) as span:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    _mark_span_failed(span, exc)
                    raise

        return wrapper

    return decorator


__all__ = ["TRACER_NAME", "StageName", "stage_span"]
