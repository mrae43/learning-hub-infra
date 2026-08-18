"""Tests for the Cohere reranker's error mapping.

Verifies that only genuinely "upstream unavailable" conditions (timeouts,
connection failures) are relabelled as ``UpstreamUnavailable``, while
non-upstream bugs propagate unchanged and HTTP status errors keep their
dedicated mapping. See issue #178.
"""

from unittest.mock import MagicMock
from uuid import uuid4

import cohere
import cohere.core.api_error
import cohere.errors
import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from core.clients.reranker_client import CohereReranker
from core.exceptions import RerankerRateLimitError, UpstreamBadResponse, UpstreamUnavailable
from core.types.responses import ScoredChunk


def _passages() -> list[ScoredChunk]:
    return [ScoredChunk(chunk_id=uuid4(), text=f"passage {i}", score=1.0) for i in range(3)]


def _reranker_with(raise_exc: Exception) -> CohereReranker:
    reranker = CohereReranker(api_key="test-key", model="rerank-test")
    client = MagicMock()
    client.rerank.side_effect = raise_exc
    # Override the lazy-init accessor so no real Cohere client is constructed;
    # the return type is the protocol's client type but we substitute a mock.
    reranker._get_client = lambda: client  # type: ignore[method-assign]
    return reranker


def test_rerank_records_rerank_stage_span(in_memory_tracing: InMemorySpanExporter) -> None:
    """The rerank seam records a ``rerank`` stage span (issue #286)."""
    reranker = CohereReranker(api_key="test-key", model="rerank-test")
    fake = MagicMock()
    result = MagicMock()
    result.index = 0
    fake.rerank.return_value = MagicMock(results=[result])
    # Override the lazy-init accessor so no real Cohere client is constructed.
    reranker._get_client = lambda: fake  # type: ignore[method-assign]

    passages = _passages()
    reranked = reranker.rerank("query", passages, top_k=2)

    assert reranked == [passages[0]]
    spans = in_memory_tracing.get_finished_spans()
    assert [s.name for s in spans] == ["rerank"]


def test_rerank_connection_error_raises_upstream_unavailable() -> None:
    """A connection failure means the upstream is unreachable (503)."""
    reranker = _reranker_with(httpx.ConnectError("connection refused"))
    with pytest.raises(UpstreamUnavailable):
        reranker.rerank("query", _passages(), top_k=2)


def test_rerank_timeout_raises_upstream_unavailable() -> None:
    """A read timeout means the upstream is unreachable (503)."""
    reranker = _reranker_with(httpx.ReadTimeout("timed out"))
    with pytest.raises(UpstreamUnavailable):
        reranker.rerank("query", _passages(), top_k=2)


def test_rerank_protocol_error_raises_upstream_unavailable() -> None:
    """A remote protocol error means the upstream is unreachable (503)."""
    reranker = _reranker_with(httpx.RemoteProtocolError("connection closed"))
    with pytest.raises(UpstreamUnavailable):
        reranker.rerank("query", _passages(), top_k=2)


def test_rerank_non_upstream_bug_propagates_unchanged() -> None:
    """A genuine bug inside the call (not a transport error) must not be
    relabelled as ``UpstreamUnavailable``; it propagates as-is."""
    reranker = _reranker_with(RuntimeError("internal programming bug"))
    with pytest.raises(RuntimeError, match="internal programming bug"):
        reranker.rerank("query", _passages(), top_k=2)


def test_rerank_attribute_error_propagates_unchanged() -> None:
    """An attribute error on the SDK result must not surface as 503."""
    reranker = _reranker_with(AttributeError("missing attribute bug"))
    with pytest.raises(AttributeError, match="missing attribute bug"):
        reranker.rerank("query", _passages(), top_k=2)


def test_rerank_rate_limit_raises_reranker_rate_limit_error() -> None:
    """A 429 from Cohere still maps to RerankerRateLimitError."""
    reranker = _reranker_with(cohere.errors.TooManyRequestsError(body={}))
    with pytest.raises(RerankerRateLimitError):
        reranker.rerank("query", _passages(), top_k=2)


def test_rerank_other_api_error_raises_upstream_bad_response() -> None:
    """A non-429 API error (4xx/5xx) still maps to UpstreamBadResponse."""
    reranker = _reranker_with(cohere.core.api_error.ApiError(status_code=400, body={}))
    with pytest.raises(UpstreamBadResponse):
        reranker.rerank("query", _passages(), top_k=2)
