"""Tests for the retry-once web-search wrapper (ADR-0013, ticket #244).

The provider is stubbed so the retry/fallback policy is exercised without a
hosted API (coding-standards.md): success, single-retry recovery,
double-failure fallback, and the no-search cases.
"""

from collections.abc import Sequence

import pytest

from depth_dive.web_search.client import StubWebSearchClient, WebSearchResult
from depth_dive.web_search.wrapper import FALLBACK_NOTE, SearchOutcome, run_web_search


def _result(title: str) -> WebSearchResult:
    return WebSearchResult(title=title, url=f"https://example.com/{title}", snippet=f"{title} span")


def _run(
    query: str,
    *,
    results: Sequence[WebSearchResult] | None = None,
    failures: int = 0,
) -> tuple[SearchOutcome, StubWebSearchClient]:
    client = StubWebSearchClient(results=results, failures=failures)
    return run_web_search(query, client), client


def test_no_search_when_query_is_empty() -> None:
    """An empty query is not a search attempt at all."""
    outcome, client = _run("")
    assert outcome.attempted is False
    assert outcome.failed is False
    assert outcome.note is None
    assert outcome.results == []
    assert client.calls == []


def test_no_search_when_query_is_whitespace() -> None:
    """A whitespace-only query is not a search attempt."""
    outcome, client = _run("   ")
    assert outcome.attempted is False
    assert client.calls == []


def test_success_returns_results_without_retry() -> None:
    """A successful search returns its results on the first (only) attempt."""
    results = [_result("attention")]
    outcome, client = _run("attention is all you need", results=results)
    assert outcome.attempted is True
    assert outcome.failed is False
    assert outcome.note is None
    assert outcome.results == results
    assert client.calls == ["attention is all you need"]


def test_first_failure_recovers_on_the_retry() -> None:
    """A failed attempt is retried exactly once and recovers with results."""
    results = [_result("transformer")]
    outcome, client = _run("transformer architecture", results=results, failures=1)
    assert outcome.attempted is True
    assert outcome.failed is False
    assert outcome.note is None
    assert outcome.results == results
    assert len(client.calls) == 2


def test_double_failure_falls_back_with_note() -> None:
    """Two consecutive failures produce a failed outcome with a note, no retry."""
    outcome, client = _run("nonsense query", failures=2)
    assert outcome.attempted is True
    assert outcome.failed is True
    assert outcome.note == FALLBACK_NOTE
    assert outcome.results == []
    assert len(client.calls) == 2


def test_failure_then_empty_results_is_a_fallback() -> None:
    """A failed retry that returns nothing still falls back (no useful results)."""
    outcome, client = _run("failing then empty", failures=1)
    assert outcome.attempted is True
    assert outcome.failed is True
    assert outcome.note is not None
    assert outcome.results == []
    assert len(client.calls) == 2


def test_empty_first_result_is_retried() -> None:
    """No useful results on the first attempt counts as a failure and is retried."""
    results = [_result("recovered")]

    class _EmptyThenFullClient:
        calls: int = 0

        def search(self, query: str) -> list[WebSearchResult]:
            self.calls += 1
            return [] if self.calls == 1 else list(results)

    client = _EmptyThenFullClient()
    outcome = run_web_search("first empty then full", client)
    assert outcome.attempted is True
    assert outcome.failed is False
    assert outcome.results == results
    assert client.calls == 2


def test_unexpected_provider_error_is_not_caught() -> None:
    """Non-``WebSearchError`` exceptions are not swallowed by the wrapper."""

    class _BoomClient:
        def search(self, query: str) -> list[WebSearchResult]:
            raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError):
        run_web_search("anything", _BoomClient())
