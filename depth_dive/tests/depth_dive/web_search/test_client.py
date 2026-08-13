"""Tests for the web-search provider clients (ADR-0012, ADR-0013).

The ``OpenAIWebSearchClient`` is exercised against hand-built ``Response``
objects so parsing and error mapping are covered without a hosted API; the
``StubWebSearchClient`` provides the deterministic in-memory stand-in used by
the wrapper and harness tests.
"""

from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, APIStatusError, OpenAIError
from openai.types.responses import Response

from depth_dive.web_search.client import (
    OpenAIWebSearchClient,
    StubWebSearchClient,
    WebSearchClient,
    WebSearchError,
    WebSearchResult,
)


def _response_payload(
    *,
    annotation: dict[str, object] | None,
    web_search_status: str = "completed",
) -> dict[str, object]:
    """Build a JSON-shaped Responses API payload for the web search tool."""
    output: list[dict[str, object]] = [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": web_search_status,
            "action": {"type": "search", "query": "attention is all you need"},
        }
    ]
    if annotation is not None:
        output.append(
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The Transformer paper is famous. See the original.",
                        "annotations": [annotation],
                    }
                ],
            }
        )
    return {
        "id": "resp_test",
        "created_at": 1,
        "object": "response",
        "model": "gpt-4o-mini",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [{"type": "web_search"}],
        "output": output,
    }


def _client(response: Response | None = None) -> tuple[OpenAIWebSearchClient, MagicMock]:
    client = OpenAIWebSearchClient(api_key="test-key", model="test-model")
    fake = MagicMock()
    if response is not None:
        fake.responses.create.return_value = response
    client._client = fake
    return client, fake


def test_openai_client_surfaces_cited_urls_as_results() -> None:
    """url_citation annotations become title/url/snippet results."""
    response = Response.model_validate(
        _response_payload(
            annotation={
                "type": "url_citation",
                "url": "https://example.com/paper",
                "title": "Attention Is All You Need",
                "start_index": 24,
                "end_index": 44,
            }
        )
    )
    client, _ = _client(response)
    results = client.search("attention is all you need")
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, WebSearchResult)
    assert result.title == "Attention Is All You Need"
    assert result.url == "https://example.com/paper"
    assert result.snippet == " famous. See the ori"


def test_openai_client_empty_citations_yields_no_results() -> None:
    """A response with no url_citation annotations surfaces no results."""
    response = Response.model_validate(_response_payload(annotation=None))
    client, _ = _client(response)
    assert client.search("attention is all you need") == []


def test_openai_client_skips_out_of_bounds_annotations() -> None:
    """Annotations whose indices exceed the part text are skipped, not raised."""
    response = Response.model_validate(
        _response_payload(
            annotation={
                "type": "url_citation",
                "url": "https://example.com/paper",
                "title": "Attention Is All You Need",
                "start_index": 5,
                "end_index": 999,
            }
        )
    )
    client, _ = _client(response)
    assert client.search("attention is all you need") == []


def test_openai_client_failed_search_call_raises() -> None:
    """A web_search_call whose status is failed is a failed search."""
    response = Response.model_validate(
        _response_payload(annotation=None, web_search_status="failed")
    )
    client, _ = _client(response)
    with pytest.raises(WebSearchError):
        client.search("attention is all you need")


def test_openai_client_connection_error_maps_to_web_search_error() -> None:
    """An unreachable API raises WebSearchError, not a RetrievalError subclass."""
    client, fake = _client()
    fake.responses.create.side_effect = APIConnectionError(request=MagicMock())
    with pytest.raises(WebSearchError):
        client.search("attention is all you need")


def test_openai_client_missing_key_maps_to_web_search_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client without a usable API key fails gracefully as WebSearchError."""

    def _raise(*, api_key: str | None = None, **_: object) -> None:
        raise OpenAIError("The api_key client option must be set")

    monkeypatch.setattr("depth_dive.web_search.client.OpenAI", _raise)
    client = OpenAIWebSearchClient(api_key=None, model="test-model")
    with pytest.raises(WebSearchError):
        client.search("attention is all you need")


def test_openai_client_bad_status_maps_to_web_search_error() -> None:
    """A 4xx/5xx API status raises WebSearchError."""
    client, fake = _client()
    fake.responses.create.side_effect = APIStatusError("bad", response=MagicMock(), body=None)
    with pytest.raises(WebSearchError):
        client.search("attention is all you need")


def test_openai_client_uses_web_search_tool() -> None:
    """The search issues one response with the web_search tool enabled."""
    response = Response.model_validate(_response_payload(annotation=None))
    client, fake = _client(response)
    client.search("attention is all you need")
    fake.responses.create.assert_called_once()
    kwargs = fake.responses.create.call_args.kwargs
    assert kwargs["input"] == "attention is all you need"
    assert kwargs["tools"] == [{"type": "web_search"}]


def test_stub_client_returns_configured_results() -> None:
    """The stub returns its configured results and records queries."""
    results = [WebSearchResult(title="t", url="https://example.com", snippet="s")]
    client = StubWebSearchClient(results=results)
    assert client.search("query one") == results
    assert client.search("query two") == results
    assert client.calls == ["query one", "query two"]


def test_stub_client_fails_configured_number_of_times() -> None:
    """The stub raises WebSearchError for the configured failure count."""
    client = StubWebSearchClient(results=[], failures=2)
    with pytest.raises(WebSearchError):
        client.search("q")
    with pytest.raises(WebSearchError):
        client.search("q")
    assert client.search("q") == []


def test_openai_client_conforms_to_protocol() -> None:
    """The concrete client satisfies the WebSearchClient protocol."""
    client, _ = _client()
    assert isinstance(client, WebSearchClient)


def test_stub_client_conforms_to_protocol() -> None:
    """The stub satisfies the WebSearchClient protocol."""
    assert isinstance(StubWebSearchClient(), WebSearchClient)
