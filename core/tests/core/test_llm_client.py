"""Unit tests for the LLM client (hosted OpenAI API mocked)."""

from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, APIStatusError
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from core.clients import llm_client as llm_module
from core.clients.llm_client import CompletionProvider, LLMClient, MockCompletionProvider
from core.config.settings import settings
from core.exceptions import UpstreamBadResponse, UpstreamUnavailable
from core.types.chat import (
    ChatContentImagePart,
    ChatContentImageURL,
    ChatContentTextPart,
    ChatMessage,
)


def _fake_completion(content: str) -> MagicMock:
    """Build a fake ``completion`` object with ``choices[0].message.content``."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    return completion


def _fake_status_error(message: str = "bad response") -> APIStatusError:
    """Build a real-ish APIStatusError (bad HTTP status → 502)."""
    response = MagicMock()
    response.status_code = 500
    response.headers = {}
    body = {"error": {"message": message}}
    return APIStatusError(message, response=response, body=body)


def _fake_connection_error(message: str = "timed out") -> APIConnectionError:
    """Build an APIConnectionError (network/timeout → 503)."""
    request = MagicMock()
    request.method = "POST"
    request.url = "https://api.openai.com/v1/chat/completions"
    return APIConnectionError(message=message, request=request)


def test_chat_returns_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat() returns the first choice's message content."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")

    fake_openai = MagicMock()
    fake_openai.chat.completions.create = MagicMock(
        return_value=_fake_completion("The answer is 42.")
    )
    monkeypatch.setattr(client, "_get_client", lambda: fake_openai)

    result = client.chat(
        messages=[ChatMessage(role="user", content="What is the answer?")],
    )

    assert result == "The answer is 42."
    fake_openai.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the answer?"}],
    )


def test_chat_serializes_multimodal_content_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A message with image content serializes to the OpenAI content-part shape."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")

    fake_openai = MagicMock()
    fake_openai.chat.completions.create = MagicMock(return_value=_fake_completion("Understood."))
    monkeypatch.setattr(client, "_get_client", lambda: fake_openai)

    message = ChatMessage(
        role="user",
        content=[
            ChatContentTextPart(text="Here is the figure:"),
            ChatContentImagePart(
                image_url=ChatContentImageURL(url="data:image/png;base64,aGVsbG8=")
            ),
        ],
    )
    client.chat(messages=[message])

    fake_openai.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the figure:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            }
        ],
    )


def test_chat_records_generate_stage_span(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_tracing: InMemorySpanExporter,
) -> None:
    """The generate seam records a ``generate`` stage span (issue #286)."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")

    fake_openai = MagicMock()
    fake_openai.chat.completions.create = MagicMock(return_value=_fake_completion("answer"))
    monkeypatch.setattr(client, "_get_client", lambda: fake_openai)

    client.chat([ChatMessage(role="user", content="x")])

    spans = in_memory_tracing.get_finished_spans()
    assert [s.name for s in spans] == ["generate"]


def test_chat_uses_default_model_from_settings() -> None:
    """LLMClient picks up settings.inference_model by default."""
    client = LLMClient(api_key="sk-test")
    assert client._model == "gpt-4o-mini"


def test_sync_client_threads_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAI client is constructed with the caller's base_url."""
    captured: dict[str, object] = {}

    def _fake_openai(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        client = MagicMock()
        client.chat.completions.create = MagicMock(return_value=_fake_completion("answer"))
        return client

    monkeypatch.setattr(llm_module, "OpenAI", _fake_openai)
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini", base_url="http://mock/v1")
    client.chat([ChatMessage(role="user", content="x")])

    assert captured["base_url"] == "http://mock/v1"


def test_sync_client_threads_settings_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit base_url, settings.openai_base_url is used."""
    captured: dict[str, object] = {}

    def _fake_openai(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        client = MagicMock()
        client.chat.completions.create = MagicMock(return_value=_fake_completion("answer"))
        return client

    monkeypatch.setattr(settings, "openai_base_url", "http://mock/v1")
    monkeypatch.setattr(llm_module, "OpenAI", _fake_openai)
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")
    client.chat([ChatMessage(role="user", content="x")])

    assert captured["base_url"] == "http://mock/v1"


def test_chat_raises_upstream_unavailable_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An APIConnectionError surfaces as UpstreamUnavailable (maps to 503)."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")

    fake_openai = MagicMock()
    fake_openai.chat.completions.create.side_effect = _fake_connection_error("conn refused")
    monkeypatch.setattr(client, "_get_client", lambda: fake_openai)

    with pytest.raises(UpstreamUnavailable):
        client.chat(messages=[ChatMessage(role="user", content="x")])


def test_chat_raises_upstream_bad_response_on_api_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An APIStatusError (bad HTTP status) surfaces as UpstreamBadResponse (maps to 502)."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")

    fake_openai = MagicMock()
    fake_openai.chat.completions.create.side_effect = _fake_status_error("500 Server Error")
    monkeypatch.setattr(client, "_get_client", lambda: fake_openai)

    with pytest.raises(UpstreamBadResponse):
        client.chat(messages=[ChatMessage(role="user", content="x")])


def test_chat_raises_upstream_bad_response_when_choices_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected response shape surfaces as UpstreamBadResponse (maps to 502)."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")

    completion = MagicMock()
    completion.choices = []  # malformed response
    fake_openai = MagicMock()
    fake_openai.chat.completions.create = MagicMock(return_value=completion)
    monkeypatch.setattr(client, "_get_client", lambda: fake_openai)

    with pytest.raises(UpstreamBadResponse):
        client.chat(messages=[ChatMessage(role="user", content="x")])


def test_chat_raises_upstream_bad_response_when_content_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A None content on the returned message surfaces as UpstreamBadResponse."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")

    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = None
    fake_openai = MagicMock()
    fake_openai.chat.completions.create = MagicMock(return_value=completion)
    monkeypatch.setattr(client, "_get_client", lambda: fake_openai)

    with pytest.raises(UpstreamBadResponse):
        client.chat(messages=[ChatMessage(role="user", content="x")])


def test_llm_client_satisfies_completion_provider_protocol() -> None:
    """LLMClient is a structural ``CompletionProvider`` implementation."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")
    assert isinstance(client, CompletionProvider)


def test_mock_completion_provider_returns_configured_response() -> None:
    """MockCompletionProvider returns the response configured at construction."""
    provider = MockCompletionProvider("Fixed answer.")
    result = provider.chat([ChatMessage(role="user", content="question")])
    assert result == "Fixed answer."


def test_mock_completion_provider_satisfies_protocol() -> None:
    """MockCompletionProvider is a structural ``CompletionProvider`` implementation."""
    provider = MockCompletionProvider()
    assert isinstance(provider, CompletionProvider)
