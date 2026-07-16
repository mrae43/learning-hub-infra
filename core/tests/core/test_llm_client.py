"""Unit tests for the LLM client (hosted OpenAI API mocked)."""

from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, APIStatusError

from core.clients.llm_client import LLMClient
from core.exceptions import UpstreamBadResponse, UpstreamUnavailable


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
        messages=[{"role": "user", "content": "What is the answer?"}],
    )

    assert result == "The answer is 42."
    fake_openai.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the answer?"}],
    )


def test_chat_uses_default_model_from_settings() -> None:
    """LLMClient picks up settings.inference_model by default."""
    client = LLMClient(api_key="sk-test")
    assert client._model == "gpt-4o-mini"


def test_chat_raises_upstream_unavailable_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An APIConnectionError surfaces as UpstreamUnavailable (maps to 503)."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")

    fake_openai = MagicMock()
    fake_openai.chat.completions.create.side_effect = _fake_connection_error("conn refused")
    monkeypatch.setattr(client, "_get_client", lambda: fake_openai)

    with pytest.raises(UpstreamUnavailable):
        client.chat(messages=[{"role": "user", "content": "x"}])


def test_chat_raises_upstream_bad_response_on_api_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An APIStatusError (bad HTTP status) surfaces as UpstreamBadResponse (maps to 502)."""
    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")

    fake_openai = MagicMock()
    fake_openai.chat.completions.create.side_effect = _fake_status_error("500 Server Error")
    monkeypatch.setattr(client, "_get_client", lambda: fake_openai)

    with pytest.raises(UpstreamBadResponse):
        client.chat(messages=[{"role": "user", "content": "x"}])


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
        client.chat(messages=[{"role": "user", "content": "x"}])


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
        client.chat(messages=[{"role": "user", "content": "x"}])
