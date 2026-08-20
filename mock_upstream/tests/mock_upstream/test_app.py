"""Tests for the mock upstream's OpenAI-compatible endpoints.

Covers endpoint shape (via FastAPI's TestClient), determinism, the
configurable-latency helper, and end-to-end compatibility with the OpenAI SDK
(the same SDK the app's clients use) via an ASGI in-memory transport.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from httpx2 import ASGITransport, AsyncClient

import mock_upstream.app as mock_app_module
from core.types.depth_dive import InteractiveAnimation
from mock_upstream.app import (
    _CANNED_SCENE_GRAPH,
    _MOCKED_ANSWER,
    _MOCKED_CITATION_URL,
    app,
)
from mock_upstream.settings import settings as mock_settings

_MOCK_CHAT_PROMPT = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is attention?"}],
}


@pytest.fixture
def client() -> TestClient:
    """A TestClient against the mock upstream app."""
    return TestClient(app)


def test_embeddings_returns_deterministic_1536_dim_vectors(client: TestClient) -> None:
    """Same input yields the same unit-norm 1536-dim vector; distinct inputs differ."""
    response = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": ["hello", "world"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["model"] == "text-embedding-3-small"
    assert len(payload["data"]) == 2

    vectors = [item["embedding"] for item in payload["data"]]
    assert [len(vector) for vector in vectors] == [1536, 1536]
    assert vectors[0] != vectors[1]

    repeated = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": ["hello", "world"]},
    ).json()
    repeated_vectors = [item["embedding"] for item in repeated["data"]]
    assert repeated_vectors == vectors

    for vector in vectors:
        norm = sum(value * value for value in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9


def test_embeddings_accepts_single_string_input(client: TestClient) -> None:
    """The OpenAI ``input`` may be a bare string rather than a list."""
    response = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "single"},
    )
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_chat_completions_returns_deterministic_answer(client: TestClient) -> None:
    """Ordinary conversations get a fixed canned answer."""
    response = client.post("/v1/chat/completions", json=_MOCK_CHAT_PROMPT)
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == _MOCKED_ANSWER
    assert payload["choices"][0]["finish_reason"] == "stop"

    repeated = client.post("/v1/chat/completions", json=_MOCK_CHAT_PROMPT).json()
    assert repeated["choices"][0]["message"]["content"] == _MOCKED_ANSWER


def test_chat_completions_returns_valid_scene_graph_for_generation_turn(
    client: TestClient,
) -> None:
    """The Depth Dive generation turn gets a scene graph the app can parse."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You produce one interactive_animation scene graph.",
                },
                {"role": "user", "content": "Build an animation."},
            ],
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    animation = InteractiveAnimation.model_validate_json(content)
    assert animation == InteractiveAnimation.model_validate(_CANNED_SCENE_GRAPH)


def test_responses_returns_cited_web_search_result(client: TestClient) -> None:
    """/v1/responses carries a completed web_search_call with a url_citation."""
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-4o-mini", "tools": [{"type": "web_search"}], "input": "query here"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["status"] == "completed"

    search_calls = [item for item in payload["output"] if item["type"] == "web_search_call"]
    assert len(search_calls) == 1
    assert search_calls[0]["status"] == "completed"

    annotations = [
        annotation
        for item in payload["output"]
        if item["type"] == "message"
        for part in item["content"]
        for annotation in part["annotations"]
    ]
    assert len(annotations) == 1
    assert annotations[0]["type"] == "url_citation"
    assert annotations[0]["url"] == _MOCKED_CITATION_URL


def test_simulate_latency_disabled_skips_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled latency performs no sleep regardless of the configured range."""
    calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(mock_settings, "latency_enabled", False)
    asyncio.run(mock_app_module._simulate_latency(30, 80))
    assert calls == []


def test_simulate_latency_zero_max_skips_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-positive max disables the sleep for that endpoint."""
    calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(mock_settings, "latency_enabled", True)
    asyncio.run(mock_app_module._simulate_latency(0, 0))
    assert calls == []


def test_simulate_latency_sleeps_within_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled latency sleeps a uniform delay within the configured range."""
    calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(mock_settings, "latency_enabled", True)
    asyncio.run(mock_app_module._simulate_latency(30, 80))
    assert len(calls) == 1
    assert 0.03 <= calls[0] <= 0.08


def test_openai_sdk_parses_mock_endpoints() -> None:
    """The app's OpenAI SDK (async path) parses every mock endpoint directly.

    Uses an in-memory ASGI transport so the SDK talks to the mock without a
    real socket — the strongest shape-compatibility check short of a live run.
    """

    async def _run() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://mock") as http_client:
            from openai import AsyncOpenAI

            sdk = AsyncOpenAI(
                api_key="sk-test",
                base_url="http://mock/v1",
                http_client=http_client,
            )

            embedding = await sdk.embeddings.create(input=["hello"], model="text-embedding-3-small")
            assert len(embedding.data) == 1
            assert len(embedding.data[0].embedding) == 1536

            completion = await sdk.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )
            assert completion.choices[0].message.content == _MOCKED_ANSWER

            search = await sdk.responses.create(
                model="gpt-4o-mini",
                tools=[{"type": "web_search"}],
                input="attention is all you need",
            )
            assert search.status == "completed"

    asyncio.run(_run())


def test_unknown_endpoint_is_404(client: TestClient) -> None:
    """Endpoints the app never calls return 404, matching the real API surface."""
    response = client.post("/v1/unimplemented", json={})
    assert response.status_code == 404


def test_fault_injection_disabled_returns_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero error rate never fails a request, whatever the error status."""
    monkeypatch.setattr(mock_settings, "error_rate", 0.0)
    monkeypatch.setattr(mock_settings, "error_status", 502)

    response = client.post("/v1/chat/completions", json=_MOCK_CHAT_PROMPT)

    assert response.status_code == 200


def test_fault_injection_returns_configured_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 100% error rate returns the configured status on every endpoint.

    Load runs set ``MOCK_ERROR_RATE``/``MOCK_ERROR_STATUS`` to induce an
    upstream 502/503 storm that the api surfaces back to clients (issue #292).
    """
    monkeypatch.setattr(mock_settings, "error_rate", 1.0)
    monkeypatch.setattr(mock_settings, "error_status", 502)

    response = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": ["hello"]},
    )

    assert response.status_code == 502
    assert "detail" in response.json()


def test_determinism_helper_rejects_zero_dimension() -> None:
    """A non-positive dimension degrades to an empty vector, never a crash."""
    assert mock_app_module._embed_vector("x", 0, 0) == []
    assert mock_app_module._embed_vector("x", 0, -5) == []
