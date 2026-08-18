"""Tests for the mock upstream's environment configuration."""

import pytest

from mock_upstream.settings import MockUpstreamSettings


def test_defaults() -> None:
    """Latency is on by default with the documented per-endpoint ranges."""
    config = MockUpstreamSettings()
    assert config.latency_enabled is True
    assert config.embeddings_latency_min_ms == 30
    assert config.embeddings_latency_max_ms == 80
    assert config.chat_latency_min_ms == 150
    assert config.chat_latency_max_ms == 400
    assert config.web_search_latency_min_ms == 150
    assert config.web_search_latency_max_ms == 400


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every knob is configurable via the MOCK_* environment variables."""
    monkeypatch.setenv("MOCK_LATENCY_ENABLED", "false")
    monkeypatch.setenv("MOCK_CHAT_LATENCY_MIN_MS", "0")
    monkeypatch.setenv("MOCK_CHAT_LATENCY_MAX_MS", "0")
    config = MockUpstreamSettings()
    assert config.latency_enabled is False
    assert config.chat_latency_min_ms == 0
    assert config.chat_latency_max_ms == 0


def test_latency_can_be_zeroed_via_max_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zeroing a range's max alone disables that endpoint's simulated latency."""
    monkeypatch.setenv("MOCK_WEB_SEARCH_LATENCY_MAX_MS", "0")
    config = MockUpstreamSettings()
    assert config.web_search_latency_max_ms == 0
