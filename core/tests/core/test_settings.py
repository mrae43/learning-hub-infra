"""Tests for project settings, focusing on the OpenAI base-URL override."""

import pytest

from core.config.settings import Settings


def test_openai_base_url_defaults_to_none() -> None:
    """Without configuration the base URL is None (SDK default endpoint)."""
    assert Settings().openai_base_url is None


def test_openai_base_url_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_BASE_URL populates the setting."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://mock-upstream:8080/v1")
    assert Settings().openai_base_url == "http://mock-upstream:8080/v1"


def test_empty_openai_base_url_coerces_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty OPENAI_BASE_URL (compose ${VAR:-}) means 'not configured'."""
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    assert Settings().openai_base_url is None


def test_empty_openai_api_key_coerces_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty OPENAI_API_KEY (compose ${VAR:-}) means 'not configured'."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert Settings().openai_api_key is None
