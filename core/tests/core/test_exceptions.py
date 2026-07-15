"""Tests for the named exception hierarchy."""

import pytest

from core.exceptions import (
    IngestionError,
    LearningHubError,
    RetrievalError,
    UpstreamBadResponse,
    UpstreamUnavailable,
)


def test_learninghuberror_is_base_exception() -> None:
    """LearningHubError inherits from Exception and is catchable."""
    assert issubclass(LearningHubError, Exception)
    with pytest.raises(LearningHubError):
        raise LearningHubError("boom")


def test_ingestionerror_inherits_learninghuberror() -> None:
    """IngestionError is a subclass of LearningHubError."""
    assert issubclass(IngestionError, LearningHubError)


def test_retrievalerror_inherits_learninghuberror() -> None:
    """RetrievalError is a subclass of LearningHubError."""
    assert issubclass(RetrievalError, LearningHubError)


def test_upstreambadresponse_inherits_retrievalerror() -> None:
    """UpstreamBadResponse is a subclass of RetrievalError."""
    assert issubclass(UpstreamBadResponse, RetrievalError)


def test_upstreamunavailable_inherits_retrievalerror() -> None:
    """UpstreamUnavailable is a subclass of RetrievalError."""
    assert issubclass(UpstreamUnavailable, RetrievalError)


def test_exception_cause_chain_preserved() -> None:
    """Chained exceptions preserve __cause__."""
    try:
        try:
            raise ValueError("root cause")
        except ValueError as exc:
            raise IngestionError("ingestion failed") from exc
    except IngestionError as exc:
        assert exc.__cause__ is not None
        assert "root cause" in str(exc.__cause__)
