"""Project-specific named exceptions."""


class LearningHubError(Exception):
    """Base class for all application errors."""


class IngestionError(LearningHubError):
    """Raised when ingestion fails in a way the caller should handle."""


class RetrievalError(LearningHubError):
    """Raised when retrieval/query fails because of an upstream or DB problem."""


class UpstreamBadResponse(RetrievalError):
    """Upstream API returned an unexpected response (maps to 502)."""


class UpstreamUnavailable(RetrievalError):
    """Upstream API could not be reached or timed out (maps to 503)."""


class RerankerRateLimitError(RetrievalError):
    """Reranker API rate-limited (trigger graceful fallback to RRF top-k)."""
