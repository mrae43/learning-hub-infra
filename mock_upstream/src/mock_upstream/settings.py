"""Environment configuration for the mock upstream service.

The mock is OpenAI-compatible by shape (same request/response contracts as the
hosted API) so the app's existing SDK clients can point at it via
``OPENAI_BASE_URL`` with zero code changes. Its own knobs live under the
``MOCK_`` prefix: per-endpoint simulated-latency ranges (``MOCK_LATENCY_ENABLED``
defaults on; set ``MOCK_LATENCY_ENABLED=false`` or any range's max to ``0``
for pure-throughput sweeps). Embedding dimensionality is fixed at 1536
(ADR-0014) and deliberately not configurable — a mismatched dimension would
break pgvector cosine queries against the real-embedded corpus.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MockUpstreamSettings(BaseSettings):
    """Configuration for the mock upstream.

    Attributes:
        latency_enabled: When True, each request sleeps a uniform random delay
            within the per-endpoint range. Set False (or zero any range's
            ``max_ms``) to remove simulated latency entirely.
        embeddings_latency_min_ms: Minimum simulated latency for ``/v1/embeddings``.
        embeddings_latency_max_ms: Maximum simulated latency for ``/v1/embeddings``.
        chat_latency_min_ms: Minimum simulated latency for ``/v1/chat/completions``.
        chat_latency_max_ms: Maximum simulated latency for ``/v1/chat/completions``.
        web_search_latency_min_ms: Minimum simulated latency for ``/v1/responses``.
        web_search_latency_max_ms: Maximum simulated latency for ``/v1/responses``.
        error_rate: Probability (0.0-1.0) that a request fails with
            ``error_status`` instead of succeeding. Zero (the default) disables
            fault injection; load runs set it to induce an upstream error storm.
        error_status: HTTP status returned when fault injection fires,
            constrained to a valid status range (100-599).
    """

    model_config = SettingsConfigDict(
        env_prefix="MOCK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    latency_enabled: bool = True
    embeddings_latency_min_ms: int = 30
    embeddings_latency_max_ms: int = 80
    chat_latency_min_ms: int = 150
    chat_latency_max_ms: int = 400
    web_search_latency_min_ms: int = 150
    web_search_latency_max_ms: int = 400
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    error_status: int = Field(default=500, ge=100, le=599)


# Global singleton used by the app. Tests construct fresh instances or
# monkeypatch this module's ``settings``.
settings = MockUpstreamSettings()
