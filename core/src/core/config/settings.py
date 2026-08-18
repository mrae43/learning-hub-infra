"""Application settings for the Learning Hub.

Settings are loaded from environment variables and a `.env` file if present.
All infra-internal knobs (database URL, active embedding model, HNSW tuning,
upload limits) live here so the API contract stays stable as infra choices evolve.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
"""The OpenAI SDK's default endpoint, used when no base URL is configured.

Clients resolve ``settings.openai_base_url or DEFAULT_OPENAI_BASE_URL`` to a
concrete value so the SDK never falls back to an empty ``OPENAI_BASE_URL``
environment variable (compose sets ``${VAR:-}`` to an empty string when the
variable is unset, which would otherwise break every OpenAI call).
"""


class Settings(BaseSettings):
    """Project-wide configuration.

    Attributes:
        database_url: Postgres+pgvector connection URL.
        openai_api_key: API key for the OpenAI embeddings client.
        openai_base_url: Optional base URL redirecting every OpenAI-backed
            client (embeddings, chat completions, web search) to another
            endpoint. The ``load`` compose profile points it at the mock
            upstream so volume runs spend no real API budget (ADR-0014 area;
            see ``CONTEXT.md`` "Mock Upstream").
        embedding_model: Active embedding model ID. All models used during MVP
            must produce 1536-dim vectors (ADR-0014).
        hnsw_ef_search: Query-time HNSW search candidate list size.
        query_top_k: Number of chunks the retrieval step fetches per query
            (server-side infra knob, not client-controlled per ADR-0014).
        inference_model: Active chat-completion model ID for generation.
        web_search_model: Model used by the Depth Dive web-search client
            (OpenAI Responses API ``web_search`` tool).
        max_upload_bytes: Maximum accepted upload size in bytes.
        allowed_file_extensions: Lower-case file extensions accepted for upload.
        otel_exporter_otlp_endpoint: OTLP/HTTP collector endpoint for the five
            RAG stage spans (embed / retrieve / rerank / generate /
            web-search). Empty means tracing stays inert: no exporter is
            installed and no spans leave the process.
        otel_service_name: ``service.name`` resource attribute applied to the
            exported spans.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = (
        "postgresql+psycopg2://learning_hub:learning_hub@localhost:5432/learning_hub"
    )
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    hnsw_ef_search: int = 40
    query_top_k: int = 5
    inference_model: str = "gpt-4o-mini"
    web_search_model: str = "gpt-4o-mini"
    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB placeholder
    allowed_file_extensions: set[str] = {"pdf", "epub", "md", "html"}
    cohere_api_key: str | None = None
    reranker_model: str = "rerank-v3.5"
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "learning-hub"

    @field_validator("openai_api_key", "openai_base_url", "cohere_api_key", mode="before")
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        """Coerce empty/whitespace env values to ``None``.

        Compose substitutes unset variables as empty strings (``${VAR:-}``);
        an empty ``OPENAI_BASE_URL`` or ``OPENAI_API_KEY`` must mean "not
        configured" (``None``) rather than a literal empty value that the
        OpenAI SDK would choke on.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value


def resolve_openai_base_url(override: str | None) -> str:
    """Resolve a client's effective OpenAI base URL.

    Every OpenAI-backed client (embeddings, chat completions, web search)
    resolves its endpoint the same way: an explicit override wins, then
    ``settings.openai_base_url``, then the SDK default. Resolving to a
    concrete value (never ``None``) stops the SDK from falling back to an
    empty ``OPENAI_BASE_URL`` environment variable, which compose produces
    for unset ``${VAR:-}`` and which would break every OpenAI call.

    Args:
        override: A caller-supplied base URL, or ``None`` to use the setting.

    Returns:
        The concrete base URL for the OpenAI client.
    """
    return (
        override if override is not None else (settings.openai_base_url or DEFAULT_OPENAI_BASE_URL)
    )


# Global singleton used by the application. Tests override via monkeypatch or
# by constructing a fresh Settings instance.
settings = Settings()
