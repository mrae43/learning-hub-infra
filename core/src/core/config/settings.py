"""Application settings for the Learning Hub.

Settings are loaded from environment variables and a `.env` file if present.
All infra-internal knobs (database URL, active embedding model, HNSW tuning,
upload limits) live here so the API contract stays stable as infra choices evolve.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide configuration.

    Attributes:
        database_url: Postgres+pgvector connection URL.
        openai_api_key: API key for the OpenAI embeddings client.
        embedding_model: Active embedding model ID. All models used during MVP
            must produce 1536-dim vectors (ADR-0014).
        hnsw_ef_search: Query-time HNSW search candidate list size.
        query_top_k: Number of chunks the retrieval step fetches per query
            (server-side infra knob, not client-controlled per ADR-0014).
        inference_model: Active chat-completion model ID for generation.
        max_upload_bytes: Maximum accepted upload size in bytes.
        allowed_file_extensions: Lower-case file extensions accepted for upload.
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
    embedding_model: str = "text-embedding-3-small"
    hnsw_ef_search: int = 40
    query_top_k: int = 5
    inference_model: str = "gpt-4o-mini"
    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB placeholder
    allowed_file_extensions: set[str] = {"pdf", "epub", "md", "html"}
    cohere_api_key: str | None = None
    reranker_model: str = "rerank-v3.5"


# Global singleton used by the application. Tests override via monkeypatch or
# by constructing a fresh Settings instance.
settings = Settings()
