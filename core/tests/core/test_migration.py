"""Tests for the initial Alembic migration."""

from collections.abc import Generator, Sequence

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, Inspector


def _run_alembic_command(url: str, *args: str) -> None:
    """Run an alembic CLI command against the given database URL."""
    import subprocess

    subprocess.run(
        ["uv", "run", "alembic", *args],
        cwd=".",
        env={**__import__("os").environ, "DATABASE_URL": url},
        check=True,
    )


@pytest.fixture
def migrated_engine(test_database_url: str) -> Generator[Engine, None, None]:
    """Upgrade the test database to head and yield an engine."""
    _run_alembic_command(test_database_url, "downgrade", "base")
    _run_alembic_command(test_database_url, "upgrade", "head")
    engine = create_engine(test_database_url, future=True)
    try:
        yield engine
    finally:
        engine.dispose()


def _table_names(inspector: Inspector) -> set[str]:
    return set(inspector.get_table_names())


def _enum_names(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT typname FROM pg_type WHERE typtype = 'e'")).fetchall()
    return {row[0] for row in rows}


def _check_constraints(engine: Engine, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT c.conname FROM pg_constraint c "
                "JOIN pg_class t ON t.oid = c.conrelid "
                "WHERE t.relname = :table_name AND c.contype = 'c'"
            ),
            {"table_name": table},
        ).fetchall()
    return {row[0] for row in rows}


def _unique_constraints(engine: Engine, table: str) -> set[str]:
    inspector = inspect(engine)
    return {uc["name"] for uc in inspector.get_unique_constraints(table) if uc["name"] is not None}


def _gin_index_info(engine: Engine, table: str, index_name: str) -> dict[str, object]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE tablename = :table_name AND indexname = :index_name"
            ),
            {"table_name": table, "index_name": index_name},
        ).fetchone()
    if row is None:
        return {}
    return {"name": row[0], "definition": row[1]}


def _hnsw_index_info(engine: Engine, table: str) -> dict[str, object]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE tablename = :table_name AND indexname = :index_name"
            ),
            {"table_name": table, "index_name": "ix_embeddings_embedding_hnsw"},
        ).fetchone()
    if row is None:
        return {}
    return {"name": row[0], "definition": row[1]}


def _column_names(inspector: Inspector, table: str) -> set[str]:
    return {col["name"] for col in inspector.get_columns(table)}


def _primary_key_columns(inspector: Inspector, table: str) -> Sequence[str]:
    result = inspector.get_pk_constraint(table)
    return result["constrained_columns"]


def test_migration_creates_expected_schema(migrated_engine: Engine) -> None:
    """The initial migration creates the ADR-0014 schema in full."""
    inspector = inspect(migrated_engine)

    assert {"documents", "chunks", "embeddings"}.issubset(_table_names(inspector))

    # Documents table shape.
    doc_columns = _column_names(inspector, "documents")
    assert doc_columns == {
        "document_id",
        "title",
        "document_type",
        "source_filename",
        "status",
        "error_message",
        "created_at",
        "updated_at",
    }

    # Chunks table shape: no updated_at (ADR-0014 exception).
    chunk_columns = _column_names(inspector, "chunks")
    assert chunk_columns == {
        "chunk_id",
        "document_id",
        "position",
        "content",
        "token_count",
        "type_metadata",
        "parent_chunk_id",
        "content_search",
        "created_at",
    }
    assert "updated_at" not in chunk_columns

    # Embeddings table shape: composite PK, no updated_at.
    embedding_columns = _column_names(inspector, "embeddings")
    assert embedding_columns == {
        "chunk_id",
        "model_name",
        "embedding",
        "created_at",
    }
    assert "updated_at" not in embedding_columns
    assert _primary_key_columns(inspector, "embeddings") == ["chunk_id", "model_name"]

    # Enums present.
    assert _enum_names(migrated_engine) == {"document_status", "document_type"}

    # CHECK constraints present.
    assert "ck_error_message_only_when_failed" in _check_constraints(migrated_engine, "documents")
    assert "ck_chunk_position_non_negative" in _check_constraints(migrated_engine, "chunks")
    assert "ck_chunk_token_count_positive" in _check_constraints(migrated_engine, "chunks")

    # parent_chunk_id FK is self-referential with ON DELETE SET NULL.
    chunk_fks = inspector.get_foreign_keys("chunks")
    parent_chunk_fk = [fk for fk in chunk_fks if fk["constrained_columns"] == ["parent_chunk_id"]]
    assert len(parent_chunk_fk) == 1
    assert parent_chunk_fk[0]["referred_table"] == "chunks"
    assert parent_chunk_fk[0]["referred_columns"] == ["chunk_id"]
    assert parent_chunk_fk[0].get("options", {}).get("ondelete", "").upper() == "SET NULL"

    # content_search column is tsvector.
    chunk_cols = inspector.get_columns("chunks")
    content_search_col = [c for c in chunk_cols if c["name"] == "content_search"]
    assert len(content_search_col) == 1
    assert str(content_search_col[0]["type"]).lower() == "tsvector"

    # GIN index on content_search.
    content_search_index = _gin_index_info(
        migrated_engine, "chunks", "ix_chunks_content_search_gin"
    )
    assert content_search_index, "GIN index on content_search not found"

    # HNSW index on embeddings.embedding with vector_cosine_ops.
    hnsw = _hnsw_index_info(migrated_engine, "embeddings")
    assert hnsw, "HNSW index not found"
    definition = str(hnsw["definition"])
    assert "USING hnsw" in definition
    assert "vector_cosine_ops" in definition


def test_migration_downgrade_reverses_cleanly(migrated_engine: Engine) -> None:
    """Downgrading to base removes the tables and enum types."""
    inspector = inspect(migrated_engine)
    assert {"documents", "chunks", "embeddings"}.issubset(_table_names(inspector))

    url = migrated_engine.url.render_as_string(hide_password=False)
    _run_alembic_command(url, "downgrade", "base")

    inspector = inspect(migrated_engine)
    remaining_tables = _table_names(inspector)
    assert {"documents", "chunks", "embeddings"}.isdisjoint(remaining_tables)
    assert _enum_names(migrated_engine) == set()


def test_vector_extension_exists(migrated_engine: Engine) -> None:
    """The pgvector extension is created by the migration."""
    with migrated_engine.connect() as conn:
        result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
    assert result == 1
