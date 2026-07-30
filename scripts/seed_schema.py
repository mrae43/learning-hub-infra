#!/usr/bin/env python3
"""Seed a PostgreSQL schema for one chunk-size config from its pre-computed sidecar.

Usage::

    uv run python scripts/seed_schema.py seed 256_10
    uv run python scripts/seed_schema.py teardown 256_10

The ``seed`` subcommand:

1. Reads the sidecar from ``eval_corpus/eval_vectors_{config}.json``
   (produced by ``scripts/generate_eval_vectors.py``).
2. Creates a dedicated PostgreSQL schema ``chunks_{config}``.
3. Creates enum types and tables inside that schema.
4. Inserts document, chunk (parent + child), and embedding rows.
5. Builds a pgvector HNSW index on the embedding column.

The ``teardown`` subcommand drops the schema with ``CASCADE``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from core.database.connection import get_engine
from core.database.schema import Base, Chunk, Document, Embedding
from core.types.document import DocumentStatus, DocumentType

# ── Constants ───────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_CORPUS_DIR = _REPO_ROOT / "eval_corpus"

CONFIG_NAMES: frozenset[str] = frozenset({"256_10", "512_15", "1024_20"})

_ENUM_DEFS: dict[str, list[str]] = {
    "document_type": ["paper", "book", "documentation"],
    "document_status": ["validating", "chunking", "embedding", "ready", "failed"],
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _config_to_schema_name(config_key: str) -> str:
    """Map a config key (e.g. ``256_10``) to its schema name (e.g. ``chunks_256_10``).

    Raises:
        ValueError: The config key is not one of the known tuning configs.
    """
    if config_key not in CONFIG_NAMES:
        raise ValueError(
            f"Unknown config key {config_key!r}. Expected one of {', '.join(sorted(CONFIG_NAMES))}."
        )
    return f"chunks_{config_key}"


def _sidecar_path(config_key: str) -> Path:
    """Return the sidecar path for a given config key (e.g. ``256_10``)."""
    return _EVAL_CORPUS_DIR / f"eval_vectors_{config_key}.json"


def _load_sidecar(path: Path) -> dict[str, Any]:
    """Load and return the sidecar JSON from *path*.

    Raises:
        FileNotFoundError: The sidecar file does not exist.
        json.JSONDecodeError: The file contains invalid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Sidecar not found: {path}")
    with open(path) as f:
        return cast("dict[str, Any]", json.load(f))


def _create_enums_in_schema(conn: Any, schema_name: str) -> None:
    """Create PostgreSQL ENUM types inside *schema_name*.

    Uses a PL/pgSQL ``DO`` block that catches ``duplicate_object`` so the
    call is idempotent (PG16 does not support ``IF NOT EXISTS`` for ENUMs).
    The explicit schema-qualified name ensures that
    ``schema_translate_map``-qualified type references in ORM DDL
    (e.g. ``chunks_256_10.document_type``) resolve correctly.
    """
    for name, values in _ENUM_DEFS.items():
        values_literal = ", ".join(f"'{v}'" for v in values)
        conn.execute(
            text(
                f"DO $$ BEGIN "
                f"CREATE TYPE {schema_name}.{name} AS ENUM ({values_literal}); "
                f"EXCEPTION WHEN duplicate_object THEN NULL; "
                f"END $$;"
            )
        )


def _create_schema_and_tables(engine: Engine, schema_name: str) -> None:
    """Create the target schema, enum types, and ORM tables."""
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
        _create_enums_in_schema(conn, schema_name)
        conn.commit()

    with engine.connect() as conn:
        conn = conn.execution_options(schema_translate_map={None: schema_name})
        Base.metadata.create_all(conn, checkfirst=True)
        conn.commit()


def _insert_sidecar(
    session: Session,
    sidecar: dict[str, Any],
) -> None:
    """Insert documents, chunks, and embeddings from *sidecar* into *session*.

    The *session* must be bound to an engine/connection that has
    ``schema_translate_map`` set so that ORM writes go to the correct schema.
    """
    model_name = sidecar.get("model", "unknown")
    vectors: dict[str, list[float]] = sidecar.get("vectors", {})

    for doc_data in sidecar.get("documents", []):
        doc = Document(
            title=doc_data["source_path"],
            document_type=DocumentType(doc_data["document_type"]),
            source_filename=Path(doc_data["source_path"]).name,
            status=DocumentStatus.READY,
        )
        session.add(doc)
        session.flush()

        for parent_data in doc_data.get("parents", []):
            parent = Chunk(
                document_id=doc.document_id,
                position=0,
                content=parent_data["content"],
                token_count=parent_data["token_count"],
                type_metadata=parent_data.get("metadata", {}),
                parent_chunk_id=None,
            )
            session.add(parent)
            session.flush()

            for child_data in parent_data.get("children", []):
                child = Chunk(
                    document_id=doc.document_id,
                    position=child_data["position"],
                    content=child_data["content"],
                    token_count=child_data["token_count"],
                    type_metadata={},
                    parent_chunk_id=parent.chunk_id,
                )
                session.add(child)
                session.flush()

                child_sha = child_data["content_sha256"]
                vec = vectors.get(child_sha)
                if vec is not None:
                    session.add(
                        Embedding(
                            chunk_id=child.chunk_id,
                            model_name=model_name,
                            embedding=vec,
                        )
                    )


def _build_hnsw_index(engine: Engine, schema_name: str) -> None:
    """Create a pgvector HNSW index on the embedding column."""
    clean_schema = schema_name.replace('"', '""')
    index_name = f"ix_embeddings_hnsw_{schema_name.removeprefix('chunks_')}"
    with engine.connect() as conn:
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f'ON "{clean_schema}".embeddings '
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
        conn.commit()


# ── Public API ──────────────────────────────────────────────────────────────


def seed_schema(
    config_key: str,
    *,
    engine: Engine | None = None,
    sidecar_override: Path | None = None,
) -> int:
    """Seed the schema for a single chunk-size config.

    Args:
        config_key: One of ``256_10``, ``512_15``, ``1024_20``.
        engine: Optional SQLAlchemy engine (defaults to ``get_engine()``).
        sidecar_override: Optional path to the sidecar file (for testing).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        schema_name = _config_to_schema_name(config_key)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    sidecar_path = sidecar_override or _sidecar_path(config_key)
    try:
        sidecar = _load_sidecar(sidecar_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    engine = engine or get_engine()

    _create_schema_and_tables(engine, schema_name)

    engine_with_schema = engine.execution_options(schema_translate_map={None: schema_name})
    SessionClass = sessionmaker(bind=engine_with_schema)
    session: Session = SessionClass()
    try:
        _insert_sidecar(session, sidecar)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    _build_hnsw_index(engine, schema_name)
    print(f"Seeded schema {schema_name!r} from {sidecar_path.name}")
    return 0


def teardown_schema(
    config_key: str,
    *,
    engine: Engine | None = None,
) -> int:
    """Drop the schema for a single chunk-size config with ``CASCADE``.

    Args:
        config_key: One of ``256_10``, ``512_15``, ``1024_20``.
        engine: Optional SQLAlchemy engine (defaults to ``get_engine()``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        schema_name = _config_to_schema_name(config_key)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    engine = engine or get_engine()
    clean_schema = schema_name.replace('"', '""')
    with engine.connect() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{clean_schema}" CASCADE'))
        conn.commit()
    print(f"Tore down schema {schema_name!r}")
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Seed or tear down a chunk-size tuning schema.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed", help="Create and populate a tuning schema.")
    seed_parser.add_argument(
        "config",
        choices=sorted(CONFIG_NAMES),
        help="Chunk-size config key (e.g. 256_10).",
    )

    teardown_parser = subparsers.add_parser("teardown", help="Drop a tuning schema with CASCADE.")
    teardown_parser.add_argument(
        "config",
        choices=sorted(CONFIG_NAMES),
        help="Chunk-size config key (e.g. 256_10).",
    )

    args = parser.parse_args(argv)

    if args.command == "seed":
        return seed_schema(args.config)
    elif args.command == "teardown":
        return teardown_schema(args.config)
    return 1


if __name__ == "__main__":
    sys.exit(main())
