"""Tests for scripts/seed_schema.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from core.database.schema import Chunk, Document, Embedding
from core.types.document import DocumentStatus
from scripts.seed_schema import (
    CONFIG_NAMES,
    _config_to_schema_name,
    _load_sidecar,
    _sidecar_path,
    seed_schema,
    teardown_schema,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


_EMBEDDING_DIM: int = 1536


def _vec(value: float) -> list[float]:
    """Build a 1536-dim vector with *value* repeated."""
    return [value] * _EMBEDDING_DIM


def _make_test_sidecar(
    *, model: str = "test-model", chunk_size: int = 256, overlap_ratio: float = 0.10
) -> dict[str, Any]:
    """Build a minimal synthetic sidecar dict with one document, one parent, two children."""
    child_a = {
        "content_sha256": "c1",
        "content": "Child A content",
        "token_count": 5,
        "position": 0,
    }
    child_b = {
        "content_sha256": "c2",
        "content": "Child B content",
        "token_count": 7,
        "position": 1,
    }
    parent = {
        "content_sha256": "p1",
        "content": "Parent content",
        "token_count": 12,
        "metadata": {"heading": "Introduction"},
        "children": [child_a, child_b],
    }
    return {
        "model": model,
        "config": {"chunk_size": chunk_size, "overlap_ratio": overlap_ratio},
        "dimensions": _EMBEDDING_DIM,
        "documents": [
            {
                "source_path": "eval_corpus/synthetic/test.md",
                "document_type": "documentation",
                "parents": [parent],
            }
        ],
        "vectors": {
            "c1": _vec(0.1),
            "c2": _vec(0.5),
        },
    }


# ── Seam: _config_to_schema_name ────────────────────────────────────────────


class TestConfigToSchemaName:
    def test_maps_256_10(self) -> None:
        assert _config_to_schema_name("256_10") == "chunks_256_10"

    def test_maps_512_15(self) -> None:
        assert _config_to_schema_name("512_15") == "chunks_512_15"

    def test_maps_1024_20(self) -> None:
        assert _config_to_schema_name("1024_20") == "chunks_1024_20"

    def test_unknown_config_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown config key"):
            _config_to_schema_name("128_05")

    def test_empty_config_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown config key"):
            _config_to_schema_name("")


# ── Seam: _sidecar_path ─────────────────────────────────────────────────────


class TestSidecarPath:
    def test_naming(self) -> None:
        path = _sidecar_path("256_10")
        assert path.name == "eval_vectors_256_10.json"

    def test_all_configs(self) -> None:
        for key in CONFIG_NAMES:
            path = _sidecar_path(key)
            assert path.name.startswith("eval_vectors_")
            assert path.name.endswith(".json")


# ── Seam: _load_sidecar ─────────────────────────────────────────────────────


class TestLoadSidecar:
    def test_loads_valid_sidecar(self, tmp_path: Path) -> None:
        sidecar = _make_test_sidecar()
        path = tmp_path / "eval_vectors_256_10.json"
        path.write_text(json.dumps(sidecar))
        loaded = _load_sidecar(path)
        assert loaded["model"] == "test-model"
        assert loaded["config"]["chunk_size"] == 256
        assert len(loaded["documents"]) == 1

    def test_missing_file_raises(self) -> None:
        path = Path("/nonexistent/sidecar.json")
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_sidecar(path)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{invalid")
        with pytest.raises(json.JSONDecodeError):
            _load_sidecar(path)


# ── Seam 4: seed_schema and teardown_schema (database-backed) ───────────────


class TestSeedAndTeardown:
    """Integration tests that use the test Postgres database.

    These tests depend on ``test_engine`` (session-scoped, from root conftest).
    Each test writes a temporary sidecar, seeds the schema, verifies the data,
    then tears down.
    """

    def test_seed_creates_schema_and_inserts_rows(
        self, test_engine: Engine, tmp_path: Path
    ) -> None:
        sidecar = _make_test_sidecar()
        sidecar_path = tmp_path / "eval_vectors_256_10.json"
        sidecar_path.write_text(json.dumps(sidecar))

        result = seed_schema("256_10", engine=test_engine, sidecar_override=sidecar_path)
        assert result == 0

        try:
            schema_name = "chunks_256_10"
            session_factory = sessionmaker(
                bind=test_engine.execution_options(schema_translate_map={None: schema_name})
            )
            session: Session = session_factory()
            try:
                docs = session.query(Document).all()
                assert len(docs) == 1
                assert docs[0].title == "eval_corpus/synthetic/test.md"
                assert docs[0].status == DocumentStatus.READY

                chunks = session.query(Chunk).all()
                # 1 parent + 2 children
                assert len(chunks) == 3

                parents = [c for c in chunks if c.parent_chunk_id is None]
                children = [c for c in chunks if c.parent_chunk_id is not None]
                assert len(parents) == 1
                assert len(children) == 2

                embeddings = session.query(Embedding).all()
                assert len(embeddings) == 2  # only children are embedded
                for emb in embeddings:
                    assert len(emb.embedding) == _EMBEDDING_DIM
                    assert emb.model_name == "test-model"

                # Verify parent-child relationship
                parent = parents[0]
                for child in children:
                    assert child.parent_chunk_id == parent.chunk_id
            finally:
                session.close()
        finally:
            teardown_schema("256_10", engine=test_engine)

    def test_seed_is_idempotent(self, test_engine: Engine, tmp_path: Path) -> None:
        """Calling seed twice with the same config should succeed (CREATE SCHEMA IF NOT EXISTS)."""
        sidecar = _make_test_sidecar()
        sidecar_path = tmp_path / "eval_vectors_256_10.json"
        sidecar_path.write_text(json.dumps(sidecar))

        try:
            result1 = seed_schema("256_10", engine=test_engine, sidecar_override=sidecar_path)
            assert result1 == 0

            result2 = seed_schema("256_10", engine=test_engine, sidecar_override=sidecar_path)
            assert result2 == 0
        finally:
            teardown_schema("256_10", engine=test_engine)

    def test_teardown_drops_schema(self, test_engine: Engine, tmp_path: Path) -> None:
        sidecar = _make_test_sidecar()
        sidecar_path = tmp_path / "eval_vectors_256_10.json"
        sidecar_path.write_text(json.dumps(sidecar))

        seed_schema("256_10", engine=test_engine, sidecar_override=sidecar_path)

        result = teardown_schema("256_10", engine=test_engine)
        assert result == 0

        # Verify the schema is gone
        with test_engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT schema_name FROM information_schema.schemata WHERE schema_name = :name"
                ),
                {"name": "chunks_256_10"},
            ).fetchone()
            assert row is None

    def test_teardown_is_idempotent(self, test_engine: Engine) -> None:
        result1 = teardown_schema("256_10", engine=test_engine)
        assert result1 == 0

        result2 = teardown_schema("256_10", engine=test_engine)
        assert result2 == 0

    def test_seed_512_15_config(self, test_engine: Engine, tmp_path: Path) -> None:
        sidecar = _make_test_sidecar(chunk_size=512, overlap_ratio=0.15)
        sidecar_path = tmp_path / "eval_vectors_512_15.json"
        sidecar_path.write_text(json.dumps(sidecar))

        try:
            result = seed_schema("512_15", engine=test_engine, sidecar_override=sidecar_path)
            assert result == 0

            schema_name = "chunks_512_15"
            session_factory = sessionmaker(
                bind=test_engine.execution_options(schema_translate_map={None: schema_name})
            )
            session: Session = session_factory()
            try:
                docs = session.query(Document).all()
                assert len(docs) == 1
                chunks = session.query(Chunk).all()
                assert len(chunks) == 3
            finally:
                session.close()
        finally:
            teardown_schema("512_15", engine=test_engine)

    def test_seed_1024_20_config(self, test_engine: Engine, tmp_path: Path) -> None:
        sidecar = _make_test_sidecar(chunk_size=1024, overlap_ratio=0.20)
        sidecar_path = tmp_path / "eval_vectors_1024_20.json"
        sidecar_path.write_text(json.dumps(sidecar))

        try:
            result = seed_schema("1024_20", engine=test_engine, sidecar_override=sidecar_path)
            assert result == 0

            schema_name = "chunks_1024_20"
            session_factory = sessionmaker(
                bind=test_engine.execution_options(schema_translate_map={None: schema_name})
            )
            session: Session = session_factory()
            try:
                docs = session.query(Document).all()
                assert len(docs) == 1
                chunks = session.query(Chunk).all()
                assert len(chunks) == 3
            finally:
                session.close()
        finally:
            teardown_schema("1024_20", engine=test_engine)

    def test_hnsw_index_exists_after_seed(self, test_engine: Engine, tmp_path: Path) -> None:
        sidecar = _make_test_sidecar()
        sidecar_path = tmp_path / "eval_vectors_256_10.json"
        sidecar_path.write_text(json.dumps(sidecar))

        try:
            seed_schema("256_10", engine=test_engine, sidecar_override=sidecar_path)

            with test_engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE schemaname = :schema AND indexname LIKE :pat"
                    ),
                    {"schema": "chunks_256_10", "pat": "ix_embeddings_hnsw_%"},
                ).fetchone()
                assert row is not None, "HNSW index was not created"
        finally:
            teardown_schema("256_10", engine=test_engine)

    def test_invalid_config_returns_error(self, test_engine: Engine) -> None:
        result = seed_schema("invalid", engine=test_engine)
        assert result == 1
