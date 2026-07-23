"""Session-scoped fixtures for retrieval eval tests.

Depends on ``test_engine`` from the root ``conftest.py``.  Because the root
``test_session`` fixture (function-scoped) calls ``drop_all`` / ``create_all``
each time, this module **must not** use ``test_session`` — it creates its own
tables once and reuses them across all parametrized eval queries via the
``eval_session`` function-scoped fixture.

Placed in a dedicated ``tests/eval/`` directory so eval tests are a
self-contained suite that cannot be interleaved with regular retrieval tests.
"""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Engine, text
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Session, sessionmaker

from core.config.settings import Settings
from core.database.schema import Base, Chunk, Document, Embedding
from core.types.document import DocumentStatus, DocumentType
from retrieval_qa._utils import _sha256

_EVAL_SET_PATH = Path(__file__).parent / "eval_set.yaml"
_EVAL_VECTORS_PATH = _EVAL_SET_PATH.with_name("eval_vectors.json")

_ENUM_DEFS = {
    "document_type": ["paper", "book", "documentation"],
    "document_status": ["validating", "chunking", "embedding", "ready", "failed"],
}


def _create_enums(engine: Engine) -> None:
    with engine.connect() as conn:
        for name, values in _ENUM_DEFS.items():
            enum_type = ENUM(*values, name=name)
            enum_type.create(bind=conn, checkfirst=True)
        conn.commit()


def _validate_eval_integrity(
    data: dict[str, Any],
    vectors: dict[str, list[float]],
    sidecar_model: str | None = None,
) -> None:
    """Validate SHA-256 hashes and vector cross-references.

    Checks:
    - Every chunk content and query string hash matches its stored hash.
    - Every YAML content hash has a corresponding entry in the vector JSON.
    - Every vector in JSON has a corresponding YAML entry (no orphans).
    - ``data["embedding_model"]`` matches ``settings.embedding_model``.
    - Sidecar JSON model metadata matches ``settings.embedding_model`` (if provided).
    """
    yaml_hashes: set[str] = set()

    expected_model = Settings().embedding_model
    assert data["embedding_model"] == expected_model, (
        f"YAML embedding_model '{data['embedding_model']}' does not match "
        f"settings.embedding_model '{expected_model}'. "
        f"Update eval_set.yaml or settings."
    )
    if sidecar_model is not None:
        assert sidecar_model == expected_model, (
            f"Sidecar JSON model '{sidecar_model}' does not match "
            f"settings.embedding_model '{expected_model}'. "
            f"Re-run scripts/generate_eval_vectors.py."
        )

    for doc in data.get("documents", []):
        for chunk in doc.get("chunks", []):
            actual = _sha256(chunk["content"])
            expected = chunk["content_sha256"]
            assert actual == expected, (
                f"SHA-256 mismatch for chunk content {chunk['content'][:60]!r}...: "
                f"got {actual}, expected {expected}. "
                f"Re-run scripts/generate_eval_vectors.py to update."
            )
            yaml_hashes.add(expected)

    for query in data.get("queries", []):
        actual = _sha256(query["query"])
        expected = query["content_sha256"]
        assert actual == expected, (
            f"SHA-256 mismatch for query {query['query']!r}: "
            f"got {actual}, expected {expected}. "
            f"Re-run scripts/generate_eval_vectors.py to update."
        )
        yaml_hashes.add(expected)

    for yaml_hash in yaml_hashes:
        assert yaml_hash in vectors, (
            f"Vector missing for entry with hash {yaml_hash}. "
            f"Re-run `uv run python scripts/generate_eval_vectors.py`"
        )

    for vec_hash in vectors:
        assert vec_hash in yaml_hashes, (
            f"Orphan vector in sidecar JSON for hash {vec_hash} — no matching YAML entry found."
        )


@pytest.fixture(scope="session")
def eval_vectors() -> dict[str, list[float]]:
    """Load sidecar vectors and validate integrity."""
    with open(_EVAL_VECTORS_PATH) as f:
        sidecar = json.load(f)
    with open(_EVAL_SET_PATH) as f:
        eval_data = yaml.safe_load(f)
    vectors: dict[str, list[float]] = sidecar["vectors"]
    _validate_eval_integrity(
        eval_data,
        vectors,
        sidecar_model=sidecar.get("model"),
    )
    return vectors


def _seed_eval_set(engine: Engine, vectors: dict[str, list[float]]) -> None:
    with open(_EVAL_SET_PATH) as f:
        data = yaml.safe_load(f)

    session = sessionmaker(bind=engine)()
    try:
        for doc_data in data["documents"]:
            doc_type = DocumentType(doc_data["document_type"])
            document = Document(
                title=doc_data["title"],
                document_type=doc_type,
                source_filename=f"{doc_data['title']}.pdf",
                status=DocumentStatus.READY,
            )
            session.add(document)
            session.flush()

            for position, chunk_data in enumerate(doc_data["chunks"]):
                chunk = Chunk(
                    document_id=document.document_id,
                    position=position,
                    content=chunk_data["content"],
                    token_count=max(1, len(chunk_data["content"].split())),
                )
                session.add(chunk)
                session.flush()

                embedding_vec = vectors[chunk_data["content_sha256"]]
                session.add(
                    Embedding(
                        chunk_id=chunk.chunk_id,
                        model_name=data["embedding_model"],
                        embedding=embedding_vec,
                    )
                )

        session.commit()
    finally:
        session.close()


@pytest.fixture(scope="session")
def eval_corpus(test_engine: Engine, eval_vectors: dict[str, list[float]]) -> Engine:
    """Create tables, enums, and seed the eval corpus once per session."""
    Base.metadata.drop_all(bind=test_engine)
    with test_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    _create_enums(test_engine)
    Base.metadata.create_all(bind=test_engine)
    _seed_eval_set(test_engine, eval_vectors)
    return test_engine


@pytest.fixture
def eval_session(eval_corpus: Engine) -> Generator[Session, None, None]:
    """Provide a fresh session against the pre-seeded eval corpus."""
    session = sessionmaker(bind=eval_corpus)()
    try:
        yield session
    finally:
        session.close()
