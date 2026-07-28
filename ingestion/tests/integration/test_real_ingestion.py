"""Integration test: real OpenAI embeddings against a real Postgres+pgvector db.

Marked `integration` and skipped by default -- it makes real network calls
and costs real money. It reuses the existing conftest.py fixtures
(test_session / test_engine), so it gets a real, freshly-migrated Postgres +
pgvector database, not a mock.
"""

import logging
import os
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from core.clients.embeddings_client import EmbeddingsClient
from core.config.settings import settings
from core.database.schema import Chunk as ChunkRow
from core.database.schema import Document, Embedding
from core.types.document import DocumentStatus, DocumentType
from ingestion.models import PendingIngestion
from ingestion.pipeline import run_ingestion

pytestmark = pytest.mark.integration

_LIVE_ENV_VAR = "RUN_LIVE_API_TESTS"


def _skip_unless_live() -> None:
    """Refuse to run unless explicitly opted in -- this costs money."""
    if os.environ.get(_LIVE_ENV_VAR) != "1":
        pytest.skip(f"Set {_LIVE_ENV_VAR}=1 to run live-API integration tests")
    if not (os.environ.get("OPENAI_API_KEY") or settings.openai_api_key):
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture
def real_document_path() -> Path:
    """Path to a real PDF or EPUB. Set via TEST_DOCUMENT_PATH."""
    raw = os.environ.get("TEST_DOCUMENT_PATH")
    if not raw:
        pytest.skip("Set TEST_DOCUMENT_PATH to a real .pdf or .epub file")
    path = Path(raw)
    if not path.exists():
        pytest.skip(f"TEST_DOCUMENT_PATH does not exist: {path}")
    return path


@pytest.fixture
def document_type_for(real_document_path: Path) -> DocumentType:
    """Infer document type from extension. Adjust if your file needs a
    specific type regardless of extension (e.g. a technical PDF that should
    be chunked as documentation, not paper)."""
    suffix = real_document_path.suffix.lower()
    if suffix == ".pdf":
        return DocumentType.PAPER
    if suffix == ".epub":
        return DocumentType.BOOK
    pytest.skip(f"Unsupported extension for this test: {suffix}")


def _make_pending_document(
    session: Session, real_document_path: Path, document_type: DocumentType
) -> PendingIngestion:
    document = Document(
        title=f"Integration test: {real_document_path.name}",
        document_type=document_type,
        source_filename=real_document_path.name,
        status=DocumentStatus.VALIDATING,
    )
    session.add(document)
    session.flush()
    return PendingIngestion(
        document_id=document.document_id,
        title=document.title,
        document_type=document_type,
        source_filename=document.source_filename,
        file_path=real_document_path,
    )


def test_real_ingestion_end_to_end(
    test_session: Session,
    real_document_path: Path,
    document_type_for: DocumentType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Full pipeline against real Postgres + real OpenAI embeddings.

    Verifies:
      - pipeline completes without raising IngestionError
      - document ends in READY
      - parent chunks (structure-aware) and child chunks (fixed-size, ADR-0016)
        both exist, with correct parent/child linkage
      - every child chunk has exactly one embedding at the right dimensionality
      - embeddings aren't degenerate (all-zero or constant), which is how a
        silent upstream failure tends to show up rather than raising
      - cost tracking log includes chunk count, API call count, and estimated cost
    """
    _skip_unless_live()

    caplog.set_level(logging.INFO)

    pending = _make_pending_document(test_session, real_document_path, document_type_for)
    client = EmbeddingsClient()  # picks up settings.openai_api_key / OPENAI_API_KEY

    run_ingestion(
        pending=pending,
        session=test_session,
        embeddings_client=client,
        model_name=settings.embedding_model,
    )
    test_session.commit()

    document = test_session.get(Document, pending.document_id)
    assert document is not None
    assert document.status == DocumentStatus.READY, document.error_message

    all_chunks = (
        test_session.query(ChunkRow).filter(ChunkRow.document_id == pending.document_id).all()
    )
    parents = [c for c in all_chunks if c.parent_chunk_id is None]
    children = [c for c in all_chunks if c.parent_chunk_id is not None]

    assert parents, "expected at least one structure-aware parent chunk"
    assert children, "expected at least one fixed-size child chunk"
    assert all(c.parent_chunk_id in {p.chunk_id for p in parents} for c in children), (
        "every child must point at a parent that was actually persisted"
    )

    child_ids = {c.chunk_id for c in children}
    embeddings = test_session.query(Embedding).filter(Embedding.chunk_id.in_(child_ids)).all()
    assert len(embeddings) == len(children), "every child chunk must have exactly one embedding"

    for emb in embeddings:
        vector = emb.embedding
        assert len(vector) == 1536, (
            f"expected 1536-dim vector for text-embedding-3-small, got {len(vector)}"
        )
        assert any(v != 0 for v in vector), "embedding is all-zero -- likely a silent API failure"
        assert len({round(v, 6) for v in vector}) > 1, "embedding is constant -- suspicious"
        assert emb.model_name == settings.embedding_model

    # Verify cost tracking log is present
    cost_records = [r for r in caplog.records if "estimated cost" in r.message]
    assert cost_records, "expected cost tracking log message"
    assert "chunks" in cost_records[0].message, (
        f"expected 'chunks' in cost log, got: {cost_records[0].message}"
    )
    assert "embedding API calls" in cost_records[0].message, (
        f"expected 'embedding API calls' in cost log, got: {cost_records[0].message}"
    )


def test_real_ingestion_is_grounded(
    test_session: Session,
    real_document_path: Path,
    document_type_for: DocumentType,
) -> None:
    """Retrieval on the real content returns grounded, correctly-cited results.

    This is the test that actually tells you retrieval quality is real, not
    just that the pipe didn't break. Needs a question/answer pair specific to
    whatever document you upload -- see TEST_DOCUMENT_QUERY /
    TEST_DOCUMENT_EXPECTED_SUBSTRING at the top of this file.
    """
    _skip_unless_live()

    query_text = os.environ.get("TEST_DOCUMENT_QUERY")
    expected_substring = os.environ.get("TEST_DOCUMENT_EXPECTED_SUBSTRING")
    if not query_text or not expected_substring:
        pytest.skip(
            "Set TEST_DOCUMENT_QUERY and TEST_DOCUMENT_EXPECTED_SUBSTRING to "
            "something you know is answerable from the real doc"
        )

    pending = _make_pending_document(test_session, real_document_path, document_type_for)
    client = EmbeddingsClient()

    run_ingestion(
        pending=pending,
        session=test_session,
        embeddings_client=client,
        model_name=settings.embedding_model,
    )
    test_session.commit()

    # Embed the query and retrieve relevant chunks
    from core.types.retrieval_config import RetrievalConfig
    from retrieval_qa.retrieval.query import retrieve_relevant_chunks

    query_vector = client.embed([query_text])[0]
    config = RetrievalConfig(
        model_name=settings.embedding_model,
        ef_search=settings.hnsw_ef_search,
        top_k=settings.query_top_k,
    )

    passages = retrieve_relevant_chunks(
        query_vector=query_vector,
        session=test_session,
        config=config,
        query_text=query_text,
    )

    assert passages, f"expected at least one retrieved chunk for query: {query_text!r}"
    passage_texts = [p.text for p in passages]
    assert any(expected_substring in text for text in passage_texts), (
        f"expected substring {expected_substring!r} not found in passages. "
        f"Retrieved: {[t[:100] for t in passage_texts]}"
    )
