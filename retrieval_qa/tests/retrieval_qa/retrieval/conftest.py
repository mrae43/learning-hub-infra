"""Session-scoped fixture that seeds the eval corpus once per session.

Depends on ``test_engine`` from the root ``conftest.py``.  Because the root
``test_session`` fixture (function-scoped) calls ``drop_all`` / ``create_all``
each time, this module **must not** use ``test_session`` — it creates its own
tables once and reuses them across all parametrized eval queries via the
``eval_session`` function-scoped fixture.

Alphabetical ordering (``eval`` < ``query``) ensures the eval tests run before
the regular retrieval tests in the same directory, so the session-scoped seed
is still present when the eval tests execute.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
import yaml
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from core.database.schema import Base, Chunk, Document, Embedding
from core.types.document import DocumentStatus, DocumentType

_EVAL_SET_PATH = Path(__file__).parent / "eval_set.yaml"

_ENUM_DEFS = {
    "document_type": ["paper", "book", "documentation"],
    "document_status": ["validating", "chunking", "embedding", "ready", "failed"],
}


def _create_enums(engine: Engine) -> None:
    with engine.connect() as conn:
        for name, values in _ENUM_DEFS.items():
            vals = ", ".join(f"'{v}'" for v in values)
            conn.execute(
                text(
                    f"DO $$ BEGIN "
                    f"CREATE TYPE {name} AS ENUM ({vals}); "
                    f"EXCEPTION WHEN duplicate_object THEN NULL; "
                    f"END $$"
                )
            )
        conn.commit()


def _seed_eval_set(engine: Engine) -> None:
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

                embedding_vec = [float(v) for v in chunk_data["embedding"]]
                session.add(
                    Embedding(
                        chunk_id=chunk.chunk_id,
                        model_name="text-embedding-3-small",
                        embedding=embedding_vec,
                    )
                )

        session.commit()
    finally:
        session.close()


@pytest.fixture(scope="session")
def eval_corpus(test_engine: Engine) -> Engine:
    """Create tables, enums, and seed the eval corpus once per session."""
    Base.metadata.drop_all(bind=test_engine)
    with test_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    _create_enums(test_engine)
    Base.metadata.create_all(bind=test_engine)
    _seed_eval_set(test_engine)
    return test_engine


@pytest.fixture
def eval_session(eval_corpus: Engine) -> Generator[Session, None, None]:
    """Provide a fresh session against the pre-seeded eval corpus."""
    session = sessionmaker(bind=eval_corpus)()
    try:
        yield session
    finally:
        session.close()
