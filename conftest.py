"""Shared pytest fixtures for database-backed tests.

Tests that need a real Postgres+pgvector database use the ``learning_hub_test``
database created on the Postgres instance pointed to by ``DATABASE_URL``.
If the database is unavailable, the fixtures skip the test.
"""

import io
import os
from collections.abc import Generator
from contextlib import contextmanager
from urllib.parse import urlparse

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# Pre-import workspace packages so their installed (src/) versions are cached
# in sys.modules before per-package test collection runs.  Without this a test
# package whose name mirrors a source package (e.g. ``tests/retrieval_qa``)
# would shadow the installed package under importlib import mode.
import api.server  # noqa: F401
import ingestion.tasks  # noqa: F401
import retrieval_qa.chunking.paper_chunker  # noqa: F401

# Pre-import the workspace packages so their installed (src/) versions are
# cached in sys.modules before per-package test collection runs. Without this,
# a test package whose name mirrors a source package (e.g. ``tests/retrieval_qa``)
# would shadow the installed package under importlib import mode and break
# submodule imports like ``retrieval_qa.chunking.paper_chunker``.
from core.config.settings import Settings
from core.database.schema import Base

# Test database name derived from the configured database URL.
_TEST_DB_NAME = "learning_hub_test"


def _test_database_url() -> str:
    """Return the URL for the test database, overriding the configured DB name."""
    base_url = os.environ.get("DATABASE_URL", Settings().database_url)
    parsed = urlparse(base_url)
    return parsed._replace(path=f"/{_TEST_DB_NAME}").geturl()


def _ensure_test_database_exists() -> str:
    """Create the test database if it does not exist."""
    base_url = os.environ.get("DATABASE_URL", Settings().database_url)
    parsed = urlparse(base_url)
    admin_url = parsed._replace(path="/postgres").geturl()
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": _TEST_DB_NAME},
            )
            if result.scalar() is None:
                conn.execute(text(f"CREATE DATABASE {_TEST_DB_NAME}"))
    except Exception as exc:
        pytest.skip(f"Postgres database unavailable: {exc}")
    finally:
        engine.dispose()
    return _test_database_url()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Provide the test database URL, creating the database if needed."""
    return _ensure_test_database_exists()


@pytest.fixture(scope="session")
def test_engine(test_database_url: str) -> Generator[Engine, None, None]:
    """Provide a SQLAlchemy engine bound to the test database."""
    engine = create_engine(test_database_url, future=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def test_session(test_engine: Engine) -> Generator[Session, None, None]:
    """Provide a clean transactional session for a single test.

    Drops and recreates all tables before each test so tests are isolated.
    """
    Base.metadata.drop_all(bind=test_engine)
    _create_enums(test_engine)
    Base.metadata.create_all(bind=test_engine)
    session = sessionmaker(bind=test_engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_paper_pdf() -> bytes:
    """Generate a small PDF with sectioned text for chunking."""
    buffer = io.BytesIO()
    canva = canvas.Canvas(buffer, pagesize=letter)
    canva.drawString(72, 720, "1 Introduction")
    canva.drawString(72, 700, "This paper introduces a novel approach to RAG.")
    canva.drawString(72, 680, "We compare several retrieval strategies.")
    canva.drawString(72, 620, "2 Methods")
    canva.drawString(72, 600, "Our method uses pgvector for retrieval.")
    canva.drawString(72, 540, "2.1 Experimental Setup")
    canva.drawString(72, 520, "We evaluate on a dataset of papers.")
    canva.drawString(72, 460, "3 Results and Discussion")
    canva.drawString(72, 440, "The results show improved recall at k.")
    canva.showPage()
    canva.save()
    buffer.seek(0)
    return buffer.read()


def _create_enums(engine: Engine) -> None:
    """Create PostgreSQL ENUM types that have ``create_type=False`` in the schema."""
    enums = {
        "document_type": ["paper", "book", "documentation"],
        "document_status": ["validating", "chunking", "embedding", "ready", "failed"],
    }
    with engine.connect() as conn:
        for name, values in enums.items():
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


def override_route_db_session(test_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make API route ``db_session`` contexts yield sessions on the test engine."""
    _create_enums(test_engine)
    Base.metadata.create_all(bind=test_engine)
    SessionFactory = sessionmaker(bind=test_engine)

    @contextmanager
    def _test_db_session() -> Generator[Session, None, None]:
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("api.routes.ingest.db_session", _test_db_session)
    monkeypatch.setattr("api.routes.documents.db_session", _test_db_session)
    monkeypatch.setattr("ingestion.tasks.SessionLocal", SessionFactory)
