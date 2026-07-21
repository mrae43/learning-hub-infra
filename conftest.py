"""Shared pytest fixtures for database-backed tests.

Tests that need a real Postgres+pgvector database use the ``learning_hub_test``
database created on the Postgres instance pointed to by ``DATABASE_URL``.
If the database is unavailable, the fixtures skip the test.
"""

import io
import os
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from urllib.parse import urlparse

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Session, sessionmaker

# Pre-import workspace packages so their installed (src/) versions are cached
# in sys.modules before per-package test collection runs.  Without this a test
# package whose name mirrors a source package (e.g. ``tests/retrieval_qa``)
# would shadow the installed package under importlib import mode.
import api.controllers.qa_controller
import api.server  # noqa: F401
import ingestion.tasks  # noqa: F401
import retrieval_qa.chunking.book_chunker
import retrieval_qa.chunking.documentation_chunker
import retrieval_qa.chunking.paper_chunker
import retrieval_qa.retrieval.query  # noqa: F401

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
    with test_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
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


@pytest.fixture
def sample_book_pdf() -> bytes:
    """Generate a small PDF with chapter headings for chunking."""
    buffer = io.BytesIO()
    canva = canvas.Canvas(buffer, pagesize=letter)
    canva.drawString(72, 720, "Chapter 1")
    canva.drawString(72, 700, "The ancient city of Rome was built on seven hills.")
    canva.drawString(72, 680, "Summary")
    canva.drawString(72, 660, "Rome has seven hills.")
    canva.showPage()
    canva.drawString(72, 720, "Chapter 2")
    canva.drawString(72, 700, "The Roman Forum was the center of public life.")
    canva.showPage()
    canva.save()
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def sample_book_epub() -> bytes:
    """Generate a small EPUB with two chapters for chunking."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # mimetype must be first, stored (not compressed).
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        # Container XML
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<container version="1.0"'
            ' xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            "<rootfiles>"
            '<rootfile full-path="OEBPS/content.opf"'
            ' media-type="application/oebps-package+xml"/>'
            "</rootfiles>"
            "</container>",
        )
        # Content.opf — no NCX needed for this simple case
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf"'
            ' unique-identifier="bookid" version="2.0">'
            "<metadata>"
            '<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Sample Book</dc:title>'
            '<dc:identifier xmlns:dc="http://purl.org/dc/elements/1.1/"'
            ' id="bookid">book-test-001</dc:identifier>'
            '<dc:language xmlns:dc="http://purl.org/dc/elements/1.1/">en</dc:language>'
            "</metadata>"
            "<manifest>"
            '<item id="chap1" href="chap_1.xhtml"'
            ' media-type="application/xhtml+xml"/>'
            '<item id="chap2" href="chap_2.xhtml"'
            ' media-type="application/xhtml+xml"/>'
            "</manifest>"
            "<spine>"
            '<itemref idref="chap1"/>'
            '<itemref idref="chap2"/>'
            "</spine>"
            "</package>",
        )
        # Chapter files
        zf.writestr(
            "OEBPS/chap_1.xhtml",
            '<?xml version="1.0" encoding="utf-8"?>'
            "<!DOCTYPE html>"
            '<html xmlns="http://www.w3.org/1999/xhtml">'
            "<head><title>Chapter 1</title></head>"
            "<body>"
            "<h1>Chapter 1</h1>"
            "<p>The ancient city of Rome was built on seven hills.</p>"
            "<h2>Summary</h2>"
            "<p>Rome has seven hills.</p>"
            "</body></html>",
        )
        zf.writestr(
            "OEBPS/chap_2.xhtml",
            '<?xml version="1.0" encoding="utf-8"?>'
            "<!DOCTYPE html>"
            '<html xmlns="http://www.w3.org/1999/xhtml">'
            "<head><title>Chapter 2</title></head>"
            "<body>"
            "<h1>Chapter 2</h1>"
            "<p>The Roman Forum was the center of public life.</p>"
            "</body></html>",
        )
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def sample_documentation_md() -> bytes:
    """Generate a small Markdown documentation file with pages and API entries."""
    return (
        b"# Installation\n\n"
        b"Install the package with pip.\n\n"
        b"# API Reference\n\n"
        b"## Users\n\n"
        b"GET /api/v1/users\n\n"
        b"Returns a list of users.\n\n"
        b"POST /api/v1/users\n\n"
        b"Creates a new user.\n"
    )


def _create_enums(engine: Engine) -> None:
    """Create PostgreSQL ENUM types that have ``create_type=False`` in the schema."""
    enums = {
        "document_type": ["paper", "book", "documentation"],
        "document_status": ["validating", "chunking", "embedding", "ready", "failed"],
    }
    with engine.connect() as conn:
        for name, values in enums.items():
            enum_type = ENUM(*values, name=name)
            enum_type.create(bind=conn, checkfirst=True)
        conn.commit()


@pytest.fixture
def override_route_db_session(test_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make API route ``db_session`` contexts yield sessions on the test engine."""
    Base.metadata.drop_all(bind=test_engine)
    with test_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
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
    monkeypatch.setattr("api.routes.retrieval_qa.db_session", _test_db_session)
    monkeypatch.setattr("ingestion.tasks.SessionLocal", SessionFactory)
