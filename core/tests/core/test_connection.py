"""Tests for lazy engine initialisation in core.database.connection."""

import importlib

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from core.database.connection import get_engine, get_session, set_engine


def test_engine_not_created_at_import_time() -> None:
    """Verify importing the module does not trigger engine creation."""
    import core.database.connection as conn

    importlib.reload(conn)
    assert conn._engine is None


def test_get_engine_creates_engine_on_first_call() -> None:
    """get_engine() creates the engine from settings the first time it is called."""
    engine = get_engine()
    assert isinstance(engine, Engine)
    assert str(engine.url) != ""


def test_get_engine_returns_same_singleton() -> None:
    """Subsequent get_engine() calls return the same engine instance."""
    first = get_engine()
    second = get_engine()
    assert first is second


def test_set_engine_replaces_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_engine() replaces the engine singleton for testing."""
    import core.database.connection as conn

    monkeypatch.setattr(conn, "_engine", None)
    monkeypatch.setattr(conn, "_sessionmaker", None)

    test_engine = create_engine("sqlite:///:memory:", future=True)
    set_engine(test_engine)
    assert get_engine() is test_engine


def test_get_session_returns_session() -> None:
    """get_session() returns a new SQLAlchemy Session."""
    session = get_session()
    assert isinstance(session, Session)
    session.close()


def test_deprecated_engine_importable() -> None:
    """The deprecated ``engine`` alias is importable and returns an Engine."""
    from core.database.connection import engine

    assert isinstance(engine, Engine)


def test_deprecated_session_local_importable() -> None:
    """The deprecated ``SessionLocal`` alias is importable and callable."""
    from core.database.connection import SessionLocal

    assert callable(SessionLocal)
    session = SessionLocal()
    assert isinstance(session, Session)
    session.close()


def test_deprecated_aliases_from_package_init() -> None:
    """``engine`` and ``SessionLocal`` are accessible via ``core.database``."""
    from core.database import SessionLocal, engine

    assert isinstance(engine, Engine)
    assert callable(SessionLocal)


def test_db_session_uses_get_session() -> None:
    """db_session() yields a valid session and commits/rollbacks correctly."""
    from core.database.connection import db_session

    with db_session() as session:
        assert isinstance(session, Session)
