"""SQLAlchemy engine and session factory with lazy initialisation.

The engine is not created at import time.  Use ``get_engine()`` / ``get_session()``
to obtain the singleton engine or a new session respectively.  Tests can inject
a test engine via ``set_engine()``.

``engine`` and ``SessionLocal`` remain importable as deprecated aliases that
delegate to the lazy functions so existing consumers continue to work.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config.settings import settings

_engine: Engine | None = None
_sessionmaker: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the application's SQLAlchemy engine, creating it lazily if needed.

    Subsequent calls return the same singleton until ``set_engine()`` replaces it.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, future=True)
    return _engine


def set_engine(test_engine: Engine) -> None:
    """Replace the engine singleton with *test_engine*.

    This is intended for testing — call once per test suite to inject a
    SQLite or test-Postgres engine.  The session-maker is reset so that
    subsequent ``get_session()`` calls bind to the new engine.
    """
    global _engine, _sessionmaker
    _engine = test_engine
    _sessionmaker = None


def get_session() -> Session:
    """Return a new ``Session`` bound to the current engine.

    The underlying ``sessionmaker`` is lazily created on first call and
    re-created after ``set_engine()`` replaces the engine.
    """
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _sessionmaker()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Yield a transactional database session.

    Commits on clean exit and rolls back on any exception.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Deprecated aliases ──────────────────────────────────────────────
# These exist so that existing imports continue to work.  Prefer
# ``get_engine()`` / ``get_session()`` in new code.
_MODULE_DIR = frozenset(
    {
        "db_session",
        "get_engine",
        "get_session",
        "set_engine",
        "engine",
        "SessionLocal",
    }
)


def __getattr__(name: str) -> Any:
    if name == "engine":
        return get_engine()
    if name == "SessionLocal":
        return get_session
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(_MODULE_DIR)
