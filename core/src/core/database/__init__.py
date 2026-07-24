"""Database access layer shared by all modules."""

from typing import Any

from core.database.connection import db_session, get_engine, get_session, set_engine
from core.database.schema import Base, Chunk, Document, Embedding

# ``engine`` and ``SessionLocal`` are accessed via ``__getattr__`` below to
# avoid creating the database engine at import time.

__all__ = [
    "Base",
    "Chunk",
    "Document",
    "Embedding",
    "SessionLocal",
    "db_session",
    "engine",
    "get_engine",
    "get_session",
    "set_engine",
]


def __getattr__(name: str) -> Any:
    if name in {"engine", "SessionLocal"}:
        from core.database.connection import __getattr__ as _conn_getattr

        return _conn_getattr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
