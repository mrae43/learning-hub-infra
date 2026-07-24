"""Database access layer shared by all modules."""

from core.database.connection import db_session, get_engine, get_session, set_engine
from core.database.schema import Base, Chunk, Document, Embedding

__all__ = [
    "Base",
    "Chunk",
    "Document",
    "Embedding",
    "db_session",
    "get_engine",
    "get_session",
    "set_engine",
]
