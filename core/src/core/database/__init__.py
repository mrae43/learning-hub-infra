"""Database access layer shared by all modules."""

from core.database.connection import SessionLocal, db_session, engine
from core.database.schema import Base, Chunk, Document, Embedding

__all__ = [
    "Base",
    "Chunk",
    "Document",
    "Embedding",
    "SessionLocal",
    "db_session",
    "engine",
]
