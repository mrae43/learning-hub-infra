"""SQLAlchemy declarative base and table models.

The schema matches ADR-0014:
- UUIDv7 primary keys generated app-side.
- Postgres enums for ``document_type`` and ``status`` with lowercase labels that
  match the Python ``StrEnum`` values.
- CHECK constraints on ``error_message``, ``position``, and ``token_count``.
- ``UNIQUE (document_id, position)`` on chunks.
- Composite PK ``(chunk_id, model_name)`` on embeddings.
- No ``updated_at`` on chunks or embeddings.
- HNSW index created in the Alembic migration, not here.
"""

from datetime import UTC, datetime
from uuid import UUID

import uuid_utils
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.sql.elements import TextClause

from core.types.document import DocumentStatus, DocumentType


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _uuid7_stdlib() -> UUID:
    """Return a stdlib ``UUID`` (v7) for compatibility with psycopg2 and Pydantic."""
    return UUID(int=uuid_utils.uuid7().int)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Document(Base):
    """An uploaded document and its ingestion status."""

    __tablename__ = "documents"

    document_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=_uuid7_stdlib,
    )
    title: Mapped[str] = mapped_column(Text())
    document_type: Mapped[DocumentType] = mapped_column(
        ENUM(
            DocumentType,
            name="document_type",
            create_type=False,
            values_callable=lambda enum: [e.value for e in enum],
        ),
    )
    source_filename: Mapped[str] = mapped_column(Text())
    status: Mapped[DocumentStatus] = mapped_column(
        ENUM(
            DocumentStatus,
            name="document_status",
            create_type=False,
            values_callable=lambda enum: [e.value for e in enum],
        ),
        default=DocumentStatus.VALIDATING,
    )
    error_message: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "error_message IS NULL OR status = 'failed'",
            name="ck_error_message_only_when_failed",
        ),
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class Chunk(Base):
    """An atomic retrievable unit produced by a document-type chunker."""

    __tablename__ = "chunks"

    chunk_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=_uuid7_stdlib,
    )
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"),
    )
    position: Mapped[int]
    content: Mapped[str]
    token_count: Mapped[int]
    type_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        server_default=func.now(),
    )

    __table_args__ = (
        CheckConstraint("position >= 0", name="ck_chunk_position_non_negative"),
        CheckConstraint("token_count > 0", name="ck_chunk_token_count_positive"),
        UniqueConstraint("document_id", "position", name="uq_chunk_document_position"),
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
    embeddings: Mapped[list["Embedding"]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
    )


class Embedding(Base):
    """A vector embedding for a chunk under a specific model."""

    __tablename__ = "embeddings"

    chunk_id: Mapped[UUID] = mapped_column(
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    )
    model_name: Mapped[str] = mapped_column(primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        server_default=func.now(),
    )

    chunk: Mapped["Chunk"] = relationship(back_populates="embeddings")


def ensure_vector_extension_query() -> TextClause:
    """Return a raw SQL expression that creates the pgvector extension."""
    return text("CREATE EXTENSION IF NOT EXISTS vector")
