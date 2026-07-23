"""Add parent_chunk_id and content_search (tsvector) to chunks.

Revision ID: edda412e1f07
Revises: 2fff441e94ab
Create Date: 2026-07-23 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TSVECTOR

# revision identifiers, used by Alembic.
revision: str = "edda412e1f07"
down_revision: str | None = "2fff441e94ab"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "parent_chunk_id",
            sa.Uuid(),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_chunks_parent_chunk_id",
        "chunks",
        "chunks",
        ["parent_chunk_id"],
        ["chunk_id"],
        ondelete="SET NULL",
    )

    op.add_column(
        "chunks",
        sa.Column(
            "content_search",
            TSVECTOR(),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_chunks_content_search_gin",
        "chunks",
        ["content_search"],
        postgresql_using="gin",
    )

    op.drop_constraint(
        "uq_chunk_document_position",
        "chunks",
        type_="unique",
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_search_gin")
    op.drop_column("chunks", "content_search")
    op.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS fk_chunks_parent_chunk_id")
    op.drop_column("chunks", "parent_chunk_id")
