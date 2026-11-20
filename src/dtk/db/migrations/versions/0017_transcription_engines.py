"""Distinguish local and external transcription engines.

Revision ID: 0017
Revises: 0016
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | Sequence[str] | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "transcriptions",
        sa.Column("engine", sa.Text(), server_default=sa.text("'local'"), nullable=False),
    )
    op.create_check_constraint(
        "ck_transcriptions_engine",
        "transcriptions",
        "engine IN ('local','qwen')",
    )
    op.drop_index("ux_transcriptions_in_flight", table_name="transcriptions")
    op.create_index(
        "ux_transcriptions_in_flight",
        "transcriptions",
        [
            "download_id",
            "source_sha256",
            "engine",
            "model",
            "normalization_requested",
            sa.text("COALESCE(normalizer, '')"),
            sa.text("COALESCE(normalization_failure_policy, '')"),
        ],
        unique=True,
        postgresql_where=sa.text("state IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("ux_transcriptions_in_flight", table_name="transcriptions")
    op.create_index(
        "ux_transcriptions_in_flight",
        "transcriptions",
        [
            "download_id",
            "source_sha256",
            "model",
            "normalization_requested",
            sa.text("COALESCE(normalizer, '')"),
            sa.text("COALESCE(normalization_failure_policy, '')"),
        ],
        unique=True,
        postgresql_where=sa.text("state IN ('queued', 'running')"),
    )
    op.drop_constraint("ck_transcriptions_engine", "transcriptions", type_="check")
    op.drop_column("transcriptions", "engine")
