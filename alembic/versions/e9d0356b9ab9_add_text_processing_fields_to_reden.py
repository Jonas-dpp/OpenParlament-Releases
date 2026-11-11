"""add_text_processing_fields_to_reden

Revision ID: e9d0356b9ab9
Revises: 294bce26fdf5
Create Date: 2026-07-11 15:32:19.919971

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9d0356b9ab9'
down_revision: Union[str, Sequence[str], None] = '294bce26fdf5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Text and versioning fields
    op.add_column('reden', sa.Column('raw_text', sa.Text(), nullable=True))
    op.add_column('reden', sa.Column('clean_text', sa.Text(), nullable=True))
    op.add_column('reden', sa.Column('normalized_text', sa.Text(), nullable=True))
    op.add_column('reden', sa.Column('metadata_text', sa.Text(), nullable=True))
    op.add_column('reden', sa.Column('processing_version', sa.Integer(), nullable=False, server_default='1'))
    op.add_column('reden', sa.Column('processed_at', sa.DateTime(), nullable=True))

    # Quality metrics fields
    op.add_column('reden', sa.Column('char_count_raw', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('reden', sa.Column('char_count_clean', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('reden', sa.Column('char_count_metadata', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('reden', sa.Column('removed_chars', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('reden', sa.Column('removed_percent', sa.Float(), nullable=False, server_default='0.0'))
    op.add_column('reden', sa.Column('count_applause', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('reden', sa.Column('count_heckling', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('reden', sa.Column('count_laughter', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('reden', sa.Column('count_president', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('reden', sa.Column('processing_time_ms', sa.Integer(), nullable=False, server_default='0'))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('reden', 'processing_time_ms')
    op.drop_column('reden', 'count_president')
    op.drop_column('reden', 'count_laughter')
    op.drop_column('reden', 'count_heckling')
    op.drop_column('reden', 'count_applause')
    op.drop_column('reden', 'removed_percent')
    op.drop_column('reden', 'removed_chars')
    op.drop_column('reden', 'char_count_metadata')
    op.drop_column('reden', 'char_count_clean')
    op.drop_column('reden', 'char_count_raw')
    op.drop_column('reden', 'processed_at')
    op.drop_column('reden', 'processing_version')
    op.drop_column('reden', 'metadata_text')
    op.drop_column('reden', 'normalized_text')
    op.drop_column('reden', 'clean_text')
    op.drop_column('reden', 'raw_text')
