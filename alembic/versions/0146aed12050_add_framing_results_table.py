"""Add framing_results table

Revision ID: 0146aed12050
Revises:
Create Date: 2026-07-08 12:42:08.466892

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0146aed12050'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('framing_results',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('wahlperiode', sa.Integer(), nullable=False),
    sa.Column('fraktion', sa.String(length=128), nullable=False),
    sa.Column('term', sa.String(length=256), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('wahlperiode', 'fraktion', 'term', name='uq_framing_wp_frak_term')
    )
    op.create_index(op.f('ix_framing_results_fraktion'), 'framing_results', ['fraktion'], unique=False)
    op.create_index(op.f('ix_framing_results_wahlperiode'), 'framing_results', ['wahlperiode'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_framing_results_wahlperiode'), table_name='framing_results')
    op.drop_index(op.f('ix_framing_results_fraktion'), table_name='framing_results')
    op.drop_table('framing_results')
