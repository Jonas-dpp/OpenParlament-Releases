"""Add lexical_results table

Revision ID: 294bce26fdf5
Revises: 0146aed12050
Create Date: 2026-07-09 00:33:31.089703

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '294bce26fdf5'
down_revision: Union[str, Sequence[str], None] = '0146aed12050'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('lexical_results',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('wahlperiode', sa.Integer(), nullable=False),
    sa.Column('fraktion', sa.String(length=128), nullable=False),
    sa.Column('ttr', sa.Float(), nullable=False),
    sa.Column('msttr', sa.Float(), nullable=True),
    sa.Column('lexical_density', sa.Float(), nullable=False),
    sa.Column('total_tokens', sa.Integer(), nullable=False),
    sa.Column('total_types', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('wahlperiode', 'fraktion', name='uq_lexical_wp_frak')
    )
    op.create_index(op.f('ix_lexical_results_fraktion'), 'lexical_results', ['fraktion'], unique=False)
    op.create_index(op.f('ix_lexical_results_wahlperiode'), 'lexical_results', ['wahlperiode'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_lexical_results_wahlperiode'), table_name='lexical_results')
    op.drop_index(op.f('ix_lexical_results_fraktion'), table_name='lexical_results')
    op.drop_table('lexical_results')
