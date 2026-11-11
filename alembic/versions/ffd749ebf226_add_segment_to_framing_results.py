"""add_segment_and_variant_to_framing_results

Revision ID: ffd749ebf226
Revises: 7a82b01c3d4f
Create Date: 2026-11-06 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ffd749ebf226'
down_revision: Union[str, Sequence[str], None] = '7a82b01c3d4f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    existing_tables = set(insp.get_table_names())

    if 'framing_results' in existing_tables:
        existing_cols = {c['name'] for c in insp.get_columns('framing_results')}
        existing_ucs = {uc['name'] for uc in insp.get_unique_constraints('framing_results')}
        existing_indexes = {idx['name'] for idx in insp.get_indexes('framing_results')}

        with op.batch_alter_table('framing_results', recreate='always') as batch_op:
            if 'segment' not in existing_cols:
                batch_op.add_column(sa.Column('segment', sa.String(length=32), nullable=False, server_default='all'))
            if 'variant' not in existing_cols:
                batch_op.add_column(sa.Column('variant', sa.String(length=32), nullable=False, server_default='global'))

            if 'ix_framing_results_segment' not in existing_indexes:
                batch_op.create_index('ix_framing_results_segment', ['segment'], unique=False)
            if 'ix_framing_results_variant' not in existing_indexes:
                batch_op.create_index('ix_framing_results_variant', ['variant'], unique=False)

            if 'uq_framing_wp_frak_term' in existing_ucs:
                batch_op.drop_constraint('uq_framing_wp_frak_term', type_='unique')
            if 'uq_framing_wp_frak_seg_term' in existing_ucs:
                batch_op.drop_constraint('uq_framing_wp_frak_seg_term', type_='unique')
            batch_op.create_unique_constraint('uq_framing_wp_frak_seg_var_term', ['wahlperiode', 'fraktion', 'segment', 'variant', 'term'])


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    existing_tables = set(insp.get_table_names())

    if 'framing_results' in existing_tables:
        # Pre-emptively deduplicate across segments and variants to avoid uniqueness collisions on (wahlperiode, fraktion, term)
        conn.execute(sa.text("""
            DELETE FROM framing_results
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM framing_results
                GROUP BY wahlperiode, fraktion, term
            )
        """))

        existing_cols = {c['name'] for c in insp.get_columns('framing_results')}
        with op.batch_alter_table('framing_results', recreate='auto') as batch_op:
            existing_ucs = {uc['name'] for uc in insp.get_unique_constraints('framing_results')}
            if 'uq_framing_wp_frak_seg_var_term' in existing_ucs:
                batch_op.drop_constraint('uq_framing_wp_frak_seg_var_term', type_='unique')
            if 'uq_framing_wp_frak_seg_term' in existing_ucs:
                batch_op.drop_constraint('uq_framing_wp_frak_seg_term', type_='unique')

            existing_indexes = {idx['name'] for idx in insp.get_indexes('framing_results')}
            if 'ix_framing_results_variant' in existing_indexes:
                batch_op.drop_index('ix_framing_results_variant')
            if 'ix_framing_results_segment' in existing_indexes:
                batch_op.drop_index('ix_framing_results_segment')

            if 'variant' in existing_cols:
                batch_op.drop_column('variant')
            if 'segment' in existing_cols:
                batch_op.drop_column('segment')

            batch_op.create_unique_constraint('uq_framing_wp_frak_term', ['wahlperiode', 'fraktion', 'term'])
