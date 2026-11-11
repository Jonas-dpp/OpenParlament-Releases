"""add_speech_complexity_and_reaction_gap_tables

Revision ID: 7a82b01c3d4f
Revises: e9d0356b9ab9
Create Date: 2026-11-04 22:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7a82b01c3d4f'
down_revision: Union[str, Sequence[str], None] = 'e9d0356b9ab9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    existing_tables = set(insp.get_table_names())

    # speech_complexity_results table
    if 'speech_complexity_results' not in existing_tables:
        op.create_table(
            'speech_complexity_results',
            sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
            sa.Column('rede_id', sa.Integer(), nullable=False),
            sa.Column('wahlperiode', sa.Integer(), nullable=False),
            sa.Column('redner_id', sa.Integer(), nullable=False),
            sa.Column('wstf_score', sa.Float(), nullable=False, server_default='0.0'),
            sa.Column('flesch_score', sa.Float(), nullable=False, server_default='0.0'),
            sa.Column('avg_sentence_length', sa.Float(), nullable=False),
            sa.Column('subordinate_clause_ratio', sa.Float(), nullable=False),
            sa.Column('avg_word_length', sa.Float(), nullable=False),
            sa.Column('long_word_ratio', sa.Float(), nullable=False),
            sa.Column('lexical_diversity', sa.Float(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['rede_id'], ['reden.rede_id']),
            sa.ForeignKeyConstraint(['redner_id'], ['redner.redner_id']),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('ix_speech_complexity_results_rede_id', 'speech_complexity_results', ['rede_id'])
        op.create_index('ix_speech_complexity_results_redner_id', 'speech_complexity_results', ['redner_id'])
        op.create_index('ix_speech_complexity_results_wahlperiode', 'speech_complexity_results', ['wahlperiode'])

    # reaction_gap_results table
    if 'reaction_gap_results' not in existing_tables:
        op.create_table(
            'reaction_gap_results',
            sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
            sa.Column('rede_id', sa.Integer(), nullable=False),
            sa.Column('zwischenruf_id', sa.Integer(), nullable=False),
            sa.Column('topic_id', sa.String(length=256), nullable=True),
            sa.Column('speaker_sentiment', sa.Float(), nullable=False),
            sa.Column('reaction_sentiment', sa.Float(), nullable=False),
            sa.Column('gap', sa.Float(), nullable=False),
            sa.Column('time_to_first_interjection', sa.Float(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['rede_id'], ['reden.rede_id']),
            sa.ForeignKeyConstraint(['zwischenruf_id'], ['zwischenrufe.ruf_id']),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('ix_reaction_gap_results_rede_id', 'reaction_gap_results', ['rede_id'])
        op.create_index('ix_reaction_gap_results_zwischenruf_id', 'reaction_gap_results', ['zwischenruf_id'])


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    existing_tables = set(insp.get_table_names())

    if 'reaction_gap_results' in existing_tables:
        existing_indexes = {idx['name'] for idx in insp.get_indexes('reaction_gap_results')}
        if 'ix_reaction_gap_results_zwischenruf_id' in existing_indexes:
            op.drop_index('ix_reaction_gap_results_zwischenruf_id', table_name='reaction_gap_results')
        if 'ix_reaction_gap_results_rede_id' in existing_indexes:
            op.drop_index('ix_reaction_gap_results_rede_id', table_name='reaction_gap_results')
        op.drop_table('reaction_gap_results')

    if 'speech_complexity_results' in existing_tables:
        existing_indexes = {idx['name'] for idx in insp.get_indexes('speech_complexity_results')}
        if 'ix_speech_complexity_results_wahlperiode' in existing_indexes:
            op.drop_index('ix_speech_complexity_results_wahlperiode', table_name='speech_complexity_results')
        if 'ix_speech_complexity_results_redner_id' in existing_indexes:
            op.drop_index('ix_speech_complexity_results_redner_id', table_name='speech_complexity_results')
        if 'ix_speech_complexity_results_rede_id' in existing_indexes:
            op.drop_index('ix_speech_complexity_results_rede_id', table_name='speech_complexity_results')
        op.drop_table('speech_complexity_results')
