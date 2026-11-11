"""
OpenParlament - Streamlit Dashboard
Copyright (C) 2026 Jonas-dpp

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

--------------------------------------------------------------------------
Database engine and session management for OpenParlament.

Usage
─────
    from src.database import get_session, init_db

    init_db()  # creates all tables (idempotent)

    with get_session() as session:
        session.add(some_object)
        # auto-commit on exit, rollback on exception
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, delete, Engine, event, update
from sqlalchemy.orm import Session, sessionmaker

from src.models import (
    Base,
    FramingResult,
    LexicalResult,
    ReactionGapResult,
    Rede,
    Zwischenruf,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "openparlament.db"


def _get_db_url() -> str:
    """Return the database URL.

    Reads ``OPENPARLAMENT_DB_URL`` from the environment if set, otherwise falls
    back to the default SQLite file path inside the ``data/`` directory.
    """
    env_url = os.environ.get("OPENPARLAMENT_DB_URL")
    if env_url:
        return env_url
    _DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{_DEFAULT_DB_PATH}"


# ─────────────────────────────────────────────────────────────────────────────
# Engine factory
# ─────────────────────────────────────────────────────────────────────────────

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return (and lazily create) the singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        db_url = _get_db_url()
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            connect_args["timeout"] = 30.0
        _engine = create_engine(db_url, connect_args=connect_args, echo=False)
        # Enable WAL mode, foreign-key enforcement, and other high-performance pragmas for SQLite.
        if db_url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragmas(dbapi_conn, _connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout to prevent 'database is locked' errors
                cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache to speed up repeated queries in Streamlit
                cursor.execute("PRAGMA temp_store=MEMORY")  # Keep temporary tables and indexes in RAM
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping to avoid slow read() system calls
                cursor.close()
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Session factory
# ─────────────────────────────────────────────────────────────────────────────

_SessionFactory: sessionmaker | None = None


def _get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autoflush=True, autocommit=False)
    return _SessionFactory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session.

    Commits automatically on successful exit; rolls back on any exception.

    Example::

        with get_session() as session:
            session.add(obj)
    """
    factory = _get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all database tables (idempotent – safe to call multiple times)."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_fts(engine)
    try:
        from scripts.db_patch import patch_schema
        patch_schema()
    except Exception:
        pass
    _ensure_indexes(engine)


def _ensure_indexes(engine: Engine) -> None:
    """Ensure that frequently queried columns have index constraints."""
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_zwischenrufe_sentiment_score ON zwischenrufe(sentiment_score)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_zwischenrufe_kategorie ON zwischenrufe(kategorie)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_zwischenrufe_ton_label ON zwischenrufe(ton_label)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_reden_sentiment_score ON reden(sentiment_score)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_reden_tagesordnungspunkt ON reden(tagesordnungspunkt)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_zwischenrufe_rede_sentiment ON zwischenrufe(rede_id, sentiment_score)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_zwischenrufe_fraktion_ton ON zwischenrufe(fraktion, ton_label)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_reden_sitzung_sentiment ON reden(sitzung_id, sentiment_score)")


def _ensure_sqlite_fts(engine: Engine) -> None:
    """Create and maintain the FTS5 index for speech text on SQLite backends."""
    if not str(engine.url).startswith("sqlite"):
        return

    with engine.begin() as conn:
        # External-content table keeps rowid aligned with reden.rede_id.
        conn.exec_driver_sql(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS reden_fts
            USING fts5(
                text,
                content='reden',
                content_rowid='rede_id',
                tokenize='unicode61'
            )
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS reden_ai AFTER INSERT ON reden BEGIN
              INSERT INTO reden_fts(rowid, text) VALUES (new.rede_id, COALESCE(new.text, ''));
            END
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS reden_ad AFTER DELETE ON reden BEGIN
              INSERT INTO reden_fts(reden_fts, rowid, text) VALUES('delete', old.rede_id, COALESCE(old.text, ''));
            END
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS reden_au AFTER UPDATE OF text ON reden BEGIN
              INSERT INTO reden_fts(reden_fts, rowid, text) VALUES('delete', old.rede_id, COALESCE(old.text, ''));
              INSERT INTO reden_fts(rowid, text) VALUES (new.rede_id, COALESCE(new.text, ''));
            END
            """
        )
        # Ensure existing rows are present in the index (only if the FTS table is empty).
        has_any = conn.exec_driver_sql("SELECT 1 FROM reden_fts LIMIT 1").fetchone()
        if has_any is None:
            conn.exec_driver_sql("INSERT INTO reden_fts(reden_fts) VALUES('rebuild')")


def reset_framing(session: Session | None = None) -> None:
    """Reset all framing and lexical analysis data.

    Deletes all records from ``framing_results`` and ``lexical_results``.
    If no session is provided, manages its own transactional session.
    """
    def _do_reset(s: Session) -> None:
        s.execute(delete(FramingResult))
        s.execute(delete(LexicalResult))

    if session is not None:
        _do_reset(session)
    else:
        with get_session() as s:
            _do_reset(s)


def reset_nlp(session: Session | None = None) -> None:
    """Reset all NLP inference results.

    Sets sentiment, tone, and addressee columns to NULL on ``reden`` and
    ``zwischenrufe``, and removes downstream sentiment-dependent records in
    ``reaction_gap_results``.
    If no session is provided, manages its own transactional session.
    """
    def _do_reset(s: Session) -> None:
        s.execute(
            update(Rede).values(
                sentiment_score=None,
                ton_label=None,
                tone_aggression=None,
                tone_sarkasmus=None,
                tone_humor=None,
                tone_neutral=None,
                tone_scores=None,
                adressaten=None,
            )
        )
        s.execute(
            update(Zwischenruf).values(
                sentiment_score=None,
                ton_label=None,
                tone_aggression=None,
                tone_sarkasmus=None,
                tone_humor=None,
                tone_neutral=None,
                tone_scores=None,
                adressaten=None,
            )
        )
        s.execute(delete(ReactionGapResult))

    if session is not None:
        _do_reset(session)
    else:
        with get_session() as s:
            _do_reset(s)


def drop_db() -> None:
    """Drop all tables (DANGER: destroys all data – for testing only)."""
    Base.metadata.drop_all(bind=get_engine())


def reset_db_state() -> None:
    """Reset the cached engine and session factory (for testing only).

    Call this after changing ``OPENPARLAMENT_DB_URL`` (e.g. via
    ``monkeypatch.setenv``) so that the next call to :func:`get_engine`
    picks up the new URL instead of reusing the previously cached engine.
    """
    global _engine, _SessionFactory
    _engine = None
    _SessionFactory = None
