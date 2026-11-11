from sqlalchemy import text
from src.database import get_session, init_db, get_engine


def test_sqlite_performance_pragmas():
    """Verify that SQLite connection performance pragmas are set correctly."""
    init_db()
    with get_session() as session:
        # Check WAL mode
        journal_mode = session.execute(text("PRAGMA journal_mode")).scalar()
        engine_url = str(get_engine().url)
        if ":memory:" in engine_url:
            assert journal_mode.upper() == "MEMORY"
        else:
            assert journal_mode.upper() == "WAL"

        # Check synchronous mode (NORMAL is 1)
        sync_mode = session.execute(text("PRAGMA synchronous")).scalar()
        assert sync_mode == 1

        # Check cache size (-64000)
        cache_size = session.execute(text("PRAGMA cache_size")).scalar()
        assert cache_size == -64000

        # Check temp store (MEMORY is 2)
        temp_store = session.execute(text("PRAGMA temp_store")).scalar()
        assert temp_store == 2


def test_sqlite_performance_indexes():
    """Verify that indices for highly-queried columns are correctly created."""
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        # Query existing indexes
        rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='index'")).fetchall()
        index_names = {row[0] for row in rows}

        expected_indexes = {
            "ix_zwischenrufe_sentiment_score",
            "ix_zwischenrufe_kategorie",
            "ix_zwischenrufe_ton_label",
            "ix_reden_sentiment_score",
            "ix_reden_tagesordnungspunkt",
        }

        for idx_name in expected_indexes:
            assert idx_name in index_names, f"Index {idx_name} was not created!"
