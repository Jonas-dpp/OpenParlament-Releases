#!/usr/bin/env python3
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
db_patch.py – Fügt fehlende Spalten zur SQLite-Datenbank hinzu und
backfüllt fehlende Werte, ohne bestehende Daten zu löschen.

Run from the project root:
    python scripts/db_patch.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import select, text  # noqa: E402
from src.database import _ensure_sqlite_fts, get_engine, get_session  # noqa: E402
from src.models import Sitzung  # noqa: E402

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("db_patch")


def _add_column_safe(conn, table: str, col_name: str, col_type: str) -> None:
    """ALTER TABLE … ADD COLUMN, silently skipping if the column already exists."""
    try:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"))
        logger.info("Added column %s.%s (%s)", table, col_name, col_type)
    except Exception as exc:
        if "duplicate column name" in str(exc).lower():
            logger.debug("Column %s.%s already exists – skipped.", table, col_name)
        else:
            logger.error("Error adding %s.%s: %s", table, col_name, exc)
            raise


def patch_schema() -> None:
    """Add any missing columns to existing tables."""
    engine = get_engine()
    logger.debug("Patching schema: %s", engine.url)
    
    # Create entirely new tables (like stylometrie_features) if they don't exist
    from src.models import Base
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        # framing_results
        logger.debug("--- Table: framing_results ---")
        # Check if table exists and if it has the legacy 3-column unique constraint
        table_exists = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='framing_results'")).fetchone()
        if table_exists:
            has_old_unique = False
            for idx in conn.execute(text("PRAGMA index_list(framing_results)")).fetchall():
                if idx[2] == 1:  # unique index
                    cols = [c[2] for c in conn.execute(text(f"PRAGMA index_info('{idx[1]}')")).fetchall()]
                    if cols in (['wahlperiode', 'fraktion', 'term'], ['wahlperiode', 'fraktion', 'segment', 'term']):
                        has_old_unique = True
                        break

            if has_old_unique:
                logger.info("Migrating framing_results table schema to uq_framing_wp_frak_seg_var_term...")
                existing_cols = {c[1] for c in conn.execute(text("PRAGMA table_info(framing_results)")).fetchall()}
                seg_expr = "segment" if "segment" in existing_cols else "'all'"
                var_expr = "variant" if "variant" in existing_cols else "'global'"
                conn.execute(text(f"""
                    CREATE TABLE framing_results_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        wahlperiode INTEGER NOT NULL,
                        fraktion VARCHAR(128) NOT NULL,
                        segment VARCHAR(32) NOT NULL DEFAULT 'all',
                        variant VARCHAR(32) NOT NULL DEFAULT 'global',
                        term VARCHAR(256) NOT NULL,
                        score FLOAT NOT NULL,
                        CONSTRAINT uq_framing_wp_frak_seg_var_term UNIQUE (wahlperiode, fraktion, segment, variant, term)
                    )
                """))
                conn.execute(text(f"""
                    INSERT INTO framing_results_new (id, wahlperiode, fraktion, segment, variant, term, score)
                    SELECT id, wahlperiode, fraktion, {seg_expr}, {var_expr}, term, score FROM framing_results
                """))
                conn.execute(text("DROP TABLE framing_results"))
                conn.execute(text("ALTER TABLE framing_results_new RENAME TO framing_results"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_framing_results_fraktion ON framing_results(fraktion)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_framing_results_wahlperiode ON framing_results(wahlperiode)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_framing_results_segment ON framing_results(segment)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_framing_results_variant ON framing_results(variant)"))
            else:
                _add_column_safe(conn, "framing_results", "segment", "VARCHAR(32) DEFAULT 'all'")
                _add_column_safe(conn, "framing_results", "variant", "VARCHAR(32) DEFAULT 'global'")

        # sitzungen
        logger.debug("--- Table: sitzungen ---")
        _add_column_safe(conn, "sitzungen", "wochentag", "VARCHAR(16)")

        # zwischenrufe
        logger.debug("--- Table: zwischenrufe ---")
        _add_column_safe(conn, "zwischenrufe", "tone_scores", "JSON")
        _add_column_safe(conn, "zwischenrufe", "tone_aggression", "FLOAT")
        _add_column_safe(conn, "zwischenrufe", "tone_sarkasmus", "FLOAT")
        _add_column_safe(conn, "zwischenrufe", "tone_humor", "FLOAT")
        _add_column_safe(conn, "zwischenrufe", "tone_neutral", "FLOAT")

        # reden
        logger.debug("--- Table: reden ---")
        for col_name, col_type in [
            ("raw_text",            "TEXT"),
            ("clean_text",          "TEXT"),
            ("normalized_text",     "TEXT"),
            ("metadata_text",       "TEXT"),
            ("processing_version",  "INTEGER DEFAULT 1"),
            ("processed_at",        "DATETIME"),
            ("char_count_raw",      "INTEGER DEFAULT 0"),
            ("char_count_clean",    "INTEGER DEFAULT 0"),
            ("char_count_metadata", "INTEGER DEFAULT 0"),
            ("removed_chars",       "INTEGER DEFAULT 0"),
            ("removed_percent",     "FLOAT DEFAULT 0.0"),
            ("count_applause",      "INTEGER DEFAULT 0"),
            ("count_heckling",       "INTEGER DEFAULT 0"),
            ("count_laughter",      "INTEGER DEFAULT 0"),
            ("count_president",     "INTEGER DEFAULT 0"),
            ("processing_time_ms",  "INTEGER DEFAULT 0"),
            ("sentiment_score",     "FLOAT"),
            ("ton_label",           "VARCHAR(64)"),
            ("tone_scores",         "JSON"),
            ("tone_aggression",     "FLOAT"),
            ("tone_sarkasmus",      "FLOAT"),
            ("tone_humor",          "FLOAT"),
            ("tone_neutral",        "FLOAT"),
            ("adressaten",          "VARCHAR(512)"),
        ]:
            _add_column_safe(conn, "reden", col_name, col_type)


def _tone_col_from_json_expr(table: str, key: str) -> str:
    return f"""
        {table}.tone_scores IS NOT NULL
        AND {table}.tone_scores != ''
        AND {table}.tone_scores != 'null'
        AND json_valid({table}.tone_scores)
        AND json_type({table}.tone_scores) = 'object'
        AND json_extract({table}.tone_scores, '$.{key}') IS NOT NULL
    """


def patch_tone_columns_backfill() -> None:
    """Backfill dedicated tone columns from legacy tone_scores JSON where possible."""
    engine = get_engine()
    logger.debug("Backfilling tone columns from tone_scores JSON where missing.")
    with engine.begin() as conn:
        for table in ("reden", "zwischenrufe"):
            conn.execute(
                text(
                    f"""
                    UPDATE {table}
                    SET
                        tone_aggression = COALESCE(tone_aggression, CAST(json_extract(tone_scores, '$.Aggression') AS FLOAT)),
                        tone_sarkasmus = COALESCE(tone_sarkasmus, CAST(json_extract(tone_scores, '$.Sarkasmus') AS FLOAT)),
                        tone_humor = COALESCE(tone_humor, CAST(json_extract(tone_scores, '$.Humor') AS FLOAT)),
                        tone_neutral = COALESCE(tone_neutral, CAST(json_extract(tone_scores, '$.Neutral') AS FLOAT))
                    WHERE
                        ({_tone_col_from_json_expr(table, "Aggression")} AND tone_aggression IS NULL)
                        OR ({_tone_col_from_json_expr(table, "Sarkasmus")} AND tone_sarkasmus IS NULL)
                        OR ({_tone_col_from_json_expr(table, "Humor")} AND tone_humor IS NULL)
                        OR ({_tone_col_from_json_expr(table, "Neutral")} AND tone_neutral IS NULL)
                    """
                )
            )


def patch_sitzung_dates() -> None:
    """Backfill datum and wochentag for Sitzungen where either value is NULL.

    Requires the corresponding XML files to be present in data/xml/.
    Only sessions whose XML file exists locally are updated.
    """
    from src.parser import BundestagXMLParser
    from src.parser import weekday_german

    xml_dir = _PROJECT_ROOT / "data" / "xml"
    if not xml_dir.exists():
        logger.warning("XML directory not found (%s) – skipping date backfill.", xml_dir)
        return

    parser = BundestagXMLParser()

    with get_session() as session:
        # Only process sessions that are still missing datum or wochentag.
        rows = session.execute(
            select(Sitzung).where(
                (Sitzung.datum.is_(None)) | (Sitzung.wochentag.is_(None))
            )
        ).scalars().all()

        if not rows:
            logger.info("All Sitzungen already have datum and wochentag – nothing to do.")
            return

        total = len(rows)
        logger.info("%d Sitzung(en) with missing datum/wochentag found.", total)
        updated = 0
        start_time = time.time()

        for idx, sitzung in enumerate(rows, start=1):
            xml_file = xml_dir / f"{sitzung.wahlperiode:02d}{sitzung.sitzungsnr:03d}.xml"
            if not xml_file.exists():
                logger.debug("XML not found locally: %s – skipping.", xml_file.name)
                # Throttling logging: log every 5 or on the last iteration
                if idx % 5 == 0 or idx == total:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    eta = avg_time * (total - idx)
                    if eta >= 3600:
                        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                    else:
                        eta_str = time.strftime('%M:%S', time.gmtime(eta))
                    logger.info(
                        f"PatchSitzungDates: {idx}/{total} | {int((idx/total)*100)}% | ETA: {eta_str}"
                    )
                continue
            try:
                result = parser.parse_file(xml_file)
                changed = False
                if sitzung.datum is None and result.sitzung.datum is not None:
                    sitzung.datum = result.sitzung.datum
                    changed = True
                if sitzung.wochentag is None and sitzung.datum is not None:
                    sitzung.wochentag = weekday_german(sitzung.datum)
                    changed = True
                if changed:
                    updated += 1
                    logger.debug(
                        "Updated WP%d/%d → datum=%s wochentag=%s",
                        sitzung.wahlperiode, sitzung.sitzungsnr,
                        sitzung.datum, sitzung.wochentag,
                    )
            except Exception:
                logger.exception("Failed to re-parse %s", xml_file.name)

            # Throttling logging: log every 5 or on the last iteration
            if idx % 5 == 0 or idx == total:
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                eta = avg_time * (total - idx)
                if eta >= 3600:
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                else:
                    eta_str = time.strftime('%M:%S', time.gmtime(eta))
                logger.info(
                    f"PatchSitzungDates: {idx}/{total} | {int((idx/total)*100)}% | ETA: {eta_str}"
                )

        session.commit()
        logger.info("Backfilled %d/%d session(s).", updated, total)


def patch_database() -> None:
    patch_schema()
    patch_tone_columns_backfill()
    _ensure_sqlite_fts(get_engine())
    patch_sitzung_dates()
    logger.info("Patch complete.")


if __name__ == "__main__":
    patch_database()
