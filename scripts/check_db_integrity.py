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
check_db_integrity.py – Database Integrity and Completeness Verification Script.

Checks for:
  1. Missing database tables and schema column mismatches (ERRORS).
  2. Non-linear session numbers (gaps, start offset, duplicates, date inversions).
  3. Missing base table fields expected upon XML import (ERRORS).
  4. Missing precomputation/NLP fields expected after post-processing (WARNINGS).
  5. Precomputed table coverage and artifact availability.

Usage:
    python scripts/check_db_integrity.py [--json] [--verbose] [--strict] [--output PATH] [--db-url URL]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is on sys.path when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import create_engine, func, inspect, select  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from src.database import get_session, init_db  # noqa: E402
from src.models import (  # noqa: E402
    FramingResult,
    LexicalResult,
    Rede,
    Redner,
    ReactionGapResult,
    Sitzung,
    SpeechComplexityResult,
    StylometrieFeatures,
    Zwischenruf,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("check_db_integrity")


def check_schema_integrity(session: Session) -> Tuple[Dict[str, Any], Set[str]]:
    """Verify that all expected ORM tables and columns exist in the database schema."""
    results: Dict[str, Any] = {
        "status": "OK",
        "errors": [],
        "warnings": [],
        "tables": {},
    }

    inspector = inspect(session.bind)
    existing_tables = set(inspector.get_table_names())
    existing_columns: Dict[str, Set[str]] = {}

    expected_models = [
        Sitzung,
        Redner,
        Rede,
        Zwischenruf,
        StylometrieFeatures,
        FramingResult,
        LexicalResult,
        SpeechComplexityResult,
        ReactionGapResult,
    ]

    valid_cols_set: Set[str] = set()
    total_models = len(expected_models)
    start_time = time.time()

    for idx, model in enumerate(expected_models, start=1):
        table_name = model.__tablename__
        if table_name not in existing_tables:
            msg = f"Schema Error: Table '{table_name}' does not exist in database."
            results["errors"].append(msg)
            logger.error(msg)
            continue

        table_cols = {col["name"] for col in inspector.get_columns(table_name)}
        existing_columns[table_name] = table_cols

        model_cols = {col.name for col in model.__table__.columns}
        missing_cols = model_cols - table_cols

        if missing_cols:
            msg = f"Schema Error: Table '{table_name}' is missing columns: {sorted(list(missing_cols))}"
            results["errors"].append(msg)
            logger.error(msg)

        for c in table_cols:
            valid_cols_set.add(f"{table_name}.{c}")

        results["tables"][table_name] = {
            "exists": True,
            "column_count": len(table_cols),
            "missing_columns": sorted(list(missing_cols)),
        }

        # Telemetry: Progress and ETA calculation per checked model
        if total_models > 0 and (idx == 1 or idx % 3 == 0 or idx == total_models):
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            eta = avg_time * (total_models - idx)
            eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
            logger.info(
                f"SchemaIntegrityCheck: {idx}/{total_models} | {int((idx/total_models)*100)}% | ETA: {eta_str}"
            )

    if results["errors"]:
        results["status"] = "ERROR"

    return results, valid_cols_set


def check_sitzungen_sequence(session: Session) -> Dict[str, Any]:
    """Check for non-linear Sitzung counts, missing session numbers, duplicates, and date inversions."""
    results: Dict[str, Any] = {
        "status": "OK",
        "errors": [],
        "warnings": [],
        "wahlperioden": {},
        "total_sitzungen": 0,
    }

    sitzungen = session.execute(
        select(Sitzung).order_by(Sitzung.wahlperiode, Sitzung.sitzungsnr)
    ).scalars().all()

    results["total_sitzungen"] = len(sitzungen)
    if not sitzungen:
        results["warnings"].append("No Sitzungen found in database.")
        results["status"] = "WARNING"
        return results

    # Group by wahlperiode
    by_wp: Dict[int, List[Sitzung]] = {}
    for s in sitzungen:
        by_wp.setdefault(s.wahlperiode, []).append(s)

    sorted_wps = sorted(by_wp.items())
    total_wps = len(sorted_wps)
    start_time = time.time()
    log_step = max(1, total_wps // 10)

    for idx, (wp, s_list) in enumerate(sorted_wps, start=1):
        numbers = [s.sitzungsnr for s in s_list]
        min_nr = min(numbers)
        max_nr = max(numbers)
        count = len(numbers)

        wp_info: Dict[str, Any] = {
            "count": count,
            "min_sitzungsnr": min_nr,
            "max_sitzungsnr": max_nr,
            "missing_sitzungsnr": [],
            "duplicate_sitzungsnr": [],
            "date_inversions": [],
            "start_offset": 0,
        }

        # Check for duplicates
        seen = set()
        duplicates = set()
        for nr in numbers:
            if nr in seen:
                duplicates.add(nr)
            seen.add(nr)

        if duplicates:
            sorted_dups = sorted(list(duplicates))
            wp_info["duplicate_sitzungsnr"] = sorted_dups
            msg = f"WP {wp}: Found duplicate Sitzungsnummern: {sorted_dups}"
            results["errors"].append(msg)
            logger.error(msg)

        # Check start offset (session sequence typically starts at 1)
        if min_nr > 1:
            wp_info["start_offset"] = min_nr - 1
            msg = f"WP {wp}: Session numbers start at {min_nr} (missing sessions 1..{min_nr - 1})"
            results["warnings"].append(msg)
            logger.warning(msg)

        # Check missing numbers in full sequence [1 .. max_nr]
        expected_range = set(range(1, max_nr + 1))
        missing = sorted(list(expected_range - set(numbers)))
        if missing:
            wp_info["missing_sitzungsnr"] = missing
            msg = f"WP {wp}: Found {len(missing)} missing Sitzung(en) in sequence range [1..{max_nr}]: {missing[:10]}{'...' if len(missing) > 10 else ''}"
            results["warnings"].append(msg)
            logger.warning(msg)

        # Check date chronology inversions
        prev_sitzung: Optional[Sitzung] = None
        for s in s_list:
            if prev_sitzung and prev_sitzung.datum and s.datum:
                if s.datum < prev_sitzung.datum:
                    inversion = {
                        "prev_sitzungsnr": prev_sitzung.sitzungsnr,
                        "prev_datum": str(prev_sitzung.datum),
                        "curr_sitzungsnr": s.sitzungsnr,
                        "curr_datum": str(s.datum),
                    }
                    wp_info["date_inversions"].append(inversion)
                    msg = f"WP {wp}: Date inversion between Sitzung {prev_sitzung.sitzungsnr} ({prev_sitzung.datum}) and Sitzung {s.sitzungsnr} ({s.datum})"
                    results["warnings"].append(msg)
                    logger.warning(msg)
            prev_sitzung = s

        results["wahlperioden"][wp] = wp_info

        # Telemetry: Progress and ETA calculation per Wahlperiode
        if total_wps > 0 and (idx == 1 or idx % log_step == 0 or idx == total_wps):
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            eta = avg_time * (total_wps - idx)
            eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
            logger.info(
                f"SitzungenSequenceCheck: {idx}/{total_wps} | {int((idx/total_wps)*100)}% | ETA: {eta_str}"
            )

    if results["errors"]:
        results["status"] = "ERROR"
    elif results["warnings"]:
        results["status"] = "WARNING"

    return results


def check_unfilled_base_cells(session: Session, valid_cols: Set[str]) -> Dict[str, Any]:
    """Check for unfilled cells in base database tables (expected to be present upon XML import)."""
    results: Dict[str, Any] = {
        "status": "OK",
        "errors": [],
        "warnings": [],
        "counts": {},
    }

    def _col_exists(table: str, col: str) -> bool:
        return f"{table}.{col}" in valid_cols

    # 1. Sitzung Base Integrity
    sitzung_missing_datum = (
        session.execute(select(func.count(Sitzung.sitzungs_id)).where(Sitzung.datum.is_(None))).scalar() or 0
        if _col_exists("sitzungen", "datum") else 0
    )

    sitzung_missing_wochentag = (
        session.execute(select(func.count(Sitzung.sitzungs_id)).where(Sitzung.wochentag.is_(None))).scalar() or 0
        if _col_exists("sitzungen", "wochentag") else 0
    )

    sitzung_missing_titel = (
        session.execute(
            select(func.count(Sitzung.sitzungs_id)).where((Sitzung.titel.is_(None)) | (Sitzung.titel == ""))
        ).scalar() or 0
        if _col_exists("sitzungen", "titel") else 0
    )

    sitzung_zero_wortzahl = (
        session.execute(select(func.count(Sitzung.sitzungs_id)).where(Sitzung.gesamtwortzahl <= 0)).scalar() or 0
        if _col_exists("sitzungen", "gesamtwortzahl") else 0
    )

    sitzung_without_reden = session.execute(
        select(func.count(Sitzung.sitzungs_id)).where(
            ~Sitzung.sitzungs_id.in_(select(Rede.sitzung_id).distinct())
        )
    ).scalar() or 0

    results["counts"]["sitzungen"] = {
        "missing_datum": sitzung_missing_datum,
        "missing_wochentag": sitzung_missing_wochentag,
        "missing_titel": sitzung_missing_titel,
        "zero_wortzahl": sitzung_zero_wortzahl,
        "without_reden": sitzung_without_reden,
    }

    if sitzung_missing_datum > 0:
        results["errors"].append(f"Sitzung: {sitzung_missing_datum} session(s) missing 'datum'.")
    if sitzung_missing_wochentag > 0:
        results["warnings"].append(f"Sitzung: {sitzung_missing_wochentag} session(s) missing 'wochentag'.")
    if sitzung_missing_titel > 0:
        results["warnings"].append(f"Sitzung: {sitzung_missing_titel} session(s) missing 'titel'.")
    if sitzung_zero_wortzahl > 0:
        results["warnings"].append(f"Sitzung: {sitzung_zero_wortzahl} session(s) have 'gesamtwortzahl' <= 0.")
    if sitzung_without_reden > 0:
        results["warnings"].append(f"Sitzung: {sitzung_without_reden} session(s) have 0 speeches.")

    # 2. Redner Base Integrity
    redner_missing_vorname = (
        session.execute(
            select(func.count(Redner.redner_id)).where((Redner.vorname.is_(None)) | (Redner.vorname == ""))
        ).scalar() or 0
        if _col_exists("redner", "vorname") else 0
    )

    redner_missing_nachname = (
        session.execute(
            select(func.count(Redner.redner_id)).where((Redner.nachname.is_(None)) | (Redner.nachname == ""))
        ).scalar() or 0
        if _col_exists("redner", "nachname") else 0
    )

    redner_missing_fraktion = (
        session.execute(
            select(func.count(Redner.redner_id)).where((Redner.fraktion.is_(None)) | (Redner.fraktion == ""))
        ).scalar() or 0
        if _col_exists("redner", "fraktion") else 0
    )

    results["counts"]["redner"] = {
        "missing_vorname": redner_missing_vorname,
        "missing_nachname": redner_missing_nachname,
        "missing_fraktion": redner_missing_fraktion,
    }

    if redner_missing_vorname > 0:
        results["errors"].append(f"Redner: {redner_missing_vorname} speaker(s) missing 'vorname'.")
    if redner_missing_nachname > 0:
        results["errors"].append(f"Redner: {redner_missing_nachname} speaker(s) missing 'nachname'.")
    if redner_missing_fraktion > 0:
        results["warnings"].append(f"Redner: {redner_missing_fraktion} speaker(s) missing 'fraktion' (e.g. guests / independent).")

    # 3. Rede Base Integrity
    total_reden = session.execute(select(func.count(Rede.rede_id))).scalar() or 0

    rede_missing_text = (
        session.execute(
            select(func.count(Rede.rede_id)).where((Rede.text.is_(None)) | (Rede.text == ""))
        ).scalar() or 0
        if _col_exists("reden", "text") else 0
    )

    rede_missing_raw_text = (
        session.execute(
            select(func.count(Rede.rede_id)).where((Rede.raw_text.is_(None)) | (Rede.raw_text == ""))
        ).scalar() or 0
        if _col_exists("reden", "raw_text") else 0
    )

    rede_missing_clean_text = (
        session.execute(
            select(func.count(Rede.rede_id)).where((Rede.clean_text.is_(None)) | (Rede.clean_text == ""))
        ).scalar() or 0
        if _col_exists("reden", "clean_text") else 0
    )

    rede_missing_top = (
        session.execute(
            select(func.count(Rede.rede_id)).where(
                (Rede.tagesordnungspunkt.is_(None)) | (Rede.tagesordnungspunkt == "")
            )
        ).scalar() or 0
        if _col_exists("reden", "tagesordnungspunkt") else 0
    )

    rede_zero_wortanzahl = (
        session.execute(select(func.count(Rede.rede_id)).where(Rede.wortanzahl <= 0)).scalar() or 0
        if _col_exists("reden", "wortanzahl") else 0
    )

    results["counts"]["reden"] = {
        "total": total_reden,
        "missing_text": rede_missing_text,
        "missing_raw_text": rede_missing_raw_text,
        "missing_clean_text": rede_missing_clean_text,
        "missing_tagesordnungspunkt": rede_missing_top,
        "zero_wortanzahl": rede_zero_wortanzahl,
    }

    if rede_missing_text > 0:
        results["errors"].append(f"Rede: {rede_missing_text} speech(es) missing 'text'.")
    if _col_exists("reden", "clean_text") and rede_missing_clean_text > 0:
        results["warnings"].append(f"Rede: {rede_missing_clean_text} speech(es) missing 'clean_text'.")
    if rede_missing_top > 0:
        results["warnings"].append(f"Rede: {rede_missing_top} speech(es) missing 'tagesordnungspunkt'.")
    if rede_zero_wortanzahl > 0:
        results["warnings"].append(f"Rede: {rede_zero_wortanzahl} speech(es) have 'wortanzahl' <= 0.")

    # 4. Zwischenruf Base Integrity
    total_zwischenrufe = session.execute(select(func.count(Zwischenruf.ruf_id))).scalar() or 0

    zwischenruf_missing_text = (
        session.execute(
            select(func.count(Zwischenruf.ruf_id)).where((Zwischenruf.text.is_(None)) | (Zwischenruf.text == ""))
        ).scalar() or 0
        if _col_exists("zwischenrufe", "text") else 0
    )

    zwischenruf_missing_fraktion = (
        session.execute(
            select(func.count(Zwischenruf.ruf_id)).where(
                (Zwischenruf.fraktion.is_(None)) | (Zwischenruf.fraktion == "")
            )
        ).scalar() or 0
        if _col_exists("zwischenrufe", "fraktion") else 0
    )

    results["counts"]["zwischenrufe"] = {
        "total": total_zwischenrufe,
        "missing_text": zwischenruf_missing_text,
        "missing_fraktion": zwischenruf_missing_fraktion,
    }

    if zwischenruf_missing_text > 0:
        results["errors"].append(f"Zwischenruf: {zwischenruf_missing_text} interjection(s) missing 'text'.")
    if zwischenruf_missing_fraktion > 0:
        results["warnings"].append(f"Zwischenruf: {zwischenruf_missing_fraktion} interjection(s) missing 'fraktion'.")

    if results["errors"]:
        results["status"] = "ERROR"
    elif results["warnings"]:
        results["status"] = "WARNING"

    return results


def check_unfilled_precomputation_cells(session: Session, valid_cols: Set[str]) -> Dict[str, Any]:
    """Check for unfilled post-processing/precomputation cells (flagged as WARNINGS for pending pipeline steps)."""
    results: Dict[str, Any] = {
        "status": "OK",
        "warnings": [],
        "counts": {},
    }

    def _col_exists(table: str, col: str) -> bool:
        return f"{table}.{col}" in valid_cols

    total_reden = session.execute(select(func.count(Rede.rede_id))).scalar() or 0
    total_zwischenrufe = session.execute(select(func.count(Zwischenruf.ruf_id))).scalar() or 0

    # 1. NLP Fields on Rede
    rede_missing_sentiment = (
        session.execute(
            select(func.count(Rede.rede_id)).where(Rede.sentiment_score.is_(None))
        ).scalar() or 0
        if _col_exists("reden", "sentiment_score") else total_reden
    )

    rede_missing_ton = (
        session.execute(
            select(func.count(Rede.rede_id)).where(Rede.ton_label.is_(None))
        ).scalar() or 0
        if _col_exists("reden", "ton_label") else total_reden
    )

    rede_missing_tone_aggression = (
        session.execute(
            select(func.count(Rede.rede_id)).where(Rede.tone_aggression.is_(None))
        ).scalar() or 0
        if _col_exists("reden", "tone_aggression") else total_reden
    )

    rede_missing_adressaten = (
        session.execute(
            select(func.count(Rede.rede_id)).where(Rede.adressaten.is_(None))
        ).scalar() or 0
        if _col_exists("reden", "adressaten") else total_reden
    )

    results["counts"]["reden_nlp"] = {
        "missing_sentiment": rede_missing_sentiment,
        "missing_ton_label": rede_missing_ton,
        "missing_tone_aggression": rede_missing_tone_aggression,
        "missing_adressaten": rede_missing_adressaten,
    }

    if rede_missing_sentiment > 0:
        results["warnings"].append(
            f"Precomputation Pending: {rede_missing_sentiment}/{total_reden} speech(es) missing 'sentiment_score'."
        )
    if rede_missing_ton > 0:
        results["warnings"].append(
            f"Precomputation Pending: {rede_missing_ton}/{total_reden} speech(es) missing 'ton_label'."
        )
    if rede_missing_tone_aggression > 0:
        results["warnings"].append(
            f"Precomputation Pending: {rede_missing_tone_aggression}/{total_reden} speech(es) missing 'tone_aggression'."
        )

    # 2. NLP Fields on Zwischenruf
    zr_missing_sentiment = (
        session.execute(
            select(func.count(Zwischenruf.ruf_id)).where(Zwischenruf.sentiment_score.is_(None))
        ).scalar() or 0
        if _col_exists("zwischenrufe", "sentiment_score") else total_zwischenrufe
    )

    zr_missing_kategorie = (
        session.execute(
            select(func.count(Zwischenruf.ruf_id)).where(Zwischenruf.kategorie.is_(None))
        ).scalar() or 0
        if _col_exists("zwischenrufe", "kategorie") else total_zwischenrufe
    )

    zr_missing_ton = (
        session.execute(
            select(func.count(Zwischenruf.ruf_id)).where(Zwischenruf.ton_label.is_(None))
        ).scalar() or 0
        if _col_exists("zwischenrufe", "ton_label") else total_zwischenrufe
    )

    results["counts"]["zwischenrufe_nlp"] = {
        "missing_sentiment": zr_missing_sentiment,
        "missing_kategorie": zr_missing_kategorie,
        "missing_ton_label": zr_missing_ton,
    }

    if zr_missing_sentiment > 0:
        results["warnings"].append(
            f"Precomputation Pending: {zr_missing_sentiment}/{total_zwischenrufe} interjection(s) missing 'sentiment_score'."
        )
    if zr_missing_kategorie > 0:
        results["warnings"].append(
            f"Precomputation Pending: {zr_missing_kategorie}/{total_zwischenrufe} interjection(s) missing 'kategorie'."
        )

    # 3. Stylometry & Speech Complexity precomputed tables
    total_stylometry = (
        session.execute(select(func.count(StylometrieFeatures.id))).scalar() or 0
        if "stylometrie_features" in {t.split(".")[0] for t in valid_cols} else 0
    )
    total_complexity = (
        session.execute(select(func.count(SpeechComplexityResult.id))).scalar() or 0
        if "speech_complexity_results" in {t.split(".")[0] for t in valid_cols} else 0
    )
    total_reaction_gap = (
        session.execute(select(func.count(ReactionGapResult.id))).scalar() or 0
        if "reaction_gap_results" in {t.split(".")[0] for t in valid_cols} else 0
    )

    missing_stylometry = total_reden - total_stylometry
    missing_complexity = total_reden - total_complexity

    results["counts"]["precomputed_tables"] = {
        "stylometrie_features": total_stylometry,
        "speech_complexity_results": total_complexity,
        "reaction_gap_results": total_reaction_gap,
        "missing_stylometry": missing_stylometry,
        "missing_complexity": missing_complexity,
    }

    if missing_stylometry > 0:
        results["warnings"].append(
            f"Precomputation Pending: {missing_stylometry}/{total_reden} speech(es) missing 'StylometrieFeatures'."
        )
    if missing_complexity > 0:
        results["warnings"].append(
            f"Precomputation Pending: {missing_complexity}/{total_reden} speech(es) missing 'SpeechComplexityResult'."
        )

    # 4. Framing & Lexical precomputation tables
    total_framing = (
        session.execute(select(func.count(FramingResult.id))).scalar() or 0
        if "framing_results" in {t.split(".")[0] for t in valid_cols} else 0
    )
    total_lexical = (
        session.execute(select(func.count(LexicalResult.id))).scalar() or 0
        if "lexical_results" in {t.split(".")[0] for t in valid_cols} else 0
    )

    results["counts"]["precomputed_tables"]["framing_results"] = total_framing
    results["counts"]["precomputed_tables"]["lexical_results"] = total_lexical

    if total_framing == 0:
        results["warnings"].append("Precomputation Pending: 'framing_results' table is completely empty.")
    if total_lexical == 0:
        results["warnings"].append("Precomputation Pending: 'lexical_results' table is completely empty.")

    # 5. Parquet precomputation files (Hypothesis 3 asymmetric attacks)
    h3_file = _PROJECT_ROOT / "data" / "h3_precomputed.parquet"
    if not h3_file.exists():
        results["warnings"].append("Precomputation Pending: 'data/h3_precomputed.parquet' file does not exist.")
    else:
        results["counts"]["h3_parquet_size_bytes"] = h3_file.stat().st_size

    if results["warnings"]:
        results["status"] = "WARNING"

    return results


def run_db_check(
    db_url: Optional[str] = None,
    verbose: bool = False,
    strict: bool = False,
) -> Dict[str, Any]:
    """Execute all database completeness and sequence integrity checks."""
    if db_url:
        engine = create_engine(db_url)
        factory = sessionmaker(bind=engine)
        session = factory()
    else:
        # First ensure schema patches and indexes are initialized
        try:
            init_db()
        except Exception as exc:
            logger.warning("init_db failed during check setup: %s", exc)
        session = get_session().__enter__()

    try:
        schema_results, valid_cols = check_schema_integrity(session)
        seq_results = check_sitzungen_sequence(session)
        base_results = check_unfilled_base_cells(session, valid_cols)
        precomp_results = check_unfilled_precomputation_cells(session, valid_cols)

        all_errors = (
            schema_results.get("errors", [])
            + seq_results.get("errors", [])
            + base_results.get("errors", [])
        )
        all_warnings = (
            schema_results.get("warnings", [])
            + seq_results.get("warnings", [])
            + base_results.get("warnings", [])
            + precomp_results.get("warnings", [])
        )

        overall_status = "OK"
        if all_errors:
            overall_status = "ERROR"
        elif all_warnings:
            overall_status = "WARNING"

        report: Dict[str, Any] = {
            "overall_status": overall_status,
            "error_count": len(all_errors),
            "warning_count": len(all_warnings),
            "errors": all_errors,
            "warnings": all_warnings,
            "schema_integrity": schema_results,
            "sitzungen_sequence": seq_results,
            "base_integrity": base_results,
            "precomputation_integrity": precomp_results,
        }
        return report
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check database completeness, unfilled cells, and Sitzung sequence continuity."
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display detailed diagnostic output.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors and exit with non-zero status if warnings are found.",
    )
    parser.add_argument("-o", "--output", type=str, help="Save report to specified JSON file path.")
    parser.add_argument("--db-url", type=str, help="Custom database URL (overrides default).")

    args = parser.parse_args()

    report = run_db_check(
        db_url=args.db_url,
        verbose=args.verbose,
        strict=args.strict,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Saved report to %s", out_path)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("\n=======================================================")
        print(f" DATABASE INTEGRITY REPORT - Status: {report['overall_status']}")
        print("=======================================================")
        print(f"Total Errors:   {report['error_count']}")
        print(f"Total Warnings: {report['warning_count']}\n")

        if report["errors"]:
            print("-------------------------------------------------------")
            print(" ERRORS (Immediate Action Required)")
            print("-------------------------------------------------------")
            for err in report["errors"]:
                print(f" [ERROR] {err}")
            print()

        if report["warnings"]:
            print("-------------------------------------------------------")
            print(" WARNINGS (Precomputation Pending or Non-Critical)")
            print("-------------------------------------------------------")
            for warn in report["warnings"]:
                print(f" [WARN] {warn}")
            print()

        if args.verbose:
            print("-------------------------------------------------------")
            print(" DETAILED COUNTS")
            print("-------------------------------------------------------")
            print(json.dumps(report["base_integrity"]["counts"], indent=2))
            print(json.dumps(report["precomputation_integrity"]["counts"], indent=2))

    # Exit code determination
    if report["error_count"] > 0 or (args.strict and report["warning_count"] > 0):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
