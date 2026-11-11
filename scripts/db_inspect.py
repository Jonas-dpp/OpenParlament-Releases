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
Enhanced DB inspection and completeness helper for OpenParlament.

Run: python scripts/db_inspect.py [--db-url URL] [--json] [--verbose] [--table TABLE]
It inspects database tables, row counts, schema structure, column fill rates,
missing entries (unfilled spots / nulls), foreign key orphans, session sequence
gaps, and precomputation/FTS5 coverage.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# make sure running from repo root works
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from src.database import get_session
from src.models import (
    FramingResult,
    LexicalResult,
    ReactionGapResult,
    Rede,
    Redner,
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
logger = logging.getLogger("db_inspect")

ALL_ORM_MODELS = (
    Sitzung,
    Redner,
    Rede,
    Zwischenruf,
    StylometrieFeatures,
    FramingResult,
    LexicalResult,
    SpeechComplexityResult,
    ReactionGapResult,
)


def _safe_percent(part: int, total: int) -> float:
    """Calculate percentage safely avoiding division by zero."""
    if total <= 0:
        return 0.0
    return round((part / total) * 100.0, 2)


def check_table_completeness(
    session: Session,
    table_name: str,
    total_rows: int,
    existing_tables: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Inspect missing/null spots and data anomalies for a specific table."""
    res: Dict[str, Any] = {
        "table_name": table_name,
        "row_count": total_rows,
        "is_empty": total_rows == 0,
        "missing_spots": {},
        "missing_spots_count": 0,
        "fill_status": "EMPTY" if total_rows == 0 else "FULLY_FILLED",
    }

    if total_rows == 0:
        return res

    missing: Dict[str, Dict[str, Any]] = {}

    def _has_col(tbl: str, col: str) -> bool:
        if existing_tables is None:
            return True
        cols = existing_tables.get(tbl)
        return cols is not None and col in cols

    def _has_tbl(tbl: str) -> bool:
        if existing_tables is None:
            return True
        return tbl in existing_tables

    if table_name == "sitzungen":
        checks = [
            ("missing_datum", "datum", Sitzung.datum.is_(None)),
            ("missing_wochentag", "wochentag", (Sitzung.wochentag.is_(None)) | (Sitzung.wochentag == "")),
            ("missing_titel", "titel", (Sitzung.titel.is_(None)) | (Sitzung.titel == "")),
            ("zero_gesamtwortzahl", "gesamtwortzahl", Sitzung.gesamtwortzahl <= 0),
        ]
        for spot_key, col_name, condition in checks:
            if not _has_col("sitzungen", col_name):
                continue
            try:
                cnt = session.execute(select(func.count(Sitzung.sitzungs_id)).where(condition)).scalar() or 0
                if cnt > 0:
                    missing[spot_key] = {"count": cnt, "percent": _safe_percent(cnt, total_rows)}
            except Exception:
                pass

        # Relational check: Sitzungen without speeches
        if _has_tbl("reden") and _has_col("reden", "sitzung_id"):
            try:
                sessions_without_speeches = session.execute(
                    select(func.count(Sitzung.sitzungs_id)).where(
                        ~Sitzung.sitzungs_id.in_(select(Rede.sitzung_id).distinct())
                    )
                ).scalar() or 0
                if sessions_without_speeches > 0:
                    missing["sessions_without_speeches"] = {
                        "count": sessions_without_speeches,
                        "percent": _safe_percent(sessions_without_speeches, total_rows),
                    }
            except Exception:
                pass

    elif table_name == "redner":
        checks = [
            ("missing_vorname", "vorname", (Redner.vorname.is_(None)) | (Redner.vorname == "")),
            ("missing_nachname", "nachname", (Redner.nachname.is_(None)) | (Redner.nachname == "")),
            ("missing_fraktion", "fraktion", (Redner.fraktion.is_(None)) | (Redner.fraktion == "")),
            ("missing_bundestag_id", "bundestag_id", (Redner.bundestag_id.is_(None)) | (Redner.bundestag_id == "")),
        ]
        for spot_key, col_name, condition in checks:
            if not _has_col("redner", col_name):
                continue
            try:
                cnt = session.execute(select(func.count(Redner.redner_id)).where(condition)).scalar() or 0
                if cnt > 0:
                    missing[spot_key] = {"count": cnt, "percent": _safe_percent(cnt, total_rows)}
            except Exception:
                pass

        # Relational check: Speakers without any speeches
        if _has_tbl("reden") and _has_col("reden", "redner_id"):
            try:
                speakers_without_speeches = session.execute(
                    select(func.count(Redner.redner_id)).where(
                        ~Redner.redner_id.in_(select(Rede.redner_id).distinct())
                    )
                ).scalar() or 0
                if speakers_without_speeches > 0:
                    missing["speakers_without_speeches"] = {
                        "count": speakers_without_speeches,
                        "percent": _safe_percent(speakers_without_speeches, total_rows),
                    }
            except Exception:
                pass

    elif table_name == "reden":
        checks = [
            ("missing_text", "text", (Rede.text.is_(None)) | (Rede.text == "")),
            ("missing_raw_text", "raw_text", (Rede.raw_text.is_(None)) | (Rede.raw_text == "")),
            ("missing_clean_text", "clean_text", (Rede.clean_text.is_(None)) | (Rede.clean_text == "")),
            ("missing_normalized_text", "normalized_text", (Rede.normalized_text.is_(None)) | (Rede.normalized_text == "")),
            ("missing_metadata_text", "metadata_text", (Rede.metadata_text.is_(None)) | (Rede.metadata_text == "")),
            ("missing_tagesordnungspunkt", "tagesordnungspunkt", (Rede.tagesordnungspunkt.is_(None)) | (Rede.tagesordnungspunkt == "")),
            ("zero_wortanzahl", "wortanzahl", Rede.wortanzahl <= 0),
            # NLP & sentiment fields
            ("missing_sentiment_score", "sentiment_score", Rede.sentiment_score.is_(None)),
            ("missing_ton_label", "ton_label", Rede.ton_label.is_(None)),
            ("missing_tone_aggression", "tone_aggression", Rede.tone_aggression.is_(None)),
            ("missing_tone_sarkasmus", "tone_sarkasmus", Rede.tone_sarkasmus.is_(None)),
            ("missing_tone_humor", "tone_humor", Rede.tone_humor.is_(None)),
            ("missing_tone_neutral", "tone_neutral", Rede.tone_neutral.is_(None)),
            ("missing_tone_scores", "tone_scores", Rede.tone_scores.is_(None)),
            ("missing_adressaten", "adressaten", Rede.adressaten.is_(None)),
        ]
        for spot_key, col_name, condition in checks:
            if not _has_col("reden", col_name):
                continue
            try:
                cnt = session.execute(select(func.count(Rede.rede_id)).where(condition)).scalar() or 0
                if cnt > 0:
                    missing[spot_key] = {"count": cnt, "percent": _safe_percent(cnt, total_rows)}
            except Exception:
                pass

        # Foreign key orphan checks
        if _has_tbl("sitzungen") and _has_col("reden", "sitzung_id"):
            try:
                orphaned_sitzung = session.execute(
                    select(func.count(Rede.rede_id)).where(
                        ~Rede.sitzung_id.in_(select(Sitzung.sitzungs_id).distinct())
                    )
                ).scalar() or 0
                if orphaned_sitzung > 0:
                    missing["orphaned_invalid_sitzung_id"] = {
                        "count": orphaned_sitzung,
                        "percent": _safe_percent(orphaned_sitzung, total_rows),
                    }
            except Exception:
                pass

        if _has_tbl("redner") and _has_col("reden", "redner_id"):
            try:
                orphaned_redner = session.execute(
                    select(func.count(Rede.rede_id)).where(
                        ~Rede.redner_id.in_(select(Redner.redner_id).distinct())
                    )
                ).scalar() or 0
                if orphaned_redner > 0:
                    missing["orphaned_invalid_redner_id"] = {
                        "count": orphaned_redner,
                        "percent": _safe_percent(orphaned_redner, total_rows),
                    }
            except Exception:
                pass

        # Precomputation coverage for speeches
        if _has_tbl("stylometrie_features") and _has_col("stylometrie_features", "rede_id"):
            try:
                speeches_missing_stylometry = session.execute(
                    select(func.count(Rede.rede_id)).where(
                        ~Rede.rede_id.in_(select(StylometrieFeatures.rede_id).distinct())
                    )
                ).scalar() or 0
                if speeches_missing_stylometry > 0:
                    missing["speeches_missing_stylometry"] = {
                        "count": speeches_missing_stylometry,
                        "percent": _safe_percent(speeches_missing_stylometry, total_rows),
                    }
            except Exception:
                pass

        if _has_tbl("speech_complexity_results") and _has_col("speech_complexity_results", "rede_id"):
            try:
                speeches_missing_complexity = session.execute(
                    select(func.count(Rede.rede_id)).where(
                        ~Rede.rede_id.in_(select(SpeechComplexityResult.rede_id).distinct())
                    )
                ).scalar() or 0
                if speeches_missing_complexity > 0:
                    missing["speeches_missing_complexity"] = {
                        "count": speeches_missing_complexity,
                        "percent": _safe_percent(speeches_missing_complexity, total_rows),
                    }
            except Exception:
                pass

    elif table_name == "zwischenrufe":
        checks = [
            ("missing_text", "text", (Zwischenruf.text.is_(None)) | (Zwischenruf.text == "")),
            ("missing_fraktion", "fraktion", (Zwischenruf.fraktion.is_(None)) | (Zwischenruf.fraktion == "")),
            ("missing_sentiment_score", "sentiment_score", Zwischenruf.sentiment_score.is_(None)),
            ("missing_kategorie", "kategorie", Zwischenruf.kategorie.is_(None)),
            ("missing_ton_label", "ton_label", Zwischenruf.ton_label.is_(None)),
            ("missing_tone_aggression", "tone_aggression", Zwischenruf.tone_aggression.is_(None)),
            ("missing_tone_sarkasmus", "tone_sarkasmus", Zwischenruf.tone_sarkasmus.is_(None)),
            ("missing_tone_humor", "tone_humor", Zwischenruf.tone_humor.is_(None)),
            ("missing_tone_neutral", "tone_neutral", Zwischenruf.tone_neutral.is_(None)),
            ("missing_tone_scores", "tone_scores", Zwischenruf.tone_scores.is_(None)),
            ("missing_adressaten", "adressaten", Zwischenruf.adressaten.is_(None)),
        ]
        for spot_key, col_name, condition in checks:
            if not _has_col("zwischenrufe", col_name):
                continue
            try:
                cnt = session.execute(select(func.count(Zwischenruf.ruf_id)).where(condition)).scalar() or 0
                if cnt > 0:
                    missing[spot_key] = {"count": cnt, "percent": _safe_percent(cnt, total_rows)}
            except Exception:
                pass

        # Foreign key orphan check: Zwischenruf referencing non-existent Rede
        if _has_tbl("reden") and _has_col("zwischenrufe", "rede_id"):
            try:
                orphaned_rede = session.execute(
                    select(func.count(Zwischenruf.ruf_id)).where(
                        ~Zwischenruf.rede_id.in_(select(Rede.rede_id).distinct())
                    )
                ).scalar() or 0
                if orphaned_rede > 0:
                    missing["orphaned_invalid_rede_id"] = {
                        "count": orphaned_rede,
                        "percent": _safe_percent(orphaned_rede, total_rows),
                    }
            except Exception:
                pass

    elif table_name == "stylometrie_features":
        checks = [
            ("missing_avg_sentence_length", "avg_sentence_length", StylometrieFeatures.avg_sentence_length.is_(None)),
            ("missing_passive_ratio", "passive_ratio", StylometrieFeatures.passive_ratio.is_(None)),
            ("missing_type_token_ratio", "type_token_ratio", StylometrieFeatures.type_token_ratio.is_(None)),
            ("missing_fill_word_ratio", "fill_word_ratio", StylometrieFeatures.fill_word_ratio.is_(None)),
            ("missing_punctuation_freq", "punctuation_freq", StylometrieFeatures.punctuation_freq.is_(None)),
        ]
        for spot_key, col_name, condition in checks:
            if not _has_col("stylometrie_features", col_name):
                continue
            try:
                cnt = session.execute(select(func.count(StylometrieFeatures.id)).where(condition)).scalar() or 0
                if cnt > 0:
                    missing[spot_key] = {"count": cnt, "percent": _safe_percent(cnt, total_rows)}
            except Exception:
                pass

        if _has_tbl("reden") and _has_col("stylometrie_features", "rede_id"):
            try:
                orphaned = session.execute(
                    select(func.count(StylometrieFeatures.id)).where(
                        ~StylometrieFeatures.rede_id.in_(select(Rede.rede_id).distinct())
                    )
                ).scalar() or 0
                if orphaned > 0:
                    missing["orphaned_invalid_rede_id"] = {
                        "count": orphaned,
                        "percent": _safe_percent(orphaned, total_rows),
                    }
            except Exception:
                pass

    elif table_name == "speech_complexity_results":
        checks = [
            ("missing_wstf_score", "wstf_score", SpeechComplexityResult.wstf_score.is_(None)),
            ("missing_flesch_score", "flesch_score", SpeechComplexityResult.flesch_score.is_(None)),
            ("missing_avg_sentence_length", "avg_sentence_length", SpeechComplexityResult.avg_sentence_length.is_(None)),
            ("missing_subordinate_clause_ratio", "subordinate_clause_ratio", SpeechComplexityResult.subordinate_clause_ratio.is_(None)),
            ("missing_avg_word_length", "avg_word_length", SpeechComplexityResult.avg_word_length.is_(None)),
            ("missing_long_word_ratio", "long_word_ratio", SpeechComplexityResult.long_word_ratio.is_(None)),
            ("missing_lexical_diversity", "lexical_diversity", SpeechComplexityResult.lexical_diversity.is_(None)),
        ]
        for spot_key, col_name, condition in checks:
            if not _has_col("speech_complexity_results", col_name):
                continue
            try:
                cnt = session.execute(select(func.count(SpeechComplexityResult.id)).where(condition)).scalar() or 0
                if cnt > 0:
                    missing[spot_key] = {"count": cnt, "percent": _safe_percent(cnt, total_rows)}
            except Exception:
                pass

        if _has_tbl("reden") and _has_col("speech_complexity_results", "rede_id"):
            try:
                orphaned_rede = session.execute(
                    select(func.count(SpeechComplexityResult.id)).where(
                        ~SpeechComplexityResult.rede_id.in_(select(Rede.rede_id).distinct())
                    )
                ).scalar() or 0
                if orphaned_rede > 0:
                    missing["orphaned_invalid_rede_id"] = {
                        "count": orphaned_rede,
                        "percent": _safe_percent(orphaned_rede, total_rows),
                    }
            except Exception:
                pass

        if _has_tbl("redner") and _has_col("speech_complexity_results", "redner_id"):
            try:
                orphaned_redner = session.execute(
                    select(func.count(SpeechComplexityResult.id)).where(
                        ~SpeechComplexityResult.redner_id.in_(select(Redner.redner_id).distinct())
                    )
                ).scalar() or 0
                if orphaned_redner > 0:
                    missing["orphaned_invalid_redner_id"] = {
                        "count": orphaned_redner,
                        "percent": _safe_percent(orphaned_redner, total_rows),
                    }
            except Exception:
                pass

    elif table_name == "reaction_gap_results":
        checks = [
            ("missing_topic_id", "topic_id", (ReactionGapResult.topic_id.is_(None)) | (ReactionGapResult.topic_id == "")),
            ("missing_speaker_sentiment", "speaker_sentiment", ReactionGapResult.speaker_sentiment.is_(None)),
            ("missing_reaction_sentiment", "reaction_sentiment", ReactionGapResult.reaction_sentiment.is_(None)),
            ("missing_gap", "gap", ReactionGapResult.gap.is_(None)),
            ("missing_time_to_first_interjection", "time_to_first_interjection", ReactionGapResult.time_to_first_interjection.is_(None)),
        ]
        for spot_key, col_name, condition in checks:
            if not _has_col("reaction_gap_results", col_name):
                continue
            try:
                cnt = session.execute(select(func.count(ReactionGapResult.id)).where(condition)).scalar() or 0
                if cnt > 0:
                    missing[spot_key] = {"count": cnt, "percent": _safe_percent(cnt, total_rows)}
            except Exception:
                pass

        if _has_tbl("reden") and _has_col("reaction_gap_results", "rede_id"):
            try:
                orphaned_rede = session.execute(
                    select(func.count(ReactionGapResult.id)).where(
                        ~ReactionGapResult.rede_id.in_(select(Rede.rede_id).distinct())
                    )
                ).scalar() or 0
                if orphaned_rede > 0:
                    missing["orphaned_invalid_rede_id"] = {
                        "count": orphaned_rede,
                        "percent": _safe_percent(orphaned_rede, total_rows),
                    }
            except Exception:
                pass

        if _has_tbl("zwischenrufe") and _has_col("reaction_gap_results", "zwischenruf_id"):
            try:
                orphaned_zr = session.execute(
                    select(func.count(ReactionGapResult.id)).where(
                        ~ReactionGapResult.zwischenruf_id.in_(select(Zwischenruf.ruf_id).distinct())
                    )
                ).scalar() or 0
                if orphaned_zr > 0:
                    missing["orphaned_invalid_zwischenruf_id"] = {
                        "count": orphaned_zr,
                        "percent": _safe_percent(orphaned_zr, total_rows),
                    }
            except Exception:
                pass

    elif table_name == "framing_results":
        checks = [
            ("missing_fraktion", "fraktion", (FramingResult.fraktion.is_(None)) | (FramingResult.fraktion == "")),
            ("missing_term", "term", (FramingResult.term.is_(None)) | (FramingResult.term == "")),
            ("missing_score", "score", FramingResult.score.is_(None)),
            ("missing_segment", "segment", (FramingResult.segment.is_(None)) | (FramingResult.segment == "")),
            ("missing_variant", "variant", (FramingResult.variant.is_(None)) | (FramingResult.variant == "")),
        ]
        for spot_key, col_name, condition in checks:
            if not _has_col("framing_results", col_name):
                continue
            try:
                cnt = session.execute(select(func.count(FramingResult.id)).where(condition)).scalar() or 0
                if cnt > 0:
                    missing[spot_key] = {"count": cnt, "percent": _safe_percent(cnt, total_rows)}
            except Exception:
                pass

    elif table_name == "lexical_results":
        checks = [
            ("missing_fraktion", "fraktion", (LexicalResult.fraktion.is_(None)) | (LexicalResult.fraktion == "")),
            ("missing_ttr", "ttr", LexicalResult.ttr.is_(None)),
            ("missing_lexical_density", "lexical_density", LexicalResult.lexical_density.is_(None)),
            ("missing_msttr", "msttr", LexicalResult.msttr.is_(None)),
            ("zero_total_tokens", "total_tokens", LexicalResult.total_tokens <= 0),
            ("zero_total_types", "total_types", LexicalResult.total_types <= 0),
        ]
        for spot_key, col_name, condition in checks:
            if not _has_col("lexical_results", col_name):
                continue
            try:
                cnt = session.execute(select(func.count(LexicalResult.id)).where(condition)).scalar() or 0
                if cnt > 0:
                    missing[spot_key] = {"count": cnt, "percent": _safe_percent(cnt, total_rows)}
            except Exception:
                pass

    res["missing_spots"] = missing
    res["missing_spots_count"] = len(missing)
    res["fill_status"] = "MISSING_ENTRIES" if missing else "FULLY_FILLED"
    return res


def inspect_session_sequence_spots(session: Session) -> Dict[str, Any]:
    """Inspect election periods and detect missing session sequence spots (gaps), including 1..min_nr-1."""
    res: Dict[str, Any] = {
        "wahlperioden": {},
        "has_gaps": False,
        "total_missing_sessions": 0,
    }

    try:
        sitzungen = session.execute(
            select(Sitzung.wahlperiode, Sitzung.sitzungsnr).order_by(Sitzung.wahlperiode, Sitzung.sitzungsnr)
        ).all()
    except Exception:
        return res

    if not sitzungen:
        return res

    by_wp: Dict[int, List[int]] = {}
    for wp, snr in sitzungen:
        by_wp.setdefault(wp, []).append(snr)

    total_missing = 0
    sorted_wps = sorted(by_wp.items())
    total_wps = len(sorted_wps)
    start_time = time.time()
    log_step = max(1, total_wps // 10)

    for idx, (wp, numbers) in enumerate(sorted_wps, 1):
        min_nr = min(numbers)
        max_nr = max(numbers)
        actual_count = len(numbers)
        # Session numbering must be continuous [1 .. max_nr]
        expected_set = set(range(1, max_nr + 1))
        missing_numbers = sorted(list(expected_set - set(numbers)))

        wp_info = {
            "session_count": actual_count,
            "min_sitzungsnr": min_nr,
            "max_sitzungsnr": max_nr,
            "missing_session_spots": missing_numbers,
            "missing_count": len(missing_numbers),
            "status": "GAP_DETECTED" if missing_numbers else "CONTINUOUS",
        }
        if missing_numbers:
            res["has_gaps"] = True
            total_missing += len(missing_numbers)
        res["wahlperioden"][wp] = wp_info

        # Telemetry: Progress and ETA calculation per checked Wahlperiode
        if total_wps > 0 and (idx == 1 or idx % log_step == 0 or idx == total_wps):
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            eta = avg_time * (total_wps - idx)
            eta_str = time.strftime("%M:%S", time.gmtime(eta)) if eta < 3600 else time.strftime("%H:%M:%S", time.gmtime(eta))
            logger.info(
                f"SessionSequenceSpotsCheck: {idx}/{total_wps} | {int((idx/total_wps)*100)}% | ETA: {eta_str}"
            )

    res["total_missing_sessions"] = total_missing
    return res


def inspect_fts5_status(session: Session, total_reden: int) -> Dict[str, Any]:
    """Inspect FTS5 virtual table status for full text search (row count, rowid coverage, content sync, integrity)."""
    res: Dict[str, Any] = {
        "exists": False,
        "row_count": 0,
        "synced": False,
        "missing_speech_spots": 0,
        "unmatched_rowids_count": 0,
        "content_mismatch_count": 0,
        "integrity_ok": False,
        "status": "NOT_CONFIGURED",
    }
    try:
        bind = session.bind
        if not bind or "sqlite" not in str(bind.url):
            res["status"] = "NON_SQLITE"
            return res

        fts_check = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='reden_fts'")
        ).scalar()

        if fts_check:
            res["exists"] = True
            cnt = session.execute(text("SELECT count(*) FROM reden_fts")).scalar() or 0
            res["row_count"] = cnt

            # Check FTS5 internal integrity command
            try:
                session.execute(text("INSERT INTO reden_fts(reden_fts) VALUES('integrity-check')"))
                res["integrity_ok"] = True
            except Exception:
                res["integrity_ok"] = False

            # Check rowid coverage: speeches missing in reden_fts
            unmatched_speeches = session.execute(
                text(
                    "SELECT count(*) FROM reden "
                    "WHERE rede_id NOT IN (SELECT rowid FROM reden_fts)"
                )
            ).scalar() or 0
            res["missing_speech_spots"] = unmatched_speeches

            # Check orphaned rowids in reden_fts not present in reden
            orphaned_fts_rowids = session.execute(
                text(
                    "SELECT count(*) FROM reden_fts "
                    "WHERE rowid NOT IN (SELECT rede_id FROM reden)"
                )
            ).scalar() or 0
            res["unmatched_rowids_count"] = unmatched_speeches + orphaned_fts_rowids

            # Check content synchronization: text content mismatch between reden and reden_fts
            content_mismatches = session.execute(
                text(
                    "SELECT count(*) FROM reden r "
                    "JOIN reden_fts f ON r.rede_id = f.rowid "
                    "WHERE COALESCE(r.text, '') != COALESCE(f.text, '')"
                )
            ).scalar() or 0
            res["content_mismatch_count"] = content_mismatches

            is_perfect_sync = (
                res["integrity_ok"]
                and cnt == total_reden
                and unmatched_speeches == 0
                and orphaned_fts_rowids == 0
                and content_mismatches == 0
            )
            res["synced"] = is_perfect_sync

            if total_reden == 0 and cnt == 0:
                res["status"] = "SYNCED_EMPTY"
            elif is_perfect_sync:
                res["status"] = "FULLY_SYNCED"
            else:
                res["status"] = "DESYNCHRONIZED"
    except Exception as exc:
        res["error"] = str(exc)
        res["status"] = "ERROR"

    return res


def inspect_database(session: Session, filter_table: Optional[str] = None) -> Dict[str, Any]:
    """Perform a comprehensive database completeness and missing entries audit."""
    inspector = inspect(session.bind)
    existing_tables_map: Dict[str, List[str]] = {}
    for t_name in inspector.get_table_names():
        try:
            existing_tables_map[t_name] = [col["name"] for col in inspector.get_columns(t_name)]
        except Exception:
            existing_tables_map[t_name] = []

    report: Dict[str, Any] = {
        "engine": str(getattr(session, "bind", "")),
        "tables": {},
        "summary": {
            "total_tables_checked": 0,
            "fully_filled_tables": 0,
            "partially_filled_tables": 0,
            "empty_tables": 0,
            "missing_tables": 0,
            "overall_fill_verdict": "EMPTY",
        },
    }

    total_non_empty = 0
    total_fully_filled = 0
    total_partially_filled = 0
    total_empty = 0
    total_missing = 0

    reden_count = 0

    target_models = [
        cls for cls in ALL_ORM_MODELS
        if not filter_table or (cls.__tablename__ == filter_table or cls.__name__ == filter_table)
    ]
    total_models = len(target_models)
    start_time = time.time()

    for idx, cls in enumerate(target_models, 1):
        tbl = cls.__tablename__

        report["summary"]["total_tables_checked"] += 1

        if tbl not in existing_tables_map:
            total_missing += 1
            report["tables"][tbl] = {
                "model_name": cls.__name__,
                "table_name": tbl,
                "exists": False,
                "row_count": 0,
                "fill_status": "MISSING_TABLE",
                "missing_spots": {},
                "missing_spots_count": 0,
            }
            continue

        try:
            cnt = session.execute(select(func.count()).select_from(cls)).scalar() or 0
        except Exception as exc:
            report["tables"][tbl] = {
                "model_name": cls.__name__,
                "table_name": tbl,
                "exists": True,
                "row_count": -1,
                "error": str(exc),
                "fill_status": "QUERY_ERROR",
                "missing_spots": {},
                "missing_spots_count": 0,
            }
            continue

        if tbl == "reden":
            reden_count = cnt

        t_res = check_table_completeness(session, tbl, cnt, existing_tables=existing_tables_map)
        t_res["model_name"] = cls.__name__
        t_res["exists"] = True
        report["tables"][tbl] = t_res

        if cnt == 0:
            total_empty += 1
        else:
            total_non_empty += 1
            if t_res["fill_status"] == "FULLY_FILLED":
                total_fully_filled += 1
            else:
                total_partially_filled += 1

        # Telemetry: Progress and ETA calculation per checked model
        if total_models > 0 and (idx == 1 or idx % 3 == 0 or idx == total_models):
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            eta = avg_time * (total_models - idx)
            eta_str = time.strftime("%M:%S", time.gmtime(eta)) if eta < 3600 else time.strftime("%H:%M:%S", time.gmtime(eta))
            logger.info(
                f"TableCompletenessCheck: {idx}/{total_models} | {int((idx/total_models)*100)}% | ETA: {eta_str}"
            )

    report["summary"]["fully_filled_tables"] = total_fully_filled
    report["summary"]["partially_filled_tables"] = total_partially_filled
    report["summary"]["empty_tables"] = total_empty
    report["summary"]["missing_tables"] = total_missing

    if total_non_empty == 0:
        report["summary"]["overall_fill_verdict"] = "EMPTY"
    elif total_partially_filled > 0 or total_empty > 0 or total_missing > 0:
        report["summary"]["overall_fill_verdict"] = "PARTIALLY_FILLED"
    else:
        report["summary"]["overall_fill_verdict"] = "FULLY_FILLED"

    # Sequence gaps check
    if not filter_table or filter_table in ("sitzungen", "Sitzung"):
        report["session_sequence_spots"] = inspect_session_sequence_spots(session)

    # FTS5 check: if filtering specifically for reden_fts, count Rede rows independently
    if filter_table == "reden_fts":
        if "reden" in existing_tables_map:
            try:
                reden_count = session.execute(select(func.count()).select_from(Rede)).scalar() or 0
            except Exception:
                reden_count = 0
        else:
            reden_count = 0

    if not filter_table or filter_table in ("reden", "Rede", "reden_fts"):
        report["fts5_status"] = inspect_fts5_status(session, reden_count)

    return report


def print_human_report(session: Session, report: Dict[str, Any], verbose: bool = False) -> None:
    """Print leveled-up human readable inspection and missing spots report."""
    print("=" * 65)
    print(" OPENPARLAMENT DATABASE INSPECTOR & COMPLETENESS REPORT")
    print("=" * 65)
    print(f"Engine Bind: {report['engine']}")
    verdict = report['summary']['overall_fill_verdict']
    print(f"Overall Fill Status: [{verdict}]")
    print(
        f"Checked Tables: {report['summary']['total_tables_checked']} | "
        f"Fully Filled: {report['summary']['fully_filled_tables']} | "
        f"Partially Filled: {report['summary']['partially_filled_tables']} | "
        f"Empty: {report['summary']['empty_tables']} | "
        f"Missing: {report['summary']['missing_tables']}"
    )
    print("-" * 65)

    print(f"{'Table':<26} | {'Rows':<8} | {'Fill Status':<16} | {'Missing Spots'}")
    print("-" * 65)

    for tbl, t_data in report["tables"].items():
        name = f"{t_data.get('model_name', tbl)} ({tbl})"
        cnt = t_data.get("row_count", 0)
        status = t_data.get("fill_status", "UNKNOWN")
        m_spots = t_data.get("missing_spots", {})
        m_count = t_data.get("missing_spots_count", 0)

        if not t_data.get("exists"):
            desc = "[TABLE MISSING IN DB]"
        elif cnt == 0:
            desc = "[TABLE EMPTY (0 rows)]"
        elif m_count == 0:
            desc = "[100% COMPLETE - No missing spots]"
        else:
            # Highlight most relevant missing spots
            spot_summaries = [f"{k}: {v['count']} ({v['percent']}%)" for k, v in list(m_spots.items())[:2]]
            if len(m_spots) > 2:
                spot_summaries.append(f"+{len(m_spots) - 2} more")
            desc = ", ".join(spot_summaries)

        print(f"{name:<26} | {cnt:<8} | {status:<16} | {desc}")

    print("-" * 65)

    # Sequence spots
    seq_spots = report.get("session_sequence_spots", {})
    if seq_spots and seq_spots.get("wahlperioden"):
        print("Plenary Session Sequence Spots:")
        for wp, wp_info in seq_spots["wahlperioden"].items():
            gaps = wp_info.get("missing_session_spots", [])
            gap_str = f"GAPS: {gaps[:8]}{'...' if len(gaps) > 8 else ''} ({len(gaps)} missing)" if gaps else "Continuous (no gaps)"
            print(f"  WP {wp}: {wp_info['session_count']} sessions [range {wp_info['min_sitzungsnr']}..{wp_info['max_sitzungsnr']}] -> {gap_str}")
        print("-" * 65)

    # FTS5 index
    fts = report.get("fts5_status", {})
    if fts and fts.get("status") != "NON_SQLITE":
        print("FTS5 Full-Text Search Index (reden_fts):")
        print(f"  Status: {fts['status']} | Rows: {fts['row_count']} | Synced with Reden: {fts['synced']}")
        if fts.get("missing_speech_spots", 0) > 0:
            print(f"  -> WARNING: {fts['missing_speech_spots']} speech(es) missing from FTS5 index!")
        if fts.get("content_mismatch_count", 0) > 0:
            print(f"  -> WARNING: {fts['content_mismatch_count']} speech(es) have content mismatches in FTS5!")
        print("-" * 65)

    # Detailed / Verbose breakdown
    if verbose or any(t.get("missing_spots_count", 0) > 0 for t in report["tables"].values()):
        has_printed_header = False
        for tbl, t_data in report["tables"].items():
            m_spots = t_data.get("missing_spots", {})
            if m_spots:
                if not has_printed_header:
                    print("DETAILED MISSING SPOTS PER TABLE:")
                    has_printed_header = True
                print(f"  [{tbl}] (Total Rows: {t_data.get('row_count', 0)}):")
                for spot, val in m_spots.items():
                    print(f"    - {spot:<32}: {val['count']} rows missing ({val['percent']}%)")
        if has_printed_header:
            print("-" * 65)

    # Framing Results Breakdown by Segment & Variant
    try:
        framing_breakdown = session.execute(
            select(
                FramingResult.segment,
                FramingResult.variant,
                func.count(FramingResult.id),
            )
            .group_by(FramingResult.segment, FramingResult.variant)
            .order_by(FramingResult.segment, FramingResult.variant)
        ).all()

        if framing_breakdown:
            print("FramingResult Breakdown (segment, variant, count):")
            for seg, var, count in framing_breakdown:
                print(f"  segment='{seg}', variant='{var}': {count} terms")
            print("-" * 65)
        else:
            print("FramingResult Breakdown: No rows in framing_results.")
            print("-" * 65)
    except Exception as exc:
        print("Failed to query framing_results breakdown:", exc)
        print("-" * 65)

    # Sample Records (Rede, Zwischenruf, FramingResult)
    print("Sample Records Structure:")
    try:
        f_row = session.execute(select(FramingResult).limit(1)).scalar_one_or_none()
        if f_row:
            f_dict = {k: v for k, v in vars(f_row).items() if not k.startswith('_')}
            print(f"  FramingResult: {f_dict}")
        else:
            print("  FramingResult: <No rows found>")
    except Exception as exc:
        print("  FramingResult sample fetch failed:", exc)

    try:
        z_row = session.execute(select(Zwischenruf).limit(1)).scalar_one_or_none()
        if z_row:
            z_dict = {k: v for k, v in vars(z_row).items() if not k.startswith('_')}
            print(f"  Zwischenruf:   {z_dict}")
        else:
            print("  Zwischenruf:   <No rows found>")
    except Exception as exc:
        print("  Zwischenruf sample fetch failed:", exc)

    try:
        r_row = session.execute(select(Rede).limit(1)).scalar_one_or_none()
        if r_row:
            r_dict = {k: v for k, v in vars(r_row).items() if not k.startswith('_')}
            if 'text' in r_dict and r_dict['text']:
                r_dict['text'] = r_dict['text'][:80] + "..."
            print(f"  Rede:          {r_dict}")
        else:
            print("  Rede:          <No rows found>")
    except Exception as exc:
        print("  Rede sample fetch failed:", exc)

    print("=" * 65)


def run_inspect(
    db_url: Optional[str] = None,
    json_output: bool = False,
    verbose: bool = False,
    filter_table: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute database inspection with custom DB url or standard session."""
    if db_url:
        engine = create_engine(db_url)
        factory = sessionmaker(bind=engine)
        session = factory()
    else:
        session = get_session().__enter__()

    try:
        report = inspect_database(session, filter_table=filter_table)
        if json_output:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        else:
            print_human_report(session, report, verbose=verbose)
        return report
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect database structure, completeness, missing spots, and FTS5 synchronization."
    )
    parser.add_argument("--db-url", type=str, default=None, help="Custom SQLite DB URL or filepath.")
    parser.add_argument("--json", action="store_true", help="Output inspection report as JSON.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all detailed missing spot metrics.")
    parser.add_argument("--table", type=str, default=None, help="Inspect only a specific table.")

    args = parser.parse_args()

    # Normalize SQLite file path if given directly instead of sqlite:/// URL
    db_url = args.db_url
    if db_url and not ("://" in db_url):
        db_url = f"sqlite:///{Path(db_url).resolve().as_posix()}"

    run_inspect(
        db_url=db_url,
        json_output=args.json,
        verbose=args.verbose,
        filter_table=args.table,
    )


if __name__ == "__main__":
    main()
