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
Unit tests for scripts/db_inspect.py
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.db_inspect import (  # noqa: E402
    inspect_database,
    inspect_session_sequence_spots,
)
from src.models import (  # noqa: E402
    Base,
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


@pytest.fixture
def test_db(tmp_path):
    """Create an isolated test database session and DB URL."""
    db_file = tmp_path / "test_db_inspect.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session, db_url
    session.close()


def test_inspect_database_empty(test_db):
    session, _ = test_db
    report = inspect_database(session)

    assert report["summary"]["total_tables_checked"] == 9
    assert report["summary"]["empty_tables"] == 9
    assert report["summary"]["overall_fill_verdict"] == "EMPTY"
    assert report["tables"]["sitzungen"]["fill_status"] == "EMPTY"
    assert report["tables"]["reden"]["fill_status"] == "EMPTY"
    assert report["session_sequence_spots"]["has_gaps"] is False


def test_inspect_database_fully_filled(test_db):
    session, _ = test_db

    # Create session
    sitzung = Sitzung(
        wahlperiode=20,
        sitzungsnr=1,
        datum=datetime.date(2023, 1, 15),
        wochentag="Sonntag",
        titel="1. Plenarsitzung",
        gesamtwortzahl=500,
    )
    # Create speaker
    redner = Redner(
        vorname="Erika",
        nachname="Mustermann",
        bundestag_id="12345",
        fraktion="SPD",
    )
    session.add_all([sitzung, redner])
    session.flush()

    # Create speech with all text layers and dedicated tone probabilities
    rede = Rede(
        sitzung_id=sitzung.sitzungs_id,
        redner_id=redner.redner_id,
        text="Herr Präsident, meine Damen und Herren!",
        raw_text="Herr Präsident, meine Damen und Herren!",
        clean_text="Herr Präsident, meine Damen und Herren!",
        normalized_text="herr präsident meine damen und herren",
        metadata_text="TOP 1: Haushalt",
        tagesordnungspunkt="TOP 1",
        wortanzahl=6,
        sentiment_score=0.42,
        ton_label="Neutral",
        tone_aggression=0.05,
        tone_sarkasmus=0.02,
        tone_humor=0.01,
        tone_neutral=0.92,
        tone_scores={"Neutral": 0.92},
        adressaten="Alle",
    )
    session.add(rede)
    session.flush()

    # Create interjection with dedicated tone probabilities
    zwischenruf = Zwischenruf(
        rede_id=rede.rede_id,
        text="Sehr richtig!",
        fraktion="SPD",
        sentiment_score=0.8,
        kategorie="Beifall",
        ton_label="Humor",
        tone_aggression=0.0,
        tone_sarkasmus=0.1,
        tone_humor=0.8,
        tone_neutral=0.1,
        tone_scores={"Humor": 0.8},
        adressaten="Erika Mustermann",
    )
    session.add(zwischenruf)
    session.flush()

    # Create stylometry
    stylometry = StylometrieFeatures(
        rede_id=rede.rede_id,
        avg_sentence_length=12.5,
        passive_ratio=0.1,
        type_token_ratio=0.75,
        fill_word_ratio=0.02,
        punctuation_freq={"!": 1},
    )
    session.add(stylometry)

    # Create speech complexity
    complexity = SpeechComplexityResult(
        rede_id=rede.rede_id,
        wahlperiode=20,
        redner_id=redner.redner_id,
        wstf_score=7.5,
        flesch_score=65.0,
        avg_sentence_length=12.5,
        subordinate_clause_ratio=0.2,
        avg_word_length=5.1,
        long_word_ratio=0.15,
        lexical_diversity=0.75,
    )
    session.add(complexity)

    # Create reaction gap
    reaction_gap = ReactionGapResult(
        rede_id=rede.rede_id,
        zwischenruf_id=zwischenruf.ruf_id,
        topic_id="TOP 1",
        speaker_sentiment=0.42,
        reaction_sentiment=0.8,
        gap=0.38,
        time_to_first_interjection=2.5,
    )
    session.add(reaction_gap)

    # Create framing result
    framing = FramingResult(
        wahlperiode=20,
        fraktion="SPD",
        segment="all",
        variant="global",
        term="gerechtigkeit",
        score=0.08,
    )
    session.add(framing)

    # Create lexical result with both total_tokens and total_types > 0
    lexical = LexicalResult(
        wahlperiode=20,
        fraktion="SPD",
        ttr=0.75,
        msttr=0.72,
        lexical_density=0.55,
        total_tokens=1000,
        total_types=750,
    )
    session.add(lexical)

    session.commit()

    report = inspect_database(session)

    assert report["summary"]["fully_filled_tables"] == 9
    assert report["summary"]["partially_filled_tables"] == 0
    assert report["summary"]["empty_tables"] == 0
    assert report["summary"]["overall_fill_verdict"] == "FULLY_FILLED"

    for _, t_data in report["tables"].items():
        assert t_data["fill_status"] == "FULLY_FILLED"
        assert t_data["missing_spots_count"] == 0


def test_inspect_database_detects_missing_spots_and_orphans(test_db):
    session, _ = test_db

    # Incomplete session: missing datum, titel, and zero wortzahl
    sitzung = Sitzung(
        wahlperiode=19,
        sitzungsnr=1,
        datum=None,
        wochentag=None,
        titel="",
        gesamtwortzahl=0,
    )
    # Incomplete speaker: missing vorname, fraktion
    redner = Redner(
        vorname="",
        nachname="Schmidt",
        fraktion=None,
    )
    session.add_all([sitzung, redner])
    session.flush()

    # Incomplete speech: missing text layers, sentiment, tones
    rede = Rede(
        sitzung_id=sitzung.sitzungs_id,
        redner_id=redner.redner_id,
        text="",
        raw_text=None,
        clean_text=None,
        normalized_text=None,
        metadata_text=None,
        tagesordnungspunkt=None,
        wortanzahl=0,
        sentiment_score=None,
        tone_aggression=None,
        tone_sarkasmus=None,
        tone_humor=None,
        tone_neutral=None,
    )
    session.add(rede)
    session.flush()

    # Incomplete interjection with orphaned invalid rede_id (99999)
    zwischenruf = Zwischenruf(
        rede_id=99999,
        text="",
        fraktion=None,
        sentiment_score=None,
        tone_aggression=None,
        tone_sarkasmus=None,
        tone_humor=None,
        tone_neutral=None,
    )
    session.add(zwischenruf)

    # Incomplete lexical with zero total_types
    lexical = LexicalResult(
        wahlperiode=19,
        fraktion="CDU/CSU",
        ttr=0.5,
        msttr=0.5,
        lexical_density=0.5,
        total_tokens=100,
        total_types=0,
    )
    session.add(lexical)

    session.commit()

    report = inspect_database(session)

    assert report["summary"]["overall_fill_verdict"] == "PARTIALLY_FILLED"

    s_res = report["tables"]["sitzungen"]
    assert s_res["fill_status"] == "MISSING_ENTRIES"
    assert "missing_datum" in s_res["missing_spots"]
    assert "missing_titel" in s_res["missing_spots"]
    assert "zero_gesamtwortzahl" in s_res["missing_spots"]

    r_res = report["tables"]["redner"]
    assert r_res["fill_status"] == "MISSING_ENTRIES"
    assert "missing_vorname" in r_res["missing_spots"]
    assert "missing_fraktion" in r_res["missing_spots"]

    rede_res = report["tables"]["reden"]
    assert rede_res["fill_status"] == "MISSING_ENTRIES"
    assert "missing_text" in rede_res["missing_spots"]
    assert "missing_normalized_text" in rede_res["missing_spots"]
    assert "missing_metadata_text" in rede_res["missing_spots"]
    assert "missing_sentiment_score" in rede_res["missing_spots"]
    assert "missing_tone_sarkasmus" in rede_res["missing_spots"]

    zr_res = report["tables"]["zwischenrufe"]
    assert zr_res["fill_status"] == "MISSING_ENTRIES"
    assert "missing_text" in zr_res["missing_spots"]
    assert "orphaned_invalid_rede_id" in zr_res["missing_spots"]
    assert "missing_tone_neutral" in zr_res["missing_spots"]

    lex_res = report["tables"]["lexical_results"]
    assert lex_res["fill_status"] == "MISSING_ENTRIES"
    assert "zero_total_types" in lex_res["missing_spots"]


def test_inspect_session_sequence_spots_gaps_including_prefix(test_db):
    session, _ = test_db
    # Sessions 5 and 6 only (missing 1, 2, 3, 4)
    s5 = Sitzung(wahlperiode=19, sitzungsnr=5, datum=datetime.date(2018, 1, 5), wochentag="Freitag")
    s6 = Sitzung(wahlperiode=19, sitzungsnr=6, datum=datetime.date(2018, 1, 6), wochentag="Samstag")
    session.add_all([s5, s6])
    session.commit()

    seq = inspect_session_sequence_spots(session)
    assert seq["has_gaps"] is True
    assert seq["total_missing_sessions"] == 4
    assert seq["wahlperioden"][19]["missing_session_spots"] == [1, 2, 3, 4]
    assert seq["wahlperioden"][19]["status"] == "GAP_DETECTED"


def test_inspect_table_filter(test_db):
    session, _ = test_db
    report = inspect_database(session, filter_table="sitzungen")
    assert report["summary"]["total_tables_checked"] == 1
    assert "sitzungen" in report["tables"]
    assert "reden" not in report["tables"]


def test_inspect_table_filter_reden_fts(test_db):
    session, _ = test_db
    # Create speech and FTS5 table
    sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=datetime.date(2023, 1, 1), wochentag="Sonntag")
    redner = Redner(vorname="Max", nachname="Mustermann", fraktion="CDU/CSU")
    session.add_all([sitzung, redner])
    session.flush()

    rede = Rede(
        sitzung_id=sitzung.sitzungs_id,
        redner_id=redner.redner_id,
        text="Testrede",
        raw_text="Testrede",
        clean_text="Testrede",
        normalized_text="testrede",
        metadata_text="",
        tagesordnungspunkt="TOP 1",
        wortanzahl=1,
    )
    session.add(rede)
    session.commit()

    # Create FTS5 virtual table and insert synchronized content
    session.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS reden_fts USING fts5(text)"))
    session.execute(text(f"INSERT INTO reden_fts(rowid, text) VALUES ({rede.rede_id}, 'Testrede')"))
    session.commit()

    report = inspect_database(session, filter_table="reden_fts")
    assert "fts5_status" in report
    assert report["fts5_status"]["synced"] is True
    assert report["fts5_status"]["status"] == "FULLY_SYNCED"


def test_fts5_status_desynchronized_content_and_missing_rows(test_db):
    session, _ = test_db
    sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=datetime.date(2023, 1, 1), wochentag="Sonntag")
    redner = Redner(vorname="Max", nachname="Mustermann", fraktion="CDU/CSU")
    session.add_all([sitzung, redner])
    session.flush()

    rede1 = Rede(
        sitzung_id=sitzung.sitzungs_id,
        redner_id=redner.redner_id,
        text="Original Rede 1",
        wortanzahl=3,
    )
    rede2 = Rede(
        sitzung_id=sitzung.sitzungs_id,
        redner_id=redner.redner_id,
        text="Original Rede 2",
        wortanzahl=3,
    )
    session.add_all([rede1, rede2])
    session.commit()

    # FTS contains rowid with mismatched content and missing rede2
    session.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS reden_fts USING fts5(text)"))
    session.execute(text(f"INSERT INTO reden_fts(rowid, text) VALUES ({rede1.rede_id}, 'Altered Content')"))
    session.commit()

    report = inspect_database(session, filter_table="reden")
    fts = report["fts5_status"]
    assert fts["synced"] is False
    assert fts["status"] == "DESYNCHRONIZED"
    assert fts["missing_speech_spots"] == 1
    assert fts["content_mismatch_count"] == 1


def test_cli_execution_with_flags(test_db, tmp_path):
    _, db_url = test_db

    # Call with --json
    cmd = [
        sys.executable,
        "scripts/db_inspect.py",
        "--db-url",
        db_url,
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert "summary" in parsed
    assert "tables" in parsed

    # Call with --verbose and --table
    cmd_verbose = [
        sys.executable,
        "scripts/db_inspect.py",
        "--db-url",
        db_url,
        "--verbose",
        "--table",
        "redner",
    ]
    res_verbose = subprocess.run(cmd_verbose, capture_output=True, text=True, check=True)
    assert "Redner (redner)" in res_verbose.stdout
    assert "OPENPARLAMENT DATABASE INSPECTOR" in res_verbose.stdout
