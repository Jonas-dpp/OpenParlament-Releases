"""
Unit tests for scripts/extract_sarcasm_case_studies.py
Tests extraction of sarcasm case studies, database queries, sentence context extraction
(3 sentences before, 1 sentence after), opposition -> government faction filtering,
annotation fields, and CSV/Excel/Markdown output generation.
"""

import sqlite3
import pytest
import subprocess
import sys
import pandas as pd
from pathlib import Path

from scripts.extract_sarcasm_case_studies import (
    split_into_sentences,
    extract_context_from_speech,
    canonicalise_faction,
    is_opposition_to_government,
    extract_case_studies,
    generate_markdown,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def mock_sarcasm_db(tmp_path):
    """Creates a temporary SQLite database populated with interjections and speeches."""
    db_file = tmp_path / "test_sarcasm.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE sitzungen (
            sitzungs_id INTEGER PRIMARY KEY,
            wahlperiode INTEGER,
            sitzungsnr INTEGER,
            datum TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE redner (
            redner_id INTEGER PRIMARY KEY,
            vorname TEXT,
            nachname TEXT,
            fraktion TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE reden (
            rede_id INTEGER PRIMARY KEY,
            sitzung_id INTEGER,
            redner_id INTEGER,
            text TEXT,
            clean_text TEXT,
            raw_text TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE zwischenrufe (
            ruf_id INTEGER PRIMARY KEY,
            rede_id INTEGER,
            sentiment_score REAL,
            fraktion TEXT,
            text TEXT
        )
    """)

    # WP 20: SPD & BÜNDNIS 90/DIE GRÜNEN & FDP = Government; CDU/CSU & AfD & DIE LINKE = Opposition
    cur.execute("INSERT INTO sitzungen VALUES (10, 20, 94, '2023-03-30')")
    cur.execute("INSERT INTO redner VALUES (100, 'Olaf', 'Scholz', 'SPD')")  # Government
    cur.execute("INSERT INTO redner VALUES (101, 'Friedrich', 'Merz', 'CDU/CSU')")  # Opposition

    speech_1_text = (
        "Sehr geehrte Frau Präsidentin! Meine Damen und Herren. Wir verabschieden heute ein wichtiges Gesetz. "
        "Zurufe von der AfD: Danke schön! Das ist der zweite Satz nach dem Ruf. Und hier kommt die Zusammenfassung."
    )
    cur.execute(
        "INSERT INTO reden VALUES (1, 10, 100, ?, ?, ?)",
        (speech_1_text, speech_1_text, speech_1_text),
    )

    speech_2_text = (
        "Erste Aussage des Oppositionsredners. Zweite Aussage. Dritte Aussage. "
        "Beifall bei der SPD Und die vierte Aussage."
    )
    cur.execute(
        "INSERT INTO reden VALUES (2, 10, 101, ?, ?, ?)",
        (speech_2_text, speech_2_text, speech_2_text),
    )

    interjections = [
        # Ruf 1: Opposition (AfD) -> Government (SPD) with sentiment 0.9880 (matches criteria)
        (1, 1, 0.9880, "AfD", "Zurufe von der AfD: Danke schön!"),
        # Ruf 2: Opposition (CDU/CSU) -> Government (SPD) with sentiment 0.9650 (matches criteria)
        (2, 1, 0.9650, "CDU/CSU", "Enak Ferlemann [CDU/CSU]: Sehr gut!"),
        # Ruf 3: Government (SPD) -> Opposition (CDU/CSU) with sentiment 0.9900 (does NOT match criteria: Gov -> Opp)
        (3, 2, 0.9900, "SPD", "Beifall bei der SPD"),
        # Ruf 4: Opposition (AfD) -> Government (SPD) with sentiment 0.9100 (does NOT match criteria: sentiment < 0.95)
        (4, 1, 0.9100, "AfD", "Das ist prima!"),
    ]
    for row in interjections:
        cur.execute("INSERT INTO zwischenrufe VALUES (?, ?, ?, ?, ?)", row)

    conn.commit()
    conn.close()

    return str(db_file)


def test_split_into_sentences():
    text = "Erster Satz. Zweiter Satz! Dritter Satz?"
    sents = split_into_sentences(text)
    assert sents == ["Erster Satz.", "Zweiter Satz!", "Dritter Satz?"]


def test_canonicalise_faction():
    assert canonicalise_faction("SPD") == "SPD"
    assert canonicalise_faction("CDU/CSU") == "CDU/CSU"
    assert canonicalise_faction(" Bündnis 90/Die Grünen ") == "BÜNDNIS 90/DIE GRÜNEN"
    assert canonicalise_faction("AfD") == "AfD"
    assert canonicalise_faction("DIE LINKE") == "DIE LINKE"
    assert canonicalise_faction("") == ""
    assert canonicalise_faction(None) == ""


def test_is_opposition_to_government():
    # WP 20: AfD (Opp) -> SPD (Gov) = True
    assert is_opposition_to_government("AfD", "SPD", 20) is True
    # WP 20: CDU/CSU (Opp) -> SPD (Gov) = True
    assert is_opposition_to_government("CDU/CSU", "SPD", 20) is True
    # WP 20: SPD (Gov) -> CDU/CSU (Opp) = False
    assert is_opposition_to_government("SPD", "CDU/CSU", 20) is False
    # WP 20: SPD (Gov) -> FDP (Gov) = False
    assert is_opposition_to_government("SPD", "FDP", 20) is False


def test_extract_context_from_speech():
    speech = (
        "Satz 1. Satz 2. Satz 3. Interjektion hier. Satz 4. Satz 5."
    )
    interjection = "Interjektion hier."
    before, after = extract_context_from_speech(speech, interjection, before_count=3, after_count=1)

    assert before == "Satz 1. Satz 2. Satz 3."
    assert after == "Satz 4."


def test_extract_case_studies_validation(mock_sarcasm_db):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        extract_case_studies(mock_sarcasm_db, limit=0)

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        extract_case_studies(mock_sarcasm_db, limit=-5)

    with pytest.raises(ValueError, match="min_sentiment must be a float between -1.0 and 1.0"):
        extract_case_studies(mock_sarcasm_db, limit=50, min_sentiment=2.5)


def test_extract_case_studies(mock_sarcasm_db):
    df_report = extract_case_studies(mock_sarcasm_db, limit=50, min_sentiment=0.95)

    # Only 2 interjections match: Ruf 1 (AfD -> SPD, 0.9880) and Ruf 2 (CDU/CSU -> SPD, 0.9650)
    assert len(df_report) == 2
    assert "interjection_id" in df_report.columns
    assert "zwischenruf_text" in df_report.columns
    assert "hauptredner" in df_report.columns
    assert "kontext_5_saetze_vorher" in df_report.columns
    assert "kontext_3_satz_nachher" in df_report.columns
    assert "manual_label" in df_report.columns
    assert "error_category" in df_report.columns
    assert "notes" in df_report.columns

    # Verify filtering accuracy
    afd_row = df_report[df_report["interjection_id"] == 1].iloc[0]
    assert afd_row["hauptredner"] == "Olaf Scholz"
    assert afd_row["hauptredner_fraktion"] == "SPD"
    assert afd_row["zwischenruf_fraktion"] == "AfD"
    assert "Sehr geehrte Frau Präsidentin!" in afd_row["kontext_5_saetze_vorher"]


def test_generate_markdown(mock_sarcasm_db):
    df_report = extract_case_studies(mock_sarcasm_db, limit=50, min_sentiment=0.95)
    md = generate_markdown(df_report)

    assert "# Qualitative Sarcasm Case Studies" in md
    assert "Olaf Scholz" in md
    assert "Zurufe von der AfD: Danke schön!" in md
    assert "manual_label" in md


def test_sarcasm_cli_execution(mock_sarcasm_db, tmp_path):
    out_csv = tmp_path / "sarcasm.csv"
    out_excel = tmp_path / "sarcasm.xlsx"
    out_md = tmp_path / "sarcasm.md"

    cmd = [
        sys.executable,
        "scripts/extract_sarcasm_case_studies.py",
        "--db", mock_sarcasm_db,
        "--limit", "50",
        "--min-sentiment", "0.95",
        "--output-csv", str(out_csv),
        "--output-excel", str(out_excel),
        "--output-md", str(out_md),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert out_csv.exists()
    assert out_md.exists()

    try:
        import openpyxl  # noqa: F401
        assert out_excel.exists()
    except ImportError:
        pass

    df_csv = pd.read_csv(out_csv)
    assert len(df_csv) == 2
    assert "manual_label" in df_csv.columns
