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
Unit tests for scripts/check_db_integrity.py
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.check_db_integrity import (  # noqa: E402
    check_schema_integrity,
    check_sitzungen_sequence,
    check_unfilled_base_cells,
    check_unfilled_precomputation_cells,
    run_db_check,
)
from src.models import Base, Rede, Redner, Sitzung  # noqa: E402


@pytest.fixture
def test_db_session(tmp_path):
    """Create a fresh isolated SQLite database session for testing."""
    db_file = tmp_path / "test_integrity.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session, db_url
    session.close()


def test_schema_integrity_complete_db(test_db_session):
    session, _ = test_db_session
    results, valid_cols = check_schema_integrity(session)
    assert results["status"] == "OK"
    assert len(results["errors"]) == 0
    assert "sitzungen.sitzungs_id" in valid_cols
    assert "reden.text" in valid_cols


def test_sitzungen_sequence_clean(test_db_session):
    session, _ = test_db_session
    # Add contiguous sessions in WP 19
    s1 = Sitzung(wahlperiode=19, sitzungsnr=1, datum=datetime.date(2020, 1, 1), wochentag="Mittwoch", titel="T1", gesamtwortzahl=100)
    s2 = Sitzung(wahlperiode=19, sitzungsnr=2, datum=datetime.date(2020, 1, 2), wochentag="Donnerstag", titel="T2", gesamtwortzahl=200)
    session.add_all([s1, s2])
    session.commit()

    seq_results = check_sitzungen_sequence(session)
    assert seq_results["status"] == "OK"
    assert len(seq_results["errors"]) == 0
    assert len(seq_results["warnings"]) == 0
    assert seq_results["wahlperioden"][19]["missing_sitzungsnr"] == []


def test_sitzungen_sequence_gaps_and_inversion(test_db_session):
    session, _ = test_db_session
    # Gapped sessions (1, 3 -> missing 2) and date inversion
    s1 = Sitzung(wahlperiode=20, sitzungsnr=1, datum=datetime.date(2022, 5, 10), wochentag="Dienstag", titel="T1", gesamtwortzahl=100)
    s3 = Sitzung(wahlperiode=20, sitzungsnr=3, datum=datetime.date(2022, 5, 5), wochentag="Donnerstag", titel="T3", gesamtwortzahl=100)
    session.add_all([s1, s3])
    session.commit()

    seq_results = check_sitzungen_sequence(session)
    assert seq_results["status"] == "WARNING"
    assert seq_results["wahlperioden"][20]["missing_sitzungsnr"] == [2]
    assert len(seq_results["wahlperioden"][20]["date_inversions"]) == 1


def test_check_unfilled_base_cells_errors(test_db_session):
    session, _ = test_db_session
    _, valid_cols = check_schema_integrity(session)

    # Session missing datum
    s = Sitzung(wahlperiode=21, sitzungsnr=1, datum=None, wochentag=None, titel="Test", gesamtwortzahl=0)
    # Speaker missing vorname
    r = Redner(vorname="", nachname="Müller", fraktion=None)
    session.add_all([s, r])
    session.commit()

    base_results = check_unfilled_base_cells(session, valid_cols)
    assert base_results["status"] == "ERROR"
    assert any("missing 'datum'" in err for err in base_results["errors"])
    assert any("missing 'vorname'" in err for err in base_results["errors"])


def test_check_unfilled_precomputation_cells_warnings(test_db_session):
    session, _ = test_db_session
    _, valid_cols = check_schema_integrity(session)

    s = Sitzung(wahlperiode=21, sitzungsnr=1, datum=datetime.date(2023, 1, 1), wochentag="Sonntag", titel="Test", gesamtwortzahl=50)
    redner = Redner(vorname="Max", nachname="Mustermann", fraktion="CDU/CSU")
    session.add_all([s, redner])
    session.flush()

    rede = Rede(sitzung_id=s.sitzungs_id, redner_id=redner.redner_id, text="Rede Text", wortanzahl=2, tagesordnungspunkt="TOP 1")
    session.add(rede)
    session.commit()

    precomp_results = check_unfilled_precomputation_cells(session, valid_cols)
    assert precomp_results["status"] == "WARNING"
    assert any("missing 'sentiment_score'" in warn for warn in precomp_results["warnings"])
    assert any("missing 'StylometrieFeatures'" in warn for warn in precomp_results["warnings"])


def test_run_db_check_end_to_end(test_db_session, tmp_path):
    session, db_url = test_db_session
    # Populate a valid minimal dataset
    s = Sitzung(wahlperiode=19, sitzungsnr=1, datum=datetime.date(2020, 1, 1), wochentag="Mittwoch", titel="Titel", gesamtwortzahl=500)
    redner = Redner(vorname="Anna", nachname="Schmidt", fraktion="SPD")
    session.add_all([s, redner])
    session.flush()

    rede = Rede(
        sitzung_id=s.sitzungs_id,
        redner_id=redner.redner_id,
        text="Sehr geehrte Frau Präsidentin!",
        raw_text="Sehr geehrte Frau Präsidentin!",
        clean_text="Sehr geehrte Frau Präsidentin!",
        wortanzahl=4,
        tagesordnungspunkt="TOP 1",
    )
    session.add(rede)
    session.commit()

    report = run_db_check(db_url=db_url)
    assert isinstance(report, dict)
    assert "overall_status" in report
    assert "sitzungen_sequence" in report
    assert "base_integrity" in report


def test_cli_execution_with_json_and_output_file(test_db_session, tmp_path):
    _, db_url = test_db_session
    out_json = tmp_path / "report.json"

    cmd = [
        sys.executable,
        "scripts/check_db_integrity.py",
        "--db-url",
        db_url,
        "--json",
        "-o",
        str(out_json),
    ]

    subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Output file should exist and contain JSON
    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "overall_status" in data
