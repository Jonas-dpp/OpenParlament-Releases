"""
Unit tests for scripts/compute_standard_deviations.py
Checks bounds validation [-1.0, +1.0] for sentiment metrics, distribution statistics,
clamping behavior, and CLI execution.
"""

import os
import json
import sqlite3
import pytest
import subprocess
import sys
import pandas as pd
from pathlib import Path

from scripts.compute_standard_deviations import compute_metrics

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def synthetic_db(tmp_path):
    """Creates a temporary SQLite database with edge cases, including out-of-bound sentiment scores."""
    db_file = tmp_path / "test_standard_deviations.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE sitzungen (
            sitzungs_id INTEGER PRIMARY KEY,
            wahlperiode INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE redner (
            redner_id INTEGER PRIMARY KEY,
            fraktion TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE reden (
            rede_id INTEGER PRIMARY KEY,
            sitzung_id INTEGER,
            redner_id INTEGER,
            sentiment_score REAL,
            text TEXT,
            clean_text TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE zwischenrufe (
            ruf_id INTEGER PRIMARY KEY,
            rede_id INTEGER,
            sentiment_score REAL,
            text TEXT
        )
    """)

    cur.execute("INSERT INTO sitzungen VALUES (1, 19), (2, 20)")
    cur.execute("INSERT INTO redner VALUES (1, 'SPD'), (2, 'CDU/CSU')")

    # Add speeches with normal and out-of-bounds scores
    for i in range(1, 60):
        # some raw out-of-bounds sentiment scores: -1.5, +2.0
        score = 2.0 if i == 1 else (-1.5 if i == 2 else (i - 30) / 30.0)
        cur.execute(
            "INSERT INTO reden VALUES (?, 1, 1, ?, 'Sehr geehrte Damen und Herren...', 'Sehr geehrte Damen und Herren...')",
            (i, score)
        )

    # Add interjections with normal and out-of-bounds scores
    # Also include extreme values for parser audit testing
    interjections = [
        (1, 1, 1.5, "Zuruf von der AfD: Danke schön!"),
        (2, 1, -2.0, "Alexander Hoffmann [CDU/CSU]: Das ist falsch!"),
        (3, 1, 0.8, "Beifall bei der SPD"),
        (4, 1, -0.9, "Buh-Rufe"),
        (5, 2, 0.5, "Zuruf"),
    ]
    for row in interjections:
        cur.execute("INSERT INTO zwischenrufe VALUES (?, ?, ?, ?)", row)

    conn.commit()
    conn.close()

    return str(db_file)


def test_sentiment_bounds_and_clamping(synthetic_db):
    """Verify that compute_metrics clamps sentiment values and produces min/max strictly in [-1.0, +1.0]."""
    results = compute_metrics(synthetic_db)

    # Check overall interjection sentiment bounds
    overall = results["zwischenruf_sentiment_overall"]
    assert overall["min"] >= -1.0
    assert overall["max"] <= 1.0
    assert -1.0 <= overall["mean"] <= 1.0
    assert -1.0 <= overall["median"] <= 1.0

    # Check interjection sentiment per WP bounds
    for wp, wp_stats in results["zwischenruf_sentiment_per_wp"].items():
        assert wp_stats["min"] >= -1.0
        assert wp_stats["max"] <= 1.0
        assert -1.0 <= wp_stats["mean"] <= 1.0
        assert -1.0 <= wp_stats["median"] <= 1.0

    # Check speech sentiment per faction bounds
    for faction, f_stats in results["speech_sentiment_per_faction"].items():
        assert f_stats["min"] >= -1.0
        assert f_stats["max"] <= 1.0
        assert -1.0 <= f_stats["mean"] <= 1.0
        assert -1.0 <= f_stats["median"] <= 1.0


def test_raw_vs_clipped_out_of_bounds_detection(synthetic_db):
    """Directly inspect database to check for values outside [-1.0, +1.0] and verify compute_metrics output handles them."""
    conn = sqlite3.connect(synthetic_db)
    df_zr = pd.read_sql_query("SELECT sentiment_score FROM zwischenrufe", conn)
    df_reden = pd.read_sql_query("SELECT sentiment_score FROM reden", conn)
    conn.close()

    # Confirm raw DB contains values outside [-1.0, +1.0] in synthetic setup
    unbounded_zr = df_zr[(df_zr["sentiment_score"] < -1.0) | (df_zr["sentiment_score"] > 1.0)]
    unbounded_reden = df_reden[(df_reden["sentiment_score"] < -1.0) | (df_reden["sentiment_score"] > 1.0)]
    assert len(unbounded_zr) > 0
    assert len(unbounded_reden) > 0

    # Execute compute_metrics and assert report statistics are strictly contained in [-1.0, +1.0]
    results = compute_metrics(synthetic_db)

    overall_min = results["zwischenruf_sentiment_overall"]["min"]
    overall_max = results["zwischenruf_sentiment_overall"]["max"]
    assert -1.0 <= overall_min <= 1.0
    assert -1.0 <= overall_max <= 1.0


def test_real_db_no_out_of_bounds_if_file_exists():
    """If data/openparlament.db exists, test that all reported sentiment metrics are strictly within [-1.0, +1.0]."""
    real_db = os.path.join(_PROJECT_ROOT, "data", "openparlament.db")
    if not os.path.exists(real_db):
        pytest.skip("data/openparlament.db does not exist")

    results = compute_metrics(real_db)

    # Check overall interjection sentiment
    overall = results["zwischenruf_sentiment_overall"]
    assert -1.0 <= overall["min"] <= 1.0, f"Min {overall['min']} outside [-1, 1]"
    assert -1.0 <= overall["max"] <= 1.0, f"Max {overall['max']} outside [-1, 1]"
    assert -1.0 <= overall["mean"] <= 1.0, f"Mean {overall['mean']} outside [-1, 1]"
    assert -1.0 <= overall["median"] <= 1.0, f"Median {overall['median']} outside [-1, 1]"

    # Check WP breakdown
    for wp, stats in results["zwischenruf_sentiment_per_wp"].items():
        assert -1.0 <= stats["min"] <= 1.0, f"{wp} min {stats['min']} outside [-1, 1]"
        assert -1.0 <= stats["max"] <= 1.0, f"{wp} max {stats['max']} outside [-1, 1]"

    # Check Faction speech breakdown
    for faction, stats in results["speech_sentiment_per_faction"].items():
        assert -1.0 <= stats["min"] <= 1.0, f"{faction} min {stats['min']} outside [-1, 1]"
        assert -1.0 <= stats["max"] <= 1.0, f"{faction} max {stats['max']} outside [-1, 1]"


def test_cli_execution(synthetic_db, tmp_path):
    """Test CLI execution producing output report JSON."""
    out_json = tmp_path / "report_out.json"
    cmd = [
        sys.executable,
        "scripts/compute_standard_deviations.py",
        "--db", synthetic_db,
        "--output", str(out_json)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "zwischenruf_sentiment_overall" in data
    assert "zwischenruf_audit_extremes" in data
    assert "zwischenruf_sentiment_per_wp" in data
    assert "speech_sentiment_per_faction" in data
