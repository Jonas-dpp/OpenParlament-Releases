"""
OpenParlament - Polarization Improvements Unit Tests
Copyright (C) 2026 Jules
"""

import pytest
import pandas as pd
from datetime import date
from src.database import drop_db, init_db, get_session
from src.analytics import InteraktionsNetzwerk


@pytest.fixture(autouse=True)
def _use_in_memory_db_polarization(monkeypatch):
    """Redirect all DB access to an in-memory SQLite database for each test."""
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", "sqlite:///:memory:")
    import src.database as db_module
    db_module._engine = None
    db_module._SessionFactory = None
    init_db()
    yield
    drop_db()
    db_module._engine = None
    db_module._SessionFactory = None


def test_polarization_index_empty_db():
    """Verify that polarization_index returns a schema-correct empty DataFrame on empty DB."""
    with get_session() as s:
        net = InteraktionsNetzwerk(s)
        df = net.polarization_index(wahlperiode=20)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    expected_cols = [
        "fraktion",
        "polarization_index",
        "negative_external_interactions",
        "all_external_interactions"
    ]
    assert list(df.columns) == expected_cols


def test_polarization_index_filtered_empty():
    """Verify that polarization_index returns a schema-correct empty DataFrame on non-matching dates."""
    with get_session() as s:
        from src.models import Sitzung, Redner, Rede, Zwischenruf
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner_spd = Redner(vorname="A", nachname="SPD-R", fraktion="SPD")
        redner_cdu = Redner(vorname="B", nachname="CDU-R", fraktion="CDU/CSU")
        s.add_all([sitzung, redner_spd, redner_cdu])
        s.flush()

        rede_spd = Rede(sitzung_id=sitzung.sitzungs_id, redner_id=redner_spd.redner_id, text="Text", wortanzahl=5)
        rede_cdu = Rede(sitzung_id=sitzung.sitzungs_id, redner_id=redner_cdu.redner_id, text="Text", wortanzahl=5)
        s.add_all([rede_spd, rede_cdu])
        s.flush()

        z1 = Zwischenruf(rede_id=rede_cdu.rede_id, text="Beifall!", fraktion="SPD", sentiment_score=0.8, kategorie="Beifall")
        s.add_all([z1])
        s.flush()

        net = InteraktionsNetzwerk(s)
        df = net.polarization_index(wahlperiode=20, datum_von=date(2025, 1, 1), datum_bis=date(2025, 12, 31))

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    expected_cols = [
        "fraktion",
        "polarization_index",
        "negative_external_interactions",
        "all_external_interactions"
    ]
    assert list(df.columns) == expected_cols
