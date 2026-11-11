"""Tests for TOP-level aggregation in SitzungsKlima and analytics."""

from datetime import date
import pytest
from src.database import drop_db, get_session, init_db
import src.database as db_module
from src.models import Sitzung, Redner, Rede, Zwischenruf
from src.analytics import SitzungsKlima


@pytest.fixture(autouse=True)
def _use_in_memory_db(monkeypatch, tmp_path):
    """Redirect all DB access to a temporary file-backed SQLite database for each test."""
    db_file = tmp_path / "test_sitzungsklima_top.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file}")
    db_module.reset_db_state()
    init_db()
    SitzungsKlima._global_top_bounds_cache.clear()
    SitzungsKlima._global_bounds_cache.clear()
    yield
    drop_db()
    db_module.reset_db_state()
    SitzungsKlima._global_top_bounds_cache.clear()
    SitzungsKlima._global_bounds_cache.clear()


def _seed_top_data(session):
    sz = Sitzung(wahlperiode=20, sitzungsnr=99, datum=date(2023, 1, 1), gesamtwortzahl=500)
    redner = Redner(vorname="Max", nachname="Mustermann", fraktion="SPD")
    rede1 = Rede(sitzung=sz, redner=redner, text="Rede TOP 10", wortanzahl=100, tagesordnungspunkt="Tagesordnungspunkt 10: Bürgergeld-Gesetz")
    rede2 = Rede(sitzung=sz, redner=redner, text="Rede ZP 12", wortanzahl=100, tagesordnungspunkt="Zusatzpunkt 12: Geschäftsordnung")

    z1 = Zwischenruf(rede=rede1, text="Sehr richtig!", fraktion="SPD", sentiment_score=0.8, ton_label="Neutral", kategorie="Beifall")
    z2 = Zwischenruf(rede=rede2, text="Widerspruch!", fraktion="AfD", sentiment_score=-0.8, ton_label="Aggression", kategorie="Unruhe")

    session.add_all([sz, redner, rede1, rede2, z1, z2])
    session.flush()


def test_clean_top_title():
    """Verify clean_top_title formats known TOP titles and falls back cleanly."""
    assert SitzungsKlima.clean_top_title("Tagesordnungspunkt 10: Bürgergeld-Gesetz") == "TOP 10: Bürgergeld-Gesetz (2./3. Beratung)"
    assert SitzungsKlima.clean_top_title("Zusatzpunkt 12: Geschäftsordnung") == "ZP 12: Auslegung der Geschäftsordnung"
    assert "TOP 4" in SitzungsKlima.clean_top_title("TOP 4: Beratung")


def test_sitzungsklima_klima_per_top():
    """Test that klima_per_top returns a DataFrame with the expected columns and valid data."""
    with get_session() as session:
        _seed_top_data(session)
        analyzer = SitzungsKlima(session)
        df = analyzer.klima_per_top()

        assert not df.empty
        expected_cols = [
            "sitzungsnr",
            "wahlperiode",
            "datum",
            "tagesordnungspunkt",
            "top_titel",
            "avg_sentiment",
            "anteil_aggression_pct",
            "zwischenrufe_pro_rede",
            "anteil_unruhe_pct",
            "anzahl_reden",
            "gesamt_zwr",
            "temperatur_index",
        ]
        for col in expected_cols:
            assert col in df.columns

        # Verify that temperature is bounded within [0, 1]
        assert (df["temperatur_index"] >= 0.0).all()
        assert (df["temperatur_index"] <= 1.0).all()

        # Check hottest_tops helper
        top_hottest = analyzer.hottest_tops(n=5)
        assert len(top_hottest) <= 5
        if len(top_hottest) > 1:
            assert top_hottest.iloc[0]["temperatur_index"] >= top_hottest.iloc[1]["temperatur_index"]
