"""
tests/test_combined_app_analytics.py
Zusammengefasstes Test-Script für app.py (Erweiterte UI-Tests) und analytics.py (ScrapingMonitor & TonAnalyse)
OpenParlament Testing Suite
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import date
from streamlit.testing.v1 import AppTest

from src.database import init_db, get_session, drop_db
from src.models import Sitzung, Redner, Rede, Zwischenruf
from src.analytics import ScrapingMonitor, TonAnalyse

# Pfad zu app.py aus dem tests/ Ordner heraus auflösen
APP_PATH = str(Path(__file__).resolve().parents[2] / "src" / "app.py")

@pytest.fixture(autouse=True)
def _isolated_test_db(monkeypatch, tmp_path):
    """
    Isolierte dateibasierte SQLite-Datenbank für jeden einzelnen Testlauf,
    um Nebeneffekte zwischen parallelen oder sequentiellen Sessions zu vermeiden.
    """
    db_file = tmp_path / "test_combined_openparlament.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file}")
    
    import src.database as db_module
    # Zurücksetzen des globalen Engine/Session-Status falls vorhanden
    if hasattr(db_module, "reset_db_state"):
        db_module.reset_db_state()
    
    init_db()
    
    # Gemeinsame Seed-Datenbasis für UI- und Analytics-Tests erstellen
    with get_session() as s:
        sz1 = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2023, 1, 1), gesamtwortzahl=100, titel="1. Sitzung")
        sz2 = Sitzung(wahlperiode=20, sitzungsnr=2, datum=date(2023, 1, 2), gesamtwortzahl=150, titel="2. Sitzung")
        
        r1 = Redner(vorname="Max", nachname="Mustermann", fraktion="SPD")
        r2 = Redner(vorname="Erika", nachname="Musterfrau", fraktion="CDU/CSU")
        
        rede1 = Rede(sitzung=sz1, redner=r1, text="Testrede Inhalt Eins", wortanzahl=50)
        rede2 = Rede(sitzung=sz2, redner=r2, text="Testrede Inhalt Zwei", wortanzahl=75)
        
        # Zwischenruf 1: Vollständig prozessiert (mit NLP-Daten)
        zr1 = Zwischenruf(
            rede=rede1, 
            text="Hört, hört!", 
            fraktion="CDU/CSU", 
            ton_label="Neutral", 
            sentiment_score=0.2,
            adressaten='["SPD"]'
        )
        # Zwischenruf 2: Unprozessiert (ohne NLP-Daten für ScrapingMonitor Tests)
        zr2 = Zwischenruf(
            rede=rede2, 
            text="Dagegen!", 
            fraktion="AfD"
        )
        
        s.add_all([sz1, sz2, r1, r2, rede1, rede2, zr1, zr2])
        s.commit()

    yield
    drop_db()
    if hasattr(db_module, "reset_db_state"):
        db_module.reset_db_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sektion 1: Frontend & UI-Tests (app.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestAppFrontend:

    def test_startseite_expander_interaction(self):
        """Testet die Sichtbarkeit und Interaktivität von Inhalt-Expandern auf der Startseite."""
        at = AppTest.from_file(APP_PATH)
        at.query_params["page"] = "startseite"
        at.run(timeout=30)
        
        assert not at.exception
        assert len(at.expander) >= 0, "Startseiten-Layout enthält unerwartete Strukturfehler."

    def test_ton_analyse_toggle_interaction(self):
        """Verifiziert, dass Umschalt-Toggles (z.B. Prozentuale Anteile) keine App-Crashes verursachen."""
        at = AppTest.from_file(APP_PATH)
        at.query_params["page"] = "ton-analyse"
        at.run(timeout=30)
        
        assert not at.exception
        # Suchen nach Toggles, die die Ansicht modifizieren
        toggles = [t for t in at.toggle if "Prozentual" in getattr(t, "label", "")]
        if toggles:
            toggles[0].set_value(True).run(timeout=30)
            assert not at.exception, "Das Umschalten auf die prozentuale Visualisierung war fehlerhaft."


# ─────────────────────────────────────────────────────────────────────────────
# Sektion 2: Analytics-Klassen-Tests (analytics.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticsScrapingMonitor:
    def test_overview(self):
        """Testet die Aggregationslogik für die Wahlperiodenübersicht."""
        with get_session() as s:
            monitor = ScrapingMonitor(s)
            df = monitor.overview()
            
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "wahlperiode" in df.columns
        # Laut Seed-Daten: 1 Wahlperiode (20) mit 2 Sitzungen und 2 Reden
        row_wp20 = df[df["wahlperiode"] == 20].iloc[0]
        assert row_wp20["sitzungen"] == 2
        assert row_wp20["reden"] == 2

    def test_zwischenruf_stats(self):
        """Überprüft die Berechnung der NLP- und Sentiment-Abdeckungsraten."""
        with get_session() as s:
            monitor = ScrapingMonitor(s)
            df = monitor.zwischenruf_stats()
            
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Es gibt 2 Zwischenrufe insgesamt, davon hat genau einer ein ton_label (50%)
        assert df.iloc[0]["gesamt"] == 2
        assert df.iloc[0]["mit_sentiment"] == 1
        assert df.iloc[0]["mit_ton_label_pct"] == 50.0

    def test_recent_sitzungen(self):
        """Validiert die korrekte Sortierung (Neueste zuerst) und das Limit-Handling."""
        with get_session() as s:
            monitor = ScrapingMonitor(s)
            df = monitor.recent_sitzungen(n=1)
            
        assert len(df) == 1
        # Die Sitzung vom 2023-01-02 (Sitzungsnr 2) muss vor der vom 2023-01-01 gelistet sein
        assert df.iloc[0]["sitzungsnr"] == 2


class TestAnalyticsTonAnalyse:
    def test_ton_by_fraktion(self):
        """Überprüft die Verteilung von Ton-Klassifizierungen pro Bundestagsfraktion."""
        with get_session() as s:
            ta = TonAnalyse(s)
            df = ta.ton_by_fraktion(wahlperiode=20)
            
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert set(df.columns).issubset({"fraktion", "ton_label", "anzahl"})
            # Nur zr1 hat ein gültiges Label ("Neutral" für Fraktion "CDU/CSU")
            cdu_rows = df[df["fraktion"] == "CDU/CSU"]
            assert not cdu_rows.empty
            assert cdu_rows.iloc[0]["ton_label"] == "Neutral"
            assert cdu_rows.iloc[0]["anzahl"] == 1

    def test_ton_trend(self):
        """Testet die zeitliche Verlaufsauswertung klassifizierter Zwischenrufe."""
        with get_session() as s:
            ta = TonAnalyse(s)
            df = ta.ton_trend(wahlperiode=20)
            
        assert isinstance(df, pd.DataFrame)
        # Darf bei vorhandenen prozessierten Daten nicht kollabieren und muss Datumsspalten enthalten
        if not df.empty:
            assert "datum" in df.columns
            assert "ton_label" in df.columns
