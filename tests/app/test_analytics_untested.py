"""
tests/test_analytics_untested.py
Unit-Tests für ScrapingMonitor und TonAnalyse.
"""
import pytest
from datetime import date
from src.database import get_session, init_db, drop_db
from src.models import Rede, Redner, Sitzung, Zwischenruf
from src.analytics import ScrapingMonitor, TonAnalyse

@pytest.fixture(autouse=True)
def setup_db(monkeypatch):
    """Setzt eine In-Memory-DB auf."""
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", "sqlite:///:memory:")
    import src.database as db_module
    if hasattr(db_module, "reset_db_state"):
        db_module.reset_db_state()
    init_db()
    yield
    drop_db()

def _seed_data(session):
    sz1 = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2023, 1, 1), gesamtwortzahl=100, titel="1. Sitzung")
    sz2 = Sitzung(wahlperiode=20, sitzungsnr=2, datum=date(2023, 1, 2), gesamtwortzahl=150, titel="2. Sitzung")
    r1 = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
    
    rede1 = Rede(sitzung=sz1, redner=r1, text="Rede 1", wortanzahl=50)
    rede2 = Rede(sitzung=sz2, redner=r1, text="Rede 2", wortanzahl=75)
    
    # Zwischenruf 1: Komplett mit NLP-Daten
    zr1 = Zwischenruf(rede=rede1, text="Gut!", fraktion="CDU/CSU", ton_label="Neutral", sentiment_score=0.5, adressaten='["SPD"]')
    # Zwischenruf 2: Ohne NLP-Daten
    zr2 = Zwischenruf(rede=rede2, text="Schlecht!", fraktion="AfD")
    
    session.add_all([sz1, sz2, r1, rede1, rede2, zr1, zr2])
    session.commit()

class TestScrapingMonitor:
    def test_overview(self):
        with get_session() as s:
            _seed_data(s)
            monitor = ScrapingMonitor(s)
            df = monitor.overview()
            
        assert not df.empty
        assert "wahlperiode" in df.columns
        assert df.iloc[0]["sitzungen"] == 2
        assert df.iloc[0]["reden"] == 2

    def test_zwischenruf_stats(self):
        with get_session() as s:
            _seed_data(s)
            monitor = ScrapingMonitor(s)
            df = monitor.zwischenruf_stats()
            
        assert not df.empty
        # Es gibt 2 Zwischenrufe gesamt, 1 davon hat NLP Daten (50%)
        assert df.iloc[0]["gesamt"] == 2
        assert df.iloc[0]["mit_sentiment"] == 1
        assert df.iloc[0]["mit_ton_label_pct"] == 50.0

    def test_recent_sitzungen(self):
        with get_session() as s:
            _seed_data(s)
            monitor = ScrapingMonitor(s)
            df = monitor.recent_sitzungen(n=1)
            
        assert not df.empty
        assert len(df) == 1
        # Die neueste Sitzung (2023-01-02) sollte zuerst kommen
        assert df.iloc[0]["sitzungsnr"] == 2

class TestTonAnalyse:
    def test_ton_by_fraktion(self):
        with get_session() as s:
            _seed_data(s)
            ta = TonAnalyse(s)
            df = ta.ton_by_fraktion(wahlperiode=20)
            
        assert not df.empty
        assert set(df.columns) == {"fraktion", "ton_label", "anzahl"}
        # Nur zr1 hat ein ton_label
        assert df.iloc[0]["fraktion"] == "CDU/CSU"
        assert df.iloc[0]["ton_label"] == "Neutral"
        assert df.iloc[0]["anzahl"] == 1

    def test_ton_trend(self):
        with get_session() as s:
            _seed_data(s)
            ta = TonAnalyse(s)
            df = ta.ton_trend(wahlperiode=20)
            
        assert not df.empty
        assert "datum" in df.columns
        assert len(df) == 1  # Nur der 2023-01-01 hat einen getaggten Zwischenruf
        assert df.iloc[0]["ton_label"] == "Neutral"

    def test_ton_by_fraktion_canonicalise(self):
        with get_session() as s:
            sz = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2023, 1, 1), gesamtwortzahl=100)
            r = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
            rede = Rede(sitzung=sz, redner=r, text="Rede", wortanzahl=50)
            # Seed same faction with different spellings/casings
            z1 = Zwischenruf(rede=rede, text="Gut!", fraktion="spd", ton_label="Neutral", sentiment_score=0.5)
            z2 = Zwischenruf(rede=rede, text="Hä?", fraktion="SPD", ton_label="Neutral", sentiment_score=0.4)
            s.add_all([sz, r, rede, z1, z2])
            s.commit()

            ta = TonAnalyse(s)
            df = ta.ton_by_fraktion(wahlperiode=20)

        assert not df.empty
        # Should be aggregated into a single row for "SPD"
        spd_rows = df[df["fraktion"] == "SPD"]
        assert len(spd_rows) == 1
        assert spd_rows.iloc[0]["anzahl"] == 2


class TestKategorieAnalyseCanonicalisation:
    def test_kategorie_by_fraktion_canonicalise(self):
        from src.analytics import KategorieAnalyse
        with get_session() as s:
            sz = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2023, 1, 1), gesamtwortzahl=100)
            r = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
            rede = Rede(sitzung=sz, redner=r, text="Rede", wortanzahl=50)
            # PDS maps to "Die Linke", "die linke" maps to "Die Linke"
            z1 = Zwischenruf(rede=rede, text="Lachen", fraktion="die linke", kategorie="Lachen")
            z2 = Zwischenruf(rede=rede, text="Lachen2", fraktion="PDS", kategorie="Lachen")
            s.add_all([sz, r, rede, z1, z2])
            s.commit()

            ka = KategorieAnalyse(s)
            df = ka.kategorie_by_fraktion(wahlperiode=20, mode="given")

        assert not df.empty
        # Should be aggregated into a single row for "Die Linke"
        linke_rows = df[df["fraktion"] == "Die Linke"]
        assert len(linke_rows) == 1
        assert linke_rows.iloc[0]["anzahl"] == 2

    def test_beifall_widerspruch_ratio_canonicalise(self):
        from src.analytics import KategorieAnalyse
        with get_session() as s:
            sz = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2023, 1, 1), gesamtwortzahl=100)
            # Lowercase and uppercase speaker factions
            r1 = Redner(vorname="Olaf", nachname="Scholz", fraktion="cdu/csu")
            r2 = Redner(vorname="Angela", nachname="Merkel", fraktion="CDU/CSU")
            rede1 = Rede(sitzung=sz, redner=r1, text="Rede 1", wortanzahl=50)
            rede2 = Rede(sitzung=sz, redner=r2, text="Rede 2", wortanzahl=50)
            # Lowercase and uppercase interjecting factions
            z1 = Zwischenruf(rede=rede1, text="Beifall", fraktion="spd", kategorie="Beifall")
            z2 = Zwischenruf(rede=rede2, text="Beifall", fraktion="SPD", kategorie="Beifall")
            s.add_all([sz, r1, r2, rede1, rede2, z1, z2])
            s.commit()

            ka = KategorieAnalyse(s)
            df = ka.beifall_widerspruch_ratio(wahlperiode=20)

        assert not df.empty
        # Should be exactly one pair row for ("SPD", "CDU/CSU")
        spd_cdu = df[(df["interruptor_fraktion"] == "SPD") & (df["sprecher_fraktion"] == "CDU/CSU")]
        assert len(spd_cdu) == 1
        assert spd_cdu.iloc[0]["beifall"] == 2
        assert spd_cdu.iloc[0]["civility_index"] == 2.0