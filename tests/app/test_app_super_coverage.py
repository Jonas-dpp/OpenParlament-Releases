"""
tests/test_app_super_coverage.py
Maximaler isolierter Coverage-Test für app.py
"""
import pytest
from pathlib import Path
from datetime import date
from streamlit.testing.v1 import AppTest

from src.database import init_db, get_session, drop_db
from src.models import Sitzung, Redner, Rede, Zwischenruf

ROOT = str(Path(__file__).resolve().parents[2]).replace('\\', '/')

@pytest.fixture(autouse=True)
def coverage_db(monkeypatch, tmp_path):
    """Baut eine frische In-Memory-Datenbank für JEDEN einzelnen Seiten-Test auf."""
    db_file = tmp_path / "test_app_super.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file}")
    
    import src.database as db_module
    if hasattr(db_module, "reset_db_state"):
        db_module.reset_db_state()
    init_db()
    
    with get_session() as s:
        sz = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2023, 1, 1), gesamtwortzahl=1000)
        r1 = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        r2 = Redner(vorname="Friedrich", nachname="Merz", fraktion="CDU/CSU")
        
        # FIX: TF-IDF benötigt mehrere Dokumente, sonst crasht sklearn (max_df < min_df)
        # Dadurch ist sichergestellt, dass prozentuale Filter (max_df) niemals unter absolute Filter (min_df) fallen.
        reden = []
        for i in range(22):  # 22 Reden, damit wir min_df=2 und max_df=0.5 problemlos testen können
            # Wir variieren den Text leicht, damit der TF-IDF-Vektorisierer echte Muster findet
            if i % 11 == 0:
                wort = "Wirtschaft"
            elif (i + 1) % 11 == 0:
                wort = "Klima"
            else:
                wort = "Energie"


            text = f"Dies ist die offizielle Testrede Nummer {i}. Wir müssen dringend über {wort} sprechen und in die Zukunft unseres Landes investieren."
            redner = r1 if i % 2 == 0 else r2
            reden.append(Rede(sitzung=sz, redner=redner, text=text, wortanzahl=20))
        
        zr = Zwischenruf(rede=reden[0], text="Hört, hört!", fraktion="SPD", ton_label="Neutral", sentiment_score=0.1)
        
        s.add_all([sz, r1, r2] + reden + [zr])
        s.commit()
        
    yield
    drop_db()
    if hasattr(db_module, "reset_db_state"):
        db_module.reset_db_state()

# Liste aller Render-Funktionen aus app.py, die wir testen wollen
RENDER_FUNCS = [
    "app.render_startseite()",
    "app.render_db_uebersicht()",
    "app.render_scraping_monitor()",
    "app.render_wahlperioden_vergleich()",
    "app.render_themen_trend(20)",
    "app.render_top_analyse(20)",
    "app.render_framing_analyse(20)",
    "app.render_aggressions_radar(20, None, None, None)",
    "app.render_interaktions_netzwerk(20, None, None)",
    "app.render_ton_analyse(20, None, None)",
    "app.render_adressaten_analyse(20, None, None)",
    "app.render_reaktions_analyse(20, None, None)",
    "app.render_redezeit_analyse(20, None, None)",
    "app.render_debattenklima(20, None, None)",
    "app.render_redner_profil(20, None, None)",
    "app.render_redner_vergleich(20, None, None)",
    "app.render_fraktions_dynamik(20, None, None)",
    "app.render_lexical_analyse(20)",
    "app.render_architecture()",
]

@pytest.mark.parametrize("render_call", RENDER_FUNCS)
def test_app_pages_isolated(render_call):
    """
    Testet jede Seite isoliert in einem eigenen Streamlit-Kontext.
    """
    test_script = f"""
import sys
sys.path.insert(0, '{ROOT}')
import streamlit as st

# --- MAGIC TRICK: st.navigation blockieren ---
# Wir überschreiben st.navigation mit einem Dummy. 
# Dadurch macht das pg.run() am Ende von app.py gar nichts, 
# und wir vermeiden Duplicate Element Keys!
class DummyNav:
    def run(self): pass
st.navigation = lambda *args, **kwargs: DummyNav()
# ---------------------------------------------

import src.app as app

# Führe nun exakt EINE einzige Render-Funktion aus
{render_call}
"""
    at = AppTest.from_string(test_script)
    at.run(timeout=30)
    
    assert not at.exception, f"Crash in der Funktion {render_call}:\n{at.exception}"
    
    # Der Metrik-Test darf nur laufen, wenn wir auch wirklich auf der DB-Übersicht sind
    if "render_db_uebersicht" in render_call:
        assert len(at.metric) >= 4, "Metriken wurden auf der DB-Übersicht nicht gefunden."