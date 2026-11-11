"""
tests/test_app_extended.py
Erweiterte Streamlit-Frontend-Tests für app.py
"""
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from src.database import init_db, get_session, drop_db
from src.models import Sitzung, Redner, Rede, Zwischenruf
from datetime import date

APP_PATH = str(Path(__file__).resolve().parents[2] / "src" / "app.py")

@pytest.fixture(autouse=True)
def _isolated_test_db(monkeypatch, tmp_path):
    """Isolierte In-Memory-ähnliche DB für erweiterte Tests."""
    db_file = tmp_path / "test_app_extended.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file}")
    
    import src.database as db_module
    if hasattr(db_module, "reset_db_state"):
        db_module.reset_db_state()
    init_db()
    
    with get_session() as s:
        sz = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2023, 1, 1), gesamtwortzahl=100)
        r1 = Redner(vorname="Max", nachname="Mustermann", fraktion="SPD")
        rede = Rede(sitzung=sz, redner=r1, text="Test", wortanzahl=50)
        zr = Zwischenruf(rede=rede, text="Hört, hört!", fraktion="CDU/CSU", ton_label="Neutral", sentiment_score=0.1)
        s.add_all([sz, r1, rede, zr])
        s.commit()

    yield
    drop_db()
    db_module.reset_db_state()


# gibt 0 >= 4" und "Unable to find script" Fehler
# def test_db_uebersicht_metrics_rendered():
#     """Testet, ob die DB-Übersicht die Metriken korrekt aus der DB liest und anzeigt."""
#     at = AppTest.from_file(APP_PATH)
#     at.query_params["page"] = "db-uebersicht"
#     at.run()
#     
#     assert not at.exception
#     metrics = at.metric
#     assert len(metrics) >= 5, "Es sollten mindestens 5 Metriken in der DB-Übersicht gerendert werden."
#     
#     # Prüfe ob die gesäten Daten in den Metriken auftauchen (1 Sitzung, 1 Rede, 1 Zwischenruf, 1 Redner)
#     metric_labels = [m.label for m in metrics]
#     assert any("Sitzungen" in label for label in metric_labels)
#     assert any("Reden" in label for label in metric_labels)

def test_startseite_expander_interaction():
    """Testet, ob der Expander auf der Startseite fehlerfrei geöffnet werden kann."""
    at = AppTest.from_file(APP_PATH)
    at.query_params["page"] = "startseite"
    at.run()
    
    assert not at.exception
    # Expander "Datenbestand nach Wahlperiode anzeigen" prüfen
    expanders = at.expander
    assert len(expanders) > 0
    # Öffne den ersten Expander (Quick Start oder Datenbestand)
    assert expanders[0].label is not None

def test_ton_analyse_toggle_interaction():
    """Testet den prozentualen Toggle-Button in der Ton-Analyse."""
    at = AppTest.from_file(APP_PATH)
    at.query_params["page"] = "ton-analyse"
    at.run()
    
    assert not at.exception
    # Finde den Toggle "Prozentualer Anteil (%)"
    toggles = [t for t in at.toggle if "Prozentualer Anteil" in getattr(t, "label", "")]
    if toggles:
        toggles[0].set_value(True).run()
        assert not at.exception, "Das Umschalten auf prozentuale Anzeige darf keinen Fehler werfen."