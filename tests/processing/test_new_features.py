"""
OpenParlament - New Features Unit Tests
Copyright (C) 2026 Jules

Unit tests for Speech Complexity, Action-Reaction-Gap, Network 2.0,
Polarization, and Model Validation.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
from sqlalchemy import select

from src.database import drop_db, get_session, init_db
from src.models import (
    Rede,
    Redner,
    Sitzung,
    Zwischenruf,
    SpeechComplexityResult,
    ReactionGapResult,
)
from src.analytics import (
    SpeechComplexityAnalysis,
    ReactionGapAnalysis,
    InteraktionsNetzwerk,
    SentimentValidation,
)


@pytest.fixture(autouse=True)
def _use_in_memory_db_new(monkeypatch):
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Database Model Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_speech_complexity_result_relationship():
    """Verify that SpeechComplexityResult is successfully persisted and related correctly."""
    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner = Redner(vorname="Max", nachname="Mustermann", fraktion="SPD")
        s.add_all([sitzung, redner])
        s.flush()

        rede = Rede(sitzung=sitzung, redner=redner, text="Das ist ein Satz.", wortanzahl=5)
        s.add(rede)
        s.flush()

        result = SpeechComplexityResult(
            rede_id=rede.rede_id,
            wahlperiode=20,
            redner_id=redner.redner_id,
            wstf_score=8.5,
            flesch_score=60.0,
            avg_sentence_length=5.0,
            subordinate_clause_ratio=0.0,
            avg_word_length=4.0,
            long_word_ratio=0.0,
            lexical_diversity=1.0,
        )
        s.add(result)

    with get_session() as s:
        saved = s.execute(select(SpeechComplexityResult)).scalars().first()
        assert saved is not None
        assert saved.id is not None
        assert saved.wstf_score == 8.5
        assert saved.flesch_score == 60.0
        # Eager load related to avoid DetachedInstanceError
        assert saved.rede.text == "Das ist ein Satz."
        assert saved.redner.nachname == "Mustermann"


def test_reaction_gap_result_relationship():
    """Verify that ReactionGapResult is successfully persisted and related correctly."""
    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner = Redner(vorname="Max", nachname="Mustermann", fraktion="SPD")
        s.add_all([sitzung, redner])
        s.flush()

        rede = Rede(sitzung=sitzung, redner=redner, text="Satz.", wortanzahl=100)
        s.add(rede)
        s.flush()

        zwr = Zwischenruf(rede_id=rede.rede_id, text="Pfui!", fraktion="AfD", sentiment_score=-0.8)
        s.add(zwr)
        s.flush()

        result = ReactionGapResult(
            rede_id=rede.rede_id,
            zwischenruf_id=zwr.ruf_id,
            topic_id="TOP 1",
            speaker_sentiment=0.0,
            reaction_sentiment=-0.8,
            gap=-0.8,
            time_to_first_interjection=15.0,
        )
        s.add(result)

    with get_session() as s:
        saved = s.execute(select(ReactionGapResult)).scalars().first()
        assert saved is not None
        assert saved.id is not None
        assert saved.gap == -0.8
        assert saved.rede.text == "Satz."
        assert saved.zwischenruf.text == "Pfui!"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Linguistic Calculations (SpeechComplexityAnalysis)
# ─────────────────────────────────────────────────────────────────────────────

def test_amstad_fre_simple_vs_complex_sentences():
    """Verify that simple sentences score higher FRE than complex ones, and other metrics."""
    with get_session() as s:
        analyzer = SpeechComplexityAnalysis(s)

    simple_text = "Das ist ein sehr einfacher Satz. Er hat kurze Worte."
    complex_text = "Obwohl das parlamentarische Verfahren verfassungsmäßig geregelt ist, führt die kontinuierliche sprachliche Simplifizierung zu einer emotionalisierten Polarisierung der gesamten Plenardebatte."

    simple_metrics = analyzer.calculate_metrics(simple_text)
    complex_metrics = analyzer.calculate_metrics(complex_text)

    # Simple text should have lower Wiener Sachtextformel score (lower required grade level) than complex text
    assert simple_metrics["wstf_score"] < complex_metrics["wstf_score"]
    assert simple_metrics["flesch_score"] > complex_metrics["flesch_score"]

    # Complex text should have larger average sentence length
    assert complex_metrics["avg_sentence_length"] > simple_metrics["avg_sentence_length"]

    # Complex text should have non-zero subordinate clause ratio due to "Obwohl"
    assert complex_metrics["subordinate_clause_ratio"] > 0.0

    # Complex text has longer words
    assert complex_metrics["avg_word_length"] > simple_metrics["avg_word_length"]
    assert complex_metrics["long_word_ratio"] > simple_metrics["long_word_ratio"]


def test_lexical_diversity_repetition_lowers_ttr():
    """Verify that repeating vocabulary lowers lexical diversity (TTR)."""
    with get_session() as s:
        analyzer = SpeechComplexityAnalysis(s)

    rich_text = "Der Hund fängt einen Vogel und spielt draußen."
    repetitive_text = "Hund Hund Hund Hund Hund Hund Hund Hund Hund Hund"

    rich_metrics = analyzer.calculate_metrics(rich_text)
    repetitive_metrics = analyzer.calculate_metrics(repetitive_text)

    assert rich_metrics["lexical_diversity"] == 1.0
    assert repetitive_metrics["lexical_diversity"] == 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Action-Reaction-Gap & Speaking Rate Heuristic (ReactionGapAnalysis)
# ─────────────────────────────────────────────────────────────────────────────

def test_reaction_gap_chronological_ordering_and_speaking_rate():
    """Verify that first interjection is detected correctly and speaking rate heuristic applies."""
    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        s.add_all([sitzung, redner])
        s.flush()

        rede = Rede(
            sitzung_id=sitzung.sitzungs_id,
            redner_id=redner.redner_id,
            text="Wir halten zusammen.",
            tagesordnungspunkt="TOP 1",
            wortanzahl=216,  # 216 words
            sentiment_score=0.4,
        )
        s.add(rede)
        s.flush()

        # Insert 3 interjections in order
        z1 = Zwischenruf(rede_id=rede.rede_id, text="Beifall!", sentiment_score=0.8, kategorie="Beifall")
        z2 = Zwischenruf(rede_id=rede.rede_id, text="Unruhe!", sentiment_score=-0.5, kategorie="Unruhe")
        z3 = Zwischenruf(rede_id=rede.rede_id, text="Pfui!", sentiment_score=-0.9, kategorie="Widerspruch")
        s.add_all([z1, z2, z3])
        s.flush()

        z1_id = z1.ruf_id

        gap_analyzer = ReactionGapAnalysis(s)
        df_gap = gap_analyzer.run(20)

    assert len(df_gap) == 1
    row = df_gap.iloc[0]

    # Chronological first is z1 (Beifall)
    assert row["zwischenruf_id"] == z1_id
    assert row["reaction_sentiment"] == 0.8
    assert row["speaker_sentiment"] == 0.4
    assert row["gap"] == pytest.approx(0.4)

    # Time to first interjection estimate:
    # Speaking rate is 2.167 words/sec
    # Total duration = 216 / 2.167 ~ 100 seconds
    # num_z = 3, so first occurs at total_duration / (3 + 1) = 100 / 4 = 25 seconds
    assert row["time_to_first_interjection"] == pytest.approx(25.0, rel=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Interactive Faction Network (NetworkX 2.0) & Polarization Index
# ─────────────────────────────────────────────────────────────────────────────

def test_interactive_network_metrics_and_polarization():
    """Verify directed graph construction, edge attributes, metrics, and polarization."""
    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner_spd = Redner(vorname="A", nachname="SPD-R", fraktion="SPD")
        redner_cdu = Redner(vorname="B", nachname="CDU-R", fraktion="CDU/CSU")
        s.add_all([sitzung, redner_spd, redner_cdu])
        s.flush()

        rede_spd = Rede(sitzung_id=sitzung.sitzungs_id, redner_id=redner_spd.redner_id, text="Text", wortanzahl=5)
        rede_cdu = Rede(sitzung_id=sitzung.sitzungs_id, redner_id=redner_cdu.redner_id, text="Text", wortanzahl=5)
        s.add_all([rede_spd, rede_cdu])
        s.flush()

        # SPD interrupts CDU: 1 Beifall (support) and 1 Widerspruch (attack)
        z1 = Zwischenruf(rede_id=rede_cdu.rede_id, text="Beifall!", fraktion="SPD", sentiment_score=0.8, kategorie="Beifall")
        z2 = Zwischenruf(rede_id=rede_cdu.rede_id, text="Nein!", fraktion="SPD", sentiment_score=-0.8, kategorie="Widerspruch")
        # CDU interrupts SPD: 2 Widersprüche (attacks)
        z3 = Zwischenruf(rede_id=rede_spd.rede_id, text="Falsch!", fraktion="CDU/CSU", sentiment_score=-0.7, kategorie="Widerspruch")
        z4 = Zwischenruf(rede_id=rede_spd.rede_id, text="Absurd!", fraktion="CDU/CSU", sentiment_score=-0.9, kategorie="Widerspruch")
        s.add_all([z1, z2, z3, z4])
        s.flush()

        net = InteraktionsNetzwerk(s)
        G = net.build_networkx_graph_2_0(20)
        metrics = net.network_metrics(G)
        pol = net.polarization_index(20)

    # Both SPD and CDU/CSU must be nodes
    assert "SPD" in G.nodes
    assert "CDU/CSU" in G.nodes

    # Outgoing edge SPD -> CDU/CSU
    edge_spd_cdu = G["SPD"]["CDU/CSU"]
    assert edge_spd_cdu["count"] == 2
    assert edge_spd_cdu["mean_sentiment"] == pytest.approx(0.0) # (+0.8 -0.8)/2
    assert edge_spd_cdu["negative_count"] == 1
    assert edge_spd_cdu["positive_count"] == 1
    assert edge_spd_cdu["applause_count"] == 1
    assert edge_spd_cdu["attack_count"] == 1

    # Outgoing edge CDU/CSU -> SPD
    edge_cdu_spd = G["CDU/CSU"]["SPD"]
    assert edge_cdu_spd["count"] == 2
    assert edge_cdu_spd["mean_sentiment"] == pytest.approx(-0.8)
    assert edge_cdu_spd["negative_count"] == 2
    assert edge_cdu_spd["attack_count"] == 2

    # Global metrics
    assert metrics["density"] > 0.0
    assert metrics["reciprocity"] == 1.0 # Mutual edges
    assert "SPD" in metrics["degree_centrality"]
    assert "SPD" in metrics["eigenvector_centrality"]

    # Polarization Index
    # For CDU/CSU: negative external is 2 (both z3, z4 are negative/attacks), total external is 2. Index = 2/2 = 1.0
    # For SPD: negative external is 1 (z2), total external is 2. Index = 1/2 = 0.5
    cdu_polar = pol[pol["fraktion"] == "CDU/CSU"].iloc[0]
    spd_polar = pol[pol["fraktion"] == "SPD"].iloc[0]
    assert cdu_polar["polarization_index"] == 1.0
    assert spd_polar["polarization_index"] == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sentiment Model Validation
# ─────────────────────────────────────────────────────────────────────────────

@patch("src.nlp.SentimentEngine._neural_score_batch")
def test_sentiment_model_validation(mock_neural):
    """Verify that SentimentValidation computes all metrics and outputs correct shapes."""
    mock_neural.side_effect = lambda texts: [0.0] * len(texts)
    with get_session() as s:
        val = SentimentValidation(s)
        res = val.evaluate()

    assert res["accuracy"] >= 0.0 and res["accuracy"] <= 1.0
    assert res["macro_precision"] >= 0.0 and res["macro_precision"] <= 1.0
    assert res["macro_recall"] >= 0.0 and res["macro_recall"] <= 1.0
    assert res["macro_f1"] >= 0.0 and res["macro_f1"] <= 1.0

    # 3x3 Confusion Matrix shape check
    cm = res["confusion_matrix"]
    assert cm.shape == (3, 3)
    assert list(cm.index) == ["Human: positive", "Human: neutral", "Human: negative"]
    assert list(cm.columns) == ["Model: positive", "Model: neutral", "Model: negative"]

    # Total samples validated should be exactly 100
    assert len(res["samples"]) == 100


@patch("src.nlp.SentimentEngine._neural_score_batch")
def test_sentiment_model_validation_empty_samples(mock_neural):
    """Verify that SentimentValidation handles empty validation samples without raising ZeroDivisionError."""
    mock_neural.side_effect = lambda texts: []
    with get_session() as s:
        val = SentimentValidation(s)
        # Override with empty list
        val.validation_samples = []
        res = val.evaluate()

    assert res["accuracy"] == 0.0
    assert res["macro_precision"] == 0.0
    assert res["macro_recall"] == 0.0
    assert res["macro_f1"] == 0.0
    assert len(res["samples"]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Streamlit Page Smoke Tests
# ─────────────────────────────────────────────────────────────────────────────

@patch("src.app.get_session")
@patch("src.app.st")
def test_page_sprachkomplexitaet_smoke(mock_st, mock_get_session):
    """Ensure _page_sprachkomplexitaet renders without raising exceptions."""
    from src.app import _page_sprachkomplexitaet

    # Configure columns mocking to return mock containers
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

    # Mock row return representing a SpeechComplexityResult
    mock_r = MagicMock()
    mock_r.rede_id = 1
    mock_r.wstf_score = 8.5
    mock_r.flesch_score = 60.0
    mock_r.avg_sentence_length = 5.0
    mock_r.subordinate_clause_ratio = 0.0
    mock_r.avg_word_length = 4.0
    mock_r.long_word_ratio = 0.0
    mock_r.lexical_diversity = 1.0

    mock_session = MagicMock()
    # Executing the select on SpeechComplexityResult yields this mock row
    mock_session.execute.return_value.fetchall.side_effect = [
        [(mock_r, "Mustermann", "Max", "SPD", "TOP 1", "Satz.")], # SpeechComplexityResult query
        [(1, 0.0, 0.0, 0.0, 0, 0, 0)] # Zwischenruf stats query
    ]
    mock_get_session.return_value.__enter__.return_value = mock_session

    _page_sprachkomplexitaet()

    # st.plotly_chart or st.metric should have been called
    assert mock_st.plotly_chart.called


@patch("src.app.get_session")
@patch("src.app.st")
def test_page_reaction_gap_smoke(mock_st, mock_get_session):
    """Ensure _page_reaction_gap renders without raising exceptions."""
    from src.app import _page_reaction_gap

    # Configure columns mocking to return mock containers
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

    mock_r = MagicMock()
    mock_r.rede_id = 1
    mock_r.topic_id = "TOP 1"
    mock_r.speaker_sentiment = 0.4
    mock_r.reaction_sentiment = -0.5
    mock_r.gap = -0.9
    mock_r.time_to_first_interjection = 12.0

    mock_session = MagicMock()
    mock_session.execute.return_value.fetchall.return_value = [
        (mock_r, "TOP 1", "Max", "Mustermann", "SPD", "AfD")
    ]
    mock_get_session.return_value.__enter__.return_value = mock_session

    _page_reaction_gap()
    assert mock_st.plotly_chart.called


@patch("src.app.get_session")
@patch("src.app.st")
def test_page_polarisation_smoke(mock_st, mock_get_session):
    """Ensure _page_polarisation renders without raising exceptions."""
    from src.app import _page_polarisation

    # Configure columns mocking to return mock containers
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

    mock_session = MagicMock()
    mock_session.execute.return_value.fetchall.return_value = [
        ("SPD", "CDU/CSU", -0.8, "Widerspruch")
    ]
    mock_get_session.return_value.__enter__.return_value = mock_session

    _page_polarisation()
    assert mock_st.plotly_chart.called


@patch("src.nlp.SentimentEngine._neural_score_batch")
@patch("src.app.get_session")
@patch("src.app.st")
def test_page_modell_validierung_smoke(mock_st, mock_get_session, mock_neural):
    """Ensure _page_modell_validierung renders without raising exceptions."""
    mock_neural.side_effect = lambda texts: [0.0] * len(texts)
    from src.app import _page_modell_validierung

    # Configure columns mocking to return mock containers
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    _page_modell_validierung()
    assert mock_st.plotly_chart.called

    
# ─────────────────────────────────────────────────────────────────────────────
# 7. Precompute Complexity CLI Script Test
# ─────────────────────────────────────────────────────────────────────────────

def test_precompute_complexity_script():
    """Verify that precompute_complexity.py successfully populates metrics and deletes old records."""
    from scripts.precompute_complexity import precompute_for_wp

    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        s.add_all([sitzung, redner])
        s.flush()

        rede = Rede(
            sitzung_id=sitzung.sitzungs_id,
            redner_id=redner.redner_id,
            text="Wir halten zusammen. Das ist sehr wichtig für uns.",
            tagesordnungspunkt="TOP 1",
            wortanzahl=10,
            sentiment_score=0.4,
        )
        s.add(rede)
        s.flush()

        # At least one interjection so that ReactionGap gets populated
        zwr = Zwischenruf(
            rede_id=rede.rede_id,
            text="Pfui!",
            fraktion="AfD",
            sentiment_score=-0.8,
            kategorie="Widerspruch",
        )
        s.add(zwr)
        s.flush()

    # Run precomputation
    with get_session() as s:
        precompute_for_wp(s, wp=20)

    # Verify rows were populated
    with get_session() as s:
        sc_rows = s.execute(select(SpeechComplexityResult)).scalars().all()
        rg_rows = s.execute(select(ReactionGapResult)).scalars().all()

        assert len(sc_rows) == 1
        assert sc_rows[0].wstf_score is not None
        assert sc_rows[0].flesch_score > 0
        assert sc_rows[0].avg_sentence_length == 4.5

        assert len(rg_rows) == 1
        assert rg_rows[0].gap == pytest.approx(-1.2)

    # Run again to verify that existing results are cleaned/re-inserted and no duplicate constraints are hit
    with get_session() as s:
        precompute_for_wp(s, wp=20)

    with get_session() as s:
        sc_rows2 = s.execute(select(SpeechComplexityResult)).scalars().all()
        rg_rows2 = s.execute(select(ReactionGapResult)).scalars().all()

        assert len(sc_rows2) == 1
        assert len(rg_rows2) == 1
        
        
# ─────────────────────────────────────────────────────────────────────────────
# 8. Verify StylometryAnalyzer.predict_faction
# ─────────────────────────────────────────────────────────────────────────────

def test_stylometry_analyzer_predict_faction_mocked():
    """Verify StylometryAnalyzer.predict_faction returns dictionary when model is present."""
    from src.analytics import StylometryAnalyzer
    import numpy as np

    with get_session() as s:
        analyzer = StylometryAnalyzer(s)

        # Mock the model
        mock_model = MagicMock()
        mock_model.classes_ = ["SPD", "CDU/CSU"]
        mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
        mock_model.feature_names_in_ = [
            "avg_sentence_length", "passive_ratio", "type_token_ratio", "fill_word_ratio",
            "punct_question", "punct_exclamation", "punct_period", "punct_comma"
        ]
        analyzer.model = mock_model

        pred_dict = analyzer.predict_faction("Das ist ein toller Text mit genügend Zeichen, um analysiert zu werden.")
        assert isinstance(pred_dict, dict)
        assert "SPD" in pred_dict
        assert pred_dict["SPD"] == 0.6
        assert pred_dict["CDU/CSU"] == 0.4

        # Convert to DataFrame to ensure our fix in app.py is safe and correct
        df_pred = pd.DataFrame([
            {"fraktion": k, "probability": v}
            for k, v in pred_dict.items()
        ])
        assert len(df_pred) == 2
        assert list(df_pred.columns) == ["fraktion", "probability"]


@patch("src.app.get_session")
@patch("src.app.st")
def test_page_stylometrie_smoke(mock_st, mock_get_session):
    """Ensure _page_stylometrie renders without raising exceptions."""
    from src.app import _page_stylometrie
    import numpy as np

    # Configure columns mocking/tabs mocking
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    mock_tab1 = MagicMock()
    mock_tab2 = MagicMock()
    mock_st.tabs.return_value = [mock_tab1, mock_tab2]
    mock_st.button.return_value = True  # Pretend Analyse button was clicked
    mock_st.text_area.return_value = "Wir müssen entschlossen voranschreiten, um die Herausforderungen unserer Zeit zu bewältigen..."

    mock_session = MagicMock()
    # Mock some data for get_faction_dna query
    mock_session.execute.return_value.fetchall.return_value = [
        ("SPD", 15.0, 0.1, 0.4, 0.05),
        ("CDU/CSU", 18.0, 0.12, 0.35, 0.04)
    ]
    mock_get_session.return_value.__enter__.return_value = mock_session

    # Mock the StylometryAnalyzer prediction part to avoid file loading
    with patch("joblib.load") as mock_load:
        mock_model = MagicMock()
        mock_model.classes_ = ["SPD", "CDU/CSU"]
        mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
        mock_model.feature_names_in_ = [
            "avg_sentence_length", "passive_ratio", "type_token_ratio", "fill_word_ratio",
            "punct_question", "punct_exclamation", "punct_period", "punct_comma"
        ]
        mock_load.return_value = mock_model

        _page_stylometrie()

    assert mock_st.plotly_chart.called


def test_precompute_h3_normalization_and_data():
    """Verify that normalize_faction_h3 correctly standardizes name variants."""
    from scripts.precompute_h3 import normalize_faction_h3, precompute_data

    # Test normalization function
    assert normalize_faction_h3("CDU") == "CDU/CSU"
    assert normalize_faction_h3("CSU") == "CDU/CSU"
    assert normalize_faction_h3("SPD") == "SPD"
    assert normalize_faction_h3("BÜNDNIS\xa090/DIE GRÜNEN") == "BÜNDNIS 90/DIE GRÜNEN"
    assert normalize_faction_h3("LINKE") == "Die Linke"
    assert normalize_faction_h3("UNKNOWN_PARTY") == "UNKNOWN_PARTY"

    # Seed the test DB and test precomputing query
    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        s.add_all([sitzung, redner])
        s.flush()

        rede = Rede(
            sitzung_id=sitzung.sitzungs_id,
            redner_id=redner.redner_id,
            text="Wir halten zusammen.",
            tagesordnungspunkt="TOP 1",
            wortanzahl=10,
            sentiment_score=0.4,
        )
        s.add(rede)
        s.flush()

        # Add interjection
        zwr = Zwischenruf(
            rede_id=rede.rede_id,
            text="Beifall!",
            fraktion="CDU",
            sentiment_score=0.8,
            kategorie="Beifall",
        )
        s.add(zwr)
        s.flush()

    # Precompute data
    df_res = precompute_data([20])
    assert len(df_res) == 1
    assert df_res.iloc[0]["sender"] == "CDU/CSU"
    assert df_res.iloc[0]["speaker"] == "SPD"
    assert df_res.iloc[0]["sentiment"] == 0.8

    # ----------------------


def test_precompute_h3_telemetry(caplog):
    """Verify that precompute_data emits the expected progress logs with ETA using dynamic throttling."""
    import logging
    from scripts.precompute_h3 import precompute_data

    # Seed the test DB with enough records to trigger logs
    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 11, 17), gesamtwortzahl=100)
        redner = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        s.add_all([sitzung, redner])
        s.flush()

        rede = Rede(
            sitzung_id=sitzung.sitzungs_id,
            redner_id=redner.redner_id,
            text="Wir halten zusammen.",
            tagesordnungspunkt="TOP 1",
            wortanzahl=10,
            sentiment_score=0.4,
        )
        s.add(rede)
        s.flush()

        # Add 10 interjections so we have a total of 10 rows
        for i in range(10):
            zwr = Zwischenruf(
                rede_id=rede.rede_id,
                text=f"Beifall {i}!",
                fraktion="CDU",
                sentiment_score=0.8,
                kategorie="Beifall",
            )
            s.add(zwr)
        s.flush()

    with caplog.at_level(logging.INFO, logger="precomp_h3"):
        df_res = precompute_data([20])

    assert len(df_res) == 10

    # Retrieve all H3DBPrepare messages
    h3_logs = [record.message for record in caplog.records if "H3DBPrepare:" in record.message]
    assert len(h3_logs) > 0
    # The first log should be H3DBPrepare: 1/10
    assert "H3DBPrepare: 1/10" in h3_logs[0]
    # The last log should be H3DBPrepare: 10/10
    assert "H3DBPrepare: 10/10" in h3_logs[-1]

    # ----------------------


@patch("src.app.get_session")
@patch("src.app.st")
def test_page_h3_asymmetric_attacks_smoke(mock_st, mock_get_session):
    """Ensure render_h3_asymmetric_attacks renders without exceptions."""
    from src.app import render_h3_asymmetric_attacks

    # Mock st.columns
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

    # Mock st.slider to return real float values
    mock_st.slider.side_effect = [-0.3, 0.3]

    # Mock the loaded DataFrame
    mock_df = pd.DataFrame([
        {"wp": 20, "sender": "CDU/CSU", "speaker": "SPD", "sentiment": -0.8},
        {"wp": 20, "sender": "SPD", "speaker": "SPD", "sentiment": 0.8},
    ])

    with patch("src.app.load_h3_data", return_value=mock_df):
        render_h3_asymmetric_attacks(20)

    assert mock_st.plotly_chart.called
@patch("src.app.get_session")
@patch("src.app.st")
def test_page_reaction_gap_new_metrics(mock_st, mock_get_session):
    """Verify that our new statistical metrics are rendered correctly in _page_reaction_gap."""
    from src.app import _page_reaction_gap

    # Mock st.columns to return MagicMocks that capture .metric calls
    cols = [MagicMock() for _ in range(10)]
    mock_st.columns.side_effect = lambda n: cols[:n]

    mock_r = MagicMock()
    mock_r.rede_id = 1
    mock_r.topic_id = "TOP 1"
    mock_r.speaker_sentiment = 0.5
    mock_r.reaction_sentiment = -0.5
    mock_r.gap = -1.0
    mock_r.time_to_first_interjection = 10.0

    mock_session = MagicMock()
    mock_session.execute.return_value.fetchall.return_value = [
        (mock_r, "TOP 1", "Max", "Mustermann", "SPD", "AfD"),
        (mock_r, "TOP 1", "Erika", "Musterfrau", "CDU/CSU", "SPD")
    ]
    mock_get_session.return_value.__enter__.return_value = mock_session

    _page_reaction_gap()

    # Verify that st.metric or column.metric was called with our new metrics
    metric_calls = []
    for col in cols:
        for call in col.metric.call_args_list:
            metric_calls.append(call[0][0])  # Get the label argument

    assert any("norm_mean_1" in label for label in metric_calls)
    assert any("norm_mean_2" in label for label in metric_calls)
    assert any("mean_diff" in label for label in metric_calls)
    assert any("R²" in label for label in metric_calls)
    assert any("p_value" in label for label in metric_calls)


@patch("src.app.get_session")
@patch("src.app.st")
def test_page_hypothesen_pruefung_h1_smoke(mock_st, mock_get_session):
    """Smoke test to ensure Hypothese 1 runs successfully on the new page."""
    from src.app import _page_hypothesen_pruefung

    # Configure st.columns to return MagicMocks
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

    # Configure st.tabs to return 3 tab MagicMocks
    tab_mocks = [MagicMock(), MagicMock(), MagicMock()]
    mock_st.tabs.return_value = tab_mocks

    # Configure st.slider to return float values
    mock_st.slider.side_effect = [-0.3, 0.3]

    # Configure st.text_input to return empty strings/defaults
    mock_st.text_input.return_value = ""

    # Mock database results
    mock_data = [
        {"sentiment": 0.5, "fraktion": "SPD", "datum": "2026-01-01"},
        {"sentiment": -0.2, "fraktion": "CDU/CSU", "datum": "2026-01-01"},
        {"sentiment": 0.4, "fraktion": "GRÜNE", "datum": "2026-01-01"},
        {"sentiment": -0.5, "fraktion": "AFD", "datum": "2026-01-01"}
    ]

    mock_eval = {
        "conflict_speeches": [],
        "control_speeches": [],
        "conflict_sentiments": [],
        "control_sentiments": []
    }

    with patch("src.app._cached_h1_data", return_value=mock_data), \
         patch("src.app._cached_h2_evaluation", return_value=mock_eval), \
         patch("src.app.load_h3_data", return_value=pd.DataFrame()):
        _page_hypothesen_pruefung()

    assert mock_st.plotly_chart.called


# ─────────────────────────────────────────────────────────────────────────────
# 9. Tests for New Interjection Analyses (Reziprozität, Waffen-Mix, Dynamik)
# ─────────────────────────────────────────────────────────────────────────────

def test_reziprozitaet_matrix():
    """Verify reziprozitaet_matrix calculates given vs received interjection counts and Netto-Aggressivität."""
    from src.analytics import AggressionsIndex

    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2022, 1, 10), gesamtwortzahl=1000)
        redner_spd = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        redner_cdu = Redner(vorname="Friedrich", nachname="Merz", fraktion="CDU/CSU")
        s.add_all([sitzung, redner_spd, redner_cdu])
        s.flush()

        rede_spd = Rede(sitzung_id=sitzung.sitzungs_id, redner_id=redner_spd.redner_id, text="Rede SPD", wortanzahl=100)
        rede_cdu = Rede(sitzung_id=sitzung.sitzungs_id, redner_id=redner_cdu.redner_id, text="Rede CDU", wortanzahl=100)
        s.add_all([rede_spd, rede_cdu])
        s.flush()

        # CDU interrupts SPD speech 3 times (negative) - with author name
        z1 = Zwischenruf(rede_id=rede_spd.rede_id, fraktion="CDU/CSU", text="Friedrich Merz [CDU/CSU]: Nein!", sentiment_score=-0.8)
        z2 = Zwischenruf(rede_id=rede_spd.rede_id, fraktion="CDU/CSU", text="Friedrich Merz [CDU/CSU]: Falsch!", sentiment_score=-0.6)
        z3 = Zwischenruf(rede_id=rede_spd.rede_id, fraktion="CDU/CSU", text="Friedrich Merz [CDU/CSU]: Unruhe!", sentiment_score=-0.4)
        # SPD interrupts CDU speech 1 time (positive) - with author name
        z4 = Zwischenruf(rede_id=rede_cdu.rede_id, fraktion="SPD", text="Olaf Scholz [SPD]: Gut!", sentiment_score=0.8)
        s.add_all([z1, z2, z3, z4])
        s.flush()

        agg = AggressionsIndex(s)
        df_frak = agg.reziprozitaet_matrix(wahlperiode=20, groupby="fraktion")
        df_red = agg.reziprozitaet_matrix(wahlperiode=20, groupby="redner")

    # Assertions for Faction Grouping
    assert not df_frak.empty
    assert "entity" in df_frak.columns
    assert "ausgeteilt_count" in df_frak.columns
    assert "kassiert_count" in df_frak.columns
    assert "netto_aggressivitaet" in df_frak.columns
    assert "reziprozitaet_index" in df_frak.columns

    cdu_row = df_frak[df_frak["entity"] == "CDU/CSU"].iloc[0]
    spd_row = df_frak[df_frak["entity"] == "SPD"].iloc[0]

    assert cdu_row["ausgeteilt_count"] == 3
    assert cdu_row["kassiert_count"] == 1
    assert spd_row["ausgeteilt_count"] == 1
    assert spd_row["kassiert_count"] == 3

    # Assertions for Speaker Grouping
    assert not df_red.empty
    assert len(df_red) == 2
    merz_row = df_red[df_red["entity"] == "Friedrich Merz"].iloc[0]
    scholz_row = df_red[df_red["entity"] == "Olaf Scholz"].iloc[0]

    assert merz_row["ausgeteilt_count"] == 3
    assert merz_row["kassiert_count"] == 1
    assert scholz_row["ausgeteilt_count"] == 1
    assert scholz_row["kassiert_count"] == 3
    assert merz_row["netto_aggressivitaet"] < 0


def test_waffen_mix_analyse():
    """Verify waffen_mix_analyse computes 100% normalized distributions across categories and ton labels."""
    from src.analytics import TonAnalyse

    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2022, 1, 10), gesamtwortzahl=500)
        redner = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        s.add_all([sitzung, redner])
        s.flush()

        rede = Rede(sitzung_id=sitzung.sitzungs_id, redner_id=redner.redner_id, text="Rede", wortanzahl=50)
        s.add(rede)
        s.flush()

        z1 = Zwischenruf(rede_id=rede.rede_id, fraktion="AfD", text="Ruf 1", kategorie="Widerspruch", ton_label="Aggression")
        z2 = Zwischenruf(rede_id=rede.rede_id, fraktion="AfD", text="Ruf 2", kategorie="Widerspruch", ton_label="Sarkasmus")
        z3 = Zwischenruf(rede_id=rede.rede_id, fraktion="AfD", text="Ruf 3", kategorie="Heiterkeit", ton_label="Humor")
        s.add_all([z1, z2, z3])
        s.flush()

        ton = TonAnalyse(s)
        df_kat = ton.waffen_mix_analyse(wahlperiode=20, dimension="kategorie")
        df_ton = ton.waffen_mix_analyse(wahlperiode=20, dimension="ton_label")
        df_comb = ton.waffen_mix_analyse(wahlperiode=20, dimension="combined")

    # Category dimension checks
    assert not df_kat.empty
    afd_kat = df_kat[df_kat["fraktion"] == "AfD"]
    assert pytest.approx(afd_kat["percentage"].sum(), 0.01) == 100.0

    # Ton Label dimension checks
    assert not df_ton.empty
    afd_ton = df_ton[df_ton["fraktion"] == "AfD"]
    assert pytest.approx(afd_ton["percentage"].sum(), 0.01) == 100.0

    # Combined dimension checks
    assert not df_comb.empty
    afd_comb = df_comb[df_comb["fraktion"] == "AfD"]
    assert pytest.approx(afd_comb["percentage"].sum(), 0.01) == 100.0


def test_temporale_unterbrechungs_dynamik():
    """Verify temporale_unterbrechungs_dynamik calculates relative positions and detects interjection cascades."""
    from src.analytics import RedeZeitAnalyse

    with get_session() as s:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2022, 1, 10), gesamtwortzahl=2000)
        redner = Redner(vorname="Alice", nachname="Weidel", fraktion="AfD")
        s.add_all([sitzung, redner])
        s.flush()

        # Speech with 100 words
        rede = Rede(sitzung_id=sitzung.sitzungs_id, redner_id=redner.redner_id, text="Lange Rede", wortanzahl=100)
        s.add(rede)
        s.flush()

        # Add 10 interjections to ensure close relative spacing (cascade >= 3 within 25% window)
        rufe = [Zwischenruf(rede_id=rede.rede_id, fraktion="SPD", text=f"Ruf {i}") for i in range(1, 11)]
        s.add_all(rufe)
        s.flush()

        rz = RedeZeitAnalyse(s)
        res = rz.temporale_unterbrechungs_dynamik(wahlperiode=20)

    assert "positions" in res
    assert "cascades" in res
    assert "total_cascades" in res

    df_pos = res["positions"]
    assert len(df_pos) == 10
    assert df_pos["rel_position_pct"].min() >= 0.0
    assert df_pos["rel_position_pct"].max() <= 100.0

    # Verify cascades
    assert res["total_cascades"] >= 1
    df_cascades = res["cascades"]
    assert not df_cascades.empty
    assert df_cascades.iloc[0]["cascade_count"] >= 3


@patch("src.app.get_session")
@patch("src.app.st")
def test_page_hypothesen_pruefung_h2_smoke(mock_st, mock_get_session):
    """Smoke test to ensure Hypothese 2 runs successfully on the new page."""
    from src.app import _page_hypothesen_pruefung

    # Configure st.columns to return MagicMocks
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

    # Configure st.tabs to return 3 tab MagicMocks
    tab_mocks = [MagicMock(), MagicMock(), MagicMock()]
    mock_st.tabs.return_value = tab_mocks

    # Configure st.slider to return float values
    mock_st.slider.side_effect = [-0.3, 0.3]

    # Mock text_input defaults
    mock_st.text_input.side_effect = ["Migration, Klimaschutz", "Tagesordnung, Ausschuss"]

    # Mock database results
    mock_eval = {
        "conflict_speeches": [{"wortanzahl": 500, "z_count": 5}, {"wortanzahl": 300, "z_count": 2}],
        "control_speeches": [{"wortanzahl": 1000, "z_count": 1}, {"wortanzahl": 800, "z_count": 0}],
        "conflict_sentiments": [-0.5, -0.6, -0.4, -0.8, -0.2],
        "control_sentiments": [0.0, 0.1, -0.1, 0.0, 0.0, 0.2]
    }

    with patch("src.app._cached_h2_evaluation", return_value=mock_eval), \
         patch("src.app.load_h3_data", return_value=pd.DataFrame()):
        _page_hypothesen_pruefung()

    assert mock_st.plotly_chart.called
