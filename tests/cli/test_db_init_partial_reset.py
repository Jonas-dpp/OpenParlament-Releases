"""
Unit and integration tests for partial database resets (Framing & NLP) in scripts/db_init.py and src/database.py.
"""

from __future__ import annotations

from unittest.mock import patch
import pytest
from sqlalchemy import select

from src.database import (
    drop_db,
    get_session,
    init_db,
    reset_db_state,
    reset_framing,
    reset_nlp,
)
from src.models import (
    FramingResult,
    LexicalResult,
    ReactionGapResult,
    Rede,
    Redner,
    Sitzung,
    Zwischenruf,
)
from scripts.db_init import main as db_init_main


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_partial_reset.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{test_db_path}")
    reset_db_state()
    init_db()
    yield
    drop_db()
    reset_db_state()


def _seed_sample_data():
    with get_session() as session:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, gesamtwortzahl=500)
        redner = Redner(vorname="Max", nachname="Mustermann", fraktion="CDU/CSU")
        session.add_all([sitzung, redner])
        session.flush()

        rede = Rede(
            sitzung_id=sitzung.sitzungs_id,
            redner_id=redner.redner_id,
            text="Dies ist eine Testrede.",
            sentiment_score=0.75,
            ton_label="Humor",
            tone_aggression=0.1,
            tone_sarkasmus=0.2,
            tone_humor=0.7,
            tone_neutral=0.0,
            tone_scores={"Humor": 0.7, "Aggression": 0.1, "Sarkasmus": 0.2, "Neutral": 0.0},
            adressaten="SPD, FDP",
        )
        session.add(rede)
        session.flush()

        zr = Zwischenruf(
            rede_id=rede.rede_id,
            text="Sehr gut!",
            fraktion="SPD",
            sentiment_score=0.8,
            ton_label="Humor",
            tone_aggression=0.0,
            tone_sarkasmus=0.1,
            tone_humor=0.8,
            tone_neutral=0.1,
            tone_scores={"Humor": 0.8},
            adressaten="Mustermann",
        )
        session.add(zr)
        session.flush()

        rg = ReactionGapResult(
            rede_id=rede.rede_id,
            zwischenruf_id=zr.ruf_id,
            speaker_sentiment=0.75,
            reaction_sentiment=0.8,
            gap=0.05,
            time_to_first_interjection=10.0,
        )
        session.add(rg)

        fr = FramingResult(
            wahlperiode=20,
            fraktion="CDU/CSU",
            segment="all",
            variant="global",
            term="steuererleichterung",
            score=4.2,
        )
        session.add(fr)

        lr = LexicalResult(
            wahlperiode=20,
            fraktion="CDU/CSU",
            ttr=0.6,
            msttr=0.55,
            lexical_density=0.5,
            total_tokens=1000,
            total_types=600,
        )
        session.add(lr)


def test_reset_framing_helper():
    _seed_sample_data()

    with get_session() as session:
        assert session.execute(select(FramingResult)).scalar() is not None
        assert session.execute(select(LexicalResult)).scalar() is not None
        assert session.execute(select(Rede)).scalar() is not None

    reset_framing()

    with get_session() as session:
        # Framing and Lexical tables should be empty
        assert session.execute(select(FramingResult)).scalars().all() == []
        assert session.execute(select(LexicalResult)).scalars().all() == []

        # Speeches and interjections must remain intact
        rede = session.execute(select(Rede)).scalar_one()
        assert rede.sentiment_score == 0.75
        assert rede.text == "Dies ist eine Testrede."

        zr = session.execute(select(Zwischenruf)).scalar_one()
        assert zr.sentiment_score == 0.8
        assert zr.text == "Sehr gut!"


def test_reset_nlp_helper():
    _seed_sample_data()

    with get_session() as session:
        rede = session.execute(select(Rede)).scalar_one()
        assert rede.sentiment_score == 0.75
        assert rede.ton_label == "Humor"

        zr = session.execute(select(Zwischenruf)).scalar_one()
        assert zr.sentiment_score == 0.8
        assert zr.ton_label == "Humor"

        assert session.execute(select(ReactionGapResult)).scalar() is not None

    reset_nlp()

    with get_session() as session:
        rede = session.execute(select(Rede)).scalar_one()
        # Text and relational attributes must remain
        assert rede.text == "Dies ist eine Testrede."
        assert rede.sentiment_score is None
        assert rede.ton_label is None
        assert rede.tone_aggression is None
        assert rede.tone_sarkasmus is None
        assert rede.tone_humor is None
        assert rede.tone_neutral is None
        assert rede.tone_scores is None
        assert rede.adressaten is None

        zr = session.execute(select(Zwischenruf)).scalar_one()
        assert zr.text == "Sehr gut!"
        assert zr.fraktion == "SPD"
        assert zr.sentiment_score is None
        assert zr.ton_label is None
        assert zr.tone_aggression is None
        assert zr.tone_sarkasmus is None
        assert zr.tone_humor is None
        assert zr.tone_neutral is None
        assert zr.tone_scores is None
        assert zr.adressaten is None

        # Reaction gap results must be cleared
        assert session.execute(select(ReactionGapResult)).scalars().all() == []

        # Framing must remain intact
        assert session.execute(select(FramingResult)).scalar() is not None
        assert session.execute(select(LexicalResult)).scalar() is not None


def test_cli_reset_framing():
    _seed_sample_data()

    with patch("sys.argv", ["db_init.py", "--reset-framing"]):
        db_init_main()

    with get_session() as session:
        assert session.execute(select(FramingResult)).scalars().all() == []
        assert session.execute(select(LexicalResult)).scalars().all() == []

        # Speeches and interjections unaffected
        rede = session.execute(select(Rede)).scalar_one()
        assert rede.sentiment_score == 0.75
        assert rede.text == "Dies ist eine Testrede."


def test_cli_reset_nlp():
    _seed_sample_data()

    with patch("sys.argv", ["db_init.py", "--reset-nlp"]):
        db_init_main()

    with get_session() as session:
        rede = session.execute(select(Rede)).scalar_one()
        assert rede.sentiment_score is None
        assert rede.ton_label is None
        assert rede.text == "Dies ist eine Testrede."

        zr = session.execute(select(Zwischenruf)).scalar_one()
        assert zr.sentiment_score is None
        assert zr.ton_label is None
        assert zr.text == "Sehr gut!"

        assert session.execute(select(ReactionGapResult)).scalars().all() == []

        # Framing unaffected
        assert session.execute(select(FramingResult)).scalar() is not None
        assert session.execute(select(LexicalResult)).scalar() is not None


def test_cli_reset_framing_and_nlp_combined():
    _seed_sample_data()

    with patch("sys.argv", ["db_init.py", "--reset-framing", "--reset-nlp"]):
        db_init_main()

    with get_session() as session:
        # Framing cleared
        assert session.execute(select(FramingResult)).scalars().all() == []
        assert session.execute(select(LexicalResult)).scalars().all() == []

        # NLP cleared
        rede = session.execute(select(Rede)).scalar_one()
        assert rede.sentiment_score is None
        assert rede.text == "Dies ist eine Testrede."

        zr = session.execute(select(Zwischenruf)).scalar_one()
        assert zr.sentiment_score is None
        assert zr.text == "Sehr gut!"

        assert session.execute(select(ReactionGapResult)).scalars().all() == []

        # Core session and speaker intact
        assert session.execute(select(Sitzung)).scalar() is not None
        assert session.execute(select(Redner)).scalar() is not None


def test_cli_partial_reset_on_empty_fresh_db(tmp_path, monkeypatch):
    fresh_db_path = tmp_path / "fresh_uninitialized.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{fresh_db_path}")
    reset_db_state()

    # Database does not have tables created yet. Partial resets should not crash with "no such table".
    with patch("sys.argv", ["db_init.py", "--reset-framing", "--reset-nlp"]):
        db_init_main()

    with get_session() as session:
        assert session.execute(select(FramingResult)).scalars().all() == []
        assert session.execute(select(Rede)).scalars().all() == []
