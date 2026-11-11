import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from scripts.run_scraper import (
    ProtocolImporter,
    run_nlp_scoring,
    main as run_scraper_main,
)
from src.models import Sitzung, Zwischenruf


def test_protocol_importer_redner_key():
    importer = ProtocolImporter()
    # 1. Has bundestag_id
    r1 = MagicMock(bundestag_id=12345, vollname="Test Name")
    assert importer._redner_key(r1) == "12345"
    # 2. No bundestag_id, fall back to vollname lowercase strip
    r2 = MagicMock(bundestag_id=None, vollname="  Test Name  ")
    assert importer._redner_key(r2) == "test name"


@patch("scripts.run_scraper.get_session")
def test_protocol_importer_import_file_already_exists(mock_get_session):
    importer = ProtocolImporter()
    path = Path("20001.xml") # WP 20, SNR 1
    mock_session = MagicMock()
    # Mocking existing Sitzung query to return a Sitzung (already exists)
    mock_session.execute.return_value.scalar_one_or_none.return_value = Sitzung(wahlperiode=20, sitzungsnr=1)
    mock_get_session.return_value.__enter__.return_value = mock_session
    res = importer.import_file(path)
    assert res is False # Skipped import


@patch("scripts.run_scraper.get_session")
def test_protocol_importer_import_file_exception_rollback(mock_get_session):
    importer = ProtocolImporter()
    path = Path("20002.xml")
    mock_session = MagicMock()
    mock_session.execute.return_value.scalar_one_or_none.return_value = None
    mock_get_session.return_value.__enter__.return_value = mock_session
    # Mock parser to raise error
    with patch.object(importer, "_parser") as mock_parser:
        mock_parser.parse_file.side_effect = ValueError("XML error")
        with pytest.raises(ValueError, match="XML error"):
            importer.import_file(path)
    # Should rollback on exception
    assert mock_session.rollback.call_count == 1


@patch("scripts.run_scraper.get_session")
@patch("scripts.run_scraper.NLPSession")
def test_run_nlp_scoring_accumulated_commit(mock_session_cls, mock_get_session):
    mock_session = MagicMock()
    # 2 unscored rows
    z1 = Zwischenruf(ruf_id=1, text="Zuruf 1")
    z2 = Zwischenruf(ruf_id=2, text="Zuruf 2")
    mock_session.execute.return_value.scalars.return_value.all.return_value = [z1, z2]
    mock_get_session.return_value.__enter__.return_value = mock_session
    mock_agent = MagicMock()
    mock_agent.sentiment_engine = MagicMock()
    mock_agent.sentiment_engine.score_batch.return_value = [0.1, 0.2]
    mock_agent.tone_classifier = MagicMock()
    mock_agent.tone_classifier.classify_batch.return_value = [("Neutral", {}), ("Lachen", {})]
    mock_agent.addressee_detector = MagicMock()
    mock_agent.addressee_detector.detect_batch.return_value = [[], []]
    mock_session_cls.return_value.__enter__.return_value = mock_agent
    # Run with batch_size=1 and commit_interval=1
    run_nlp_scoring(batch_size=1, commit_interval=1)
    # commit should be called because commit_interval is 1
    assert mock_session.commit.call_count > 0


@patch("scripts.run_scraper.get_session")
@patch("scripts.run_scraper.NLPSession")
def test_run_nlp_scoring_exception_rollback(mock_session_cls, mock_get_session):
    mock_session = MagicMock()
    z1 = Zwischenruf(ruf_id=1, text="Zuruf 1")
    mock_session.execute.return_value.scalars.return_value.all.return_value = [z1]
    mock_get_session.return_value.__enter__.return_value = mock_session
    mock_agent = MagicMock()
    mock_agent.sentiment_engine = MagicMock()
    mock_agent.sentiment_engine.score_batch.return_value = [0.1]
    mock_agent.tone_classifier = None
    mock_agent.addressee_detector = None
    mock_session_cls.return_value.__enter__.return_value = mock_agent
    # Make execute update fail or commit fail to trigger rollback
    mock_session.commit.side_effect = Exception("Commit error")
    with pytest.raises(Exception, match="Commit error"):
        run_nlp_scoring(batch_size=1, commit_interval=1)
    assert mock_session.rollback.call_count == 1


@patch("scripts.run_scraper.BundestagScraper")
@patch("scripts.run_scraper.ProtocolImporter")
def test_run_scraper_main_flow(mock_importer_cls, mock_scraper_cls):
    test_args = [
        "run_scraper.py",
        "--wahlperiode", "19", "21",
        "--max-pages", "1",
        "--delay", "0.0",
    ]
    mock_scraper = MagicMock()
    mock_scraper.fetch_protocol_urls.side_effect = [
        ["http://test.url/19_1"],
        ["http://test.url/21_1"],
    ]
    mock_scraper.download_all.return_value = [Path("19001.xml"), Path("21001.xml")]
    mock_scraper_cls.return_value = mock_scraper
    mock_importer = MagicMock()
    mock_importer.import_file.return_value = True
    mock_importer_cls.return_value = mock_importer
    with patch("sys.argv", test_args):
        run_scraper_main()
    assert mock_scraper.fetch_protocol_urls.call_count == 2
    mock_scraper.fetch_protocol_urls.assert_any_call(wahlperiode=19, max_pages=1)
    mock_scraper.fetch_protocol_urls.assert_any_call(wahlperiode=21, max_pages=1)
    assert mock_importer.import_file.call_count == 2


@patch("scripts.run_scraper.BundestagScraper")
@patch("scripts.run_scraper.ProtocolImporter")
def test_run_scraper_default_wahlperioden(mock_importer_cls, mock_scraper_cls):
    test_args = [
        "run_scraper.py",
        "--max-pages", "1",
        "--delay", "0.0",
    ]
    mock_scraper = MagicMock()
    mock_scraper.fetch_protocol_urls.return_value = []
    mock_scraper.download_all.return_value = []
    mock_scraper_cls.return_value = mock_scraper
    mock_importer_cls.return_value = MagicMock()
    with patch("sys.argv", test_args):
        run_scraper_main()
    # Default wahlperioden: 18, 19, 20, 21
    assert mock_scraper.fetch_protocol_urls.call_count == 4
    called_wps = [call.kwargs["wahlperiode"] for call in mock_scraper.fetch_protocol_urls.call_args_list]
    assert called_wps == [18, 19, 20, 21]
