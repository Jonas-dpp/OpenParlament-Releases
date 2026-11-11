from unittest.mock import patch, MagicMock
from scripts.run_nlp_cli import (
    tone_scores_to_columns,
    chunked,
    is_column_attr,
    _pk_attr,
    _serialize_for_column,
    _is_json_column,
    gather_targets,
    main as run_nlp_cli_main,
)
from src.database import get_session, init_db
from src.models import Zwischenruf, Rede


def test_tone_scores_to_columns():
    scores = {"Aggression": 0.8, "Sarkasmus": 0.5, "Humor": 0.1, "Neutral": 0.2}
    cols = tone_scores_to_columns(scores)
    assert cols["tone_aggression"] == 0.8
    assert cols["tone_neutral"] == 0.2
    # None fallback
    assert tone_scores_to_columns(None)["tone_aggression"] == 0.0


def test_chunked():
    lst = [1, 2, 3, 4, 5]
    chunks = list(chunked(lst, 2))
    assert chunks == [[1, 2], [3, 4], [5]]


def test_is_column_attr_and_json_column():
    assert is_column_attr(Zwischenruf, "sentiment_score") is True
    assert is_column_attr(Zwischenruf, "invalid_col_name") is False
    assert _is_json_column(Zwischenruf, "tone_scores") is True
    # text column is not json
    assert _is_json_column(Zwischenruf, "text") is False


def test_pk_attr():
    assert _pk_attr(Zwischenruf) == "ruf_id"
    assert _pk_attr(Rede) == "rede_id"


def test_serialize_for_column():
    # If JSON column, returns dict/list as-is
    val = {"foo": "bar"}
    assert _serialize_for_column(Zwischenruf, "tone_scores", val) == val
    # If TEXT column, serializes to JSON string
    serialized = _serialize_for_column(Zwischenruf, "text", val)
    assert '{"foo": "bar"}' in serialized or '{"foo":' in serialized


def test_gather_targets_empty():
    # Ensure tables and missing columns exist
    init_db()
    with get_session() as db_session:
        rows = gather_targets(db_session, target="zwischenrufe", limit=5, only_unscored=True, sentiment_field="sentiment_score")
        assert isinstance(rows, list)


@patch("scripts.run_nlp_cli.get_session")
@patch("scripts.run_nlp_cli.NLPSession")
def test_run_nlp_cli_main_dry_run(mock_session_cls, mock_get_session):
    # Mocking arguments
    test_args = [
        "run_nlp_cli.py",
        "--target", "zwischenrufe",
        "--batch-size", "2",
        "--dry-run",
        "--limit", "2",
        "--no-sentiment",
    ]
    # Mock rows to be returned
    z1 = Zwischenruf(ruf_id=1, text="Test 1")
    z2 = Zwischenruf(ruf_id=2, text="Test 2")
    mock_session = MagicMock()
    mock_session.execute.return_value.scalars.return_value = [z1, z2]
    mock_get_session.return_value.__enter__.return_value = mock_session
    # Mock NLP Session and engines
    mock_agent = MagicMock()
    mock_agent.sentiment_engine = None
    mock_agent.tone_classifier = MagicMock()
    mock_agent.tone_classifier.classify_batch.return_value = [
        ("Neutral", {"Aggression": 0.0, "Sarkasmus": 0.0, "Humor": 0.0, "Neutral": 1.0}),
        ("Aggression", {"Aggression": 1.0, "Sarkasmus": 0.0, "Humor": 0.0, "Neutral": 0.0})
    ]
    mock_agent.addressee_detector = MagicMock()
    mock_agent.addressee_detector.detect_batch.return_value = [["Ad1"], ["Ad2"]]
    mock_session_cls.return_value.__enter__.return_value = mock_agent
    with patch("sys.argv", test_args):
        run_nlp_cli_main()
    # Under dry-run, commit should NOT be called on mock_session
    assert mock_session.commit.call_count == 0


@patch("scripts.run_nlp_cli.get_session")
@patch("scripts.run_nlp_cli.NLPSession")
def test_run_nlp_cli_main_persist_and_rollback(mock_session_cls, mock_get_session):
    # Mocking arguments for live run with exception to cover rollback
    test_args = [
        "run_nlp_cli.py",
        "--target", "zwischenrufe",
        "--batch-size", "2",
        "--limit", "2",
    ]
    z1 = Zwischenruf(ruf_id=1, text="Test 1")
    mock_session = MagicMock()
    mock_session.execute.return_value.scalars.return_value = [z1]
    mock_get_session.return_value.__enter__.return_value = mock_session
    # Make commit raise exception to trigger session.rollback()
    mock_session.commit.side_effect = Exception("Database write error")
    # Mock NLP Session and engines
    mock_agent = MagicMock()
    mock_agent.sentiment_engine = MagicMock()
    mock_agent.sentiment_engine.score_batch.return_value = [0.5]
    mock_agent.tone_classifier = MagicMock()
    mock_agent.tone_classifier.classify_batch.return_value = [("Neutral", {})]
    mock_agent.addressee_detector = MagicMock()
    mock_agent.addressee_detector.detect_batch.return_value = [[]]
    mock_session_cls.return_value.__enter__.return_value = mock_agent
    with patch("sys.argv", test_args):
        run_nlp_cli_main()
    # Rollback should be called due to exception in commit
    assert mock_session.rollback.call_count == 1
