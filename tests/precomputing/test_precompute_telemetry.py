import logging
from unittest.mock import MagicMock, patch
import pandas as pd

from scripts.precompute_complexity import precompute_for_wp as precompute_complexity_wp
from scripts.precompute_framing import precompute_for_wp as precompute_framing_wp


def test_precompute_complexity_telemetry(caplog):
    """Test that precompute_complexity.py correctly logs telemetry and ETA during DB prepare."""
    caplog.set_level(logging.INFO)

    # 1. Mock the pandas DataFrames returned by the analyzers
    # We want df_complexity to have > 1000 records to trigger the throttled log step at index 1000
    df_complexity_data = []
    for idx in range(1, 1005):
        df_complexity_data.append({
            "rede_id": idx,
            "redner_id": 100 + idx,
            "flesch_score": 50.0,
            "avg_sentence_length": 15.0,
            "subordinate_clause_ratio": 0.2,
            "avg_word_length": 5.0,
            "long_word_ratio": 0.25,
            "lexical_diversity": 0.6,
        })
    df_complexity = pd.DataFrame(df_complexity_data)

    df_gap_data = []
    for idx in range(1, 1002):
        df_gap_data.append({
            "rede_id": idx,
            "zwischenruf_id": 1000 + idx,
            "topic_id": "TOP1",
            "speaker_sentiment": 0.1,
            "reaction_sentiment": -0.2,
            "gap": -0.3,
            "time_to_first_interjection": 10.0,
        })
    df_gap = pd.DataFrame(df_gap_data)

    # 2. Mock the DB Session and the sub-transactions
    mock_session = MagicMock()
    # Mock the count query returning 1 so the fast pre-check passes
    mock_session.execute.return_value.scalar.return_value = 1

    # 3. Patch the run methods of the analyzers
    with patch("scripts.precompute_complexity.SpeechComplexityAnalysis") as MockSC, \
         patch("scripts.precompute_complexity.ReactionGapAnalysis") as MockRG:

        mock_sc_instance = MockSC.return_value
        mock_sc_instance.run.return_value = df_complexity

        mock_rg_instance = MockRG.return_value
        mock_rg_instance.run.return_value = df_gap

        # Execute
        precompute_complexity_wp(mock_session, wp=20)

    # 4. Assertions on captured logs
    log_text = caplog.text

    assert "SpeechComplexityDBPrepare: 1/1004 | 0% | ETA:" in log_text
    assert "SpeechComplexityDBPrepare: 1000/1004 | 99% | ETA:" in log_text
    assert "SpeechComplexityDBPrepare: 1004/1004 | 100% | ETA:" in log_text

    assert "ReactionGapDBPrepare: 1/1001 | 0% | ETA:" in log_text
    assert "ReactionGapDBPrepare: 1000/1001 | 99% | ETA:" in log_text
    assert "ReactionGapDBPrepare: 1001/1001 | 100% | ETA:" in log_text


def test_precompute_framing_telemetry(caplog):
    """Test that precompute_framing.py correctly logs telemetry and ETA during DB prepare."""
    caplog.set_level(logging.INFO)

    # 1. Mock the pandas DataFrames returned by the analyzers
    # We want df_terms to have > 500 records to trigger the throttled log step at index 500
    df_terms_data = []
    for idx in range(1, 502):
        df_terms_data.append({
            "fraktion": "SPD",
            "term": f"arbeit_{idx}",
            "score": 0.85,
        })
    df_terms = pd.DataFrame(df_terms_data)

    # We want df_lexical to have > 5 records to trigger the throttled log step at index 5
    df_lexical_data = []
    for idx in range(1, 7):
        df_lexical_data.append({
            "fraktion": f"Faction_{idx}",
            "ttr": 0.5,
            "msttr": 0.6,
            "lexical_density": 0.4,
            "total_tokens": 1000,
            "total_types": 500,
        })
    df_lexical = pd.DataFrame(df_lexical_data)

    # 2. Mock the DB Session
    mock_session = MagicMock()

    # 3. Patch the run methods of the analyzers
    with patch("scripts.precompute_framing.FramingAnalysis") as MockFA, \
         patch("scripts.precompute_framing.LexicalAnalysis") as MockLA:

        mock_fa_instance = MockFA.return_value
        mock_fa_instance.load_segment_documents.side_effect = lambda wp, segment, **kwargs: (
            pd.DataFrame([{"fraktion": "SPD", "text": "arbeit"}]) if segment == "all" else pd.DataFrame()
        )
        mock_fa_instance.analyze_segment.side_effect = lambda segment, variant, **kwargs: (
            df_terms if variant == "global" and segment == "all" else pd.DataFrame()
        )

        mock_la_instance = MockLA.return_value
        mock_la_instance.run.return_value = df_lexical

        # Execute
        precompute_framing_wp(mock_session, wp=20, n_terms=500)

    # 4. Assertions on captured logs
    log_text = caplog.text

    assert "FramingDBPrepare: 1/501 | 0% | ETA:" in log_text
    assert "FramingDBPrepare: 500/501 | 99% | ETA:" in log_text
    assert "FramingDBPrepare: 501/501 | 100% | ETA:" in log_text

    assert "LexicalDBPrepare: 1/6 | 16% | ETA:" in log_text
    assert "LexicalDBPrepare: 5/6 | 83% | ETA:" in log_text
    assert "LexicalDBPrepare: 6/6 | 100% | ETA:" in log_text
