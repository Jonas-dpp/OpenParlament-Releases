import pytest
from unittest.mock import MagicMock
from src.analytics import ThemenKarriere, _fts_query

def test_fts_query_length_limit():
    # Safe keyword
    assert _fts_query("Klimaschutz") == '"Klimaschutz"'

    # Overly long keyword
    long_kw = "a" * 251
    with pytest.raises(ValueError, match="Keyword exceeds the maximum safe limit of 250 characters."):
        _fts_query(long_kw)

def test_themen_karriere_keyword_length_limits():
    mock_session = MagicMock()
    tk = ThemenKarriere(mock_session)
    long_kw = "x" * 251

    # Verify keyword_trend raises ValueError
    with pytest.raises(ValueError, match="Keyword exceeds the maximum safe limit of 250 characters."):
        tk.keyword_trend(long_kw)

    # Verify multi_wp_keyword_trend raises ValueError
    with pytest.raises(ValueError, match="Keyword exceeds the maximum safe limit of 250 characters."):
        tk.multi_wp_keyword_trend(long_kw)

    # Verify keyword_peak_by_wp raises ValueError
    with pytest.raises(ValueError, match="Keyword exceeds the maximum safe limit of 250 characters."):
        tk.keyword_peak_by_wp(long_kw)

    # Verify keyword_aggression_correlation raises ValueError
    with pytest.raises(ValueError, match="Keyword exceeds the maximum safe limit of 250 characters."):
        tk.keyword_aggression_correlation(long_kw)
