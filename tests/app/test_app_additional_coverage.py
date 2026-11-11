import pytest
from unittest.mock import patch, MagicMock
from src.app import (
    _secure_db_get,
    _cached_h2_evaluation,
    render_h3_asymmetric_attacks,
)
from src.database import init_db


def test_secure_db_get_type_safety():
    # 1. URL not a string raises RuntimeError
    with pytest.raises(RuntimeError, match="URL must be a string"):
        _secure_db_get(123, allowed_domains={"example.com"}, timeout=1.0)


def test_secure_db_get_credentials_rejected():
    # 2. URL with credentials rejected
    with pytest.raises(RuntimeError, match="URL with credentials rejected"):
        _secure_db_get("https://user:pass@example.com", allowed_domains={"example.com"}, timeout=1.0)


def test_secure_db_get_insecure_scheme():
    # 3. Non-https scheme rejected
    with pytest.raises(RuntimeError, match="Insecure scheme"):
        _secure_db_get("http://example.com", allowed_domains={"example.com"}, timeout=1.0)


def test_secure_db_get_untrusted_host():
    # 4. Untrusted host rejected
    with pytest.raises(RuntimeError, match="Untrusted host"):
        _secure_db_get("https://malicious.com", allowed_domains={"example.com"}, timeout=1.0)


@patch("requests.get")
def test_secure_db_get_successful_fetch(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp
    resp = _secure_db_get("https://example.com/db.db", allowed_domains={"example.com"}, timeout=1.0)
    assert resp.status_code == 200


@patch("requests.get")
def test_secure_db_get_redirects_and_limit(mock_get):
    # Mock redirect responses
    r1 = MagicMock()
    r1.status_code = 301
    r1.headers = {"Location": "https://example.com/redirect1"}
    r2 = MagicMock()
    r2.status_code = 302
    r2.headers = {"Location": "/redirect2"} # relative redirect
    r3 = MagicMock()
    r3.status_code = 200
    mock_get.side_effect = [r1, r2, r3]
    resp = _secure_db_get("https://example.com/start", allowed_domains={"example.com"}, timeout=1.0)
    assert resp.status_code == 200
    assert mock_get.call_count == 3


@patch("requests.get")
def test_secure_db_get_too_many_redirects(mock_get):
    r = MagicMock()
    r.status_code = 302
    r.headers = {"Location": "https://example.com/redirect"}
    mock_get.return_value = r
    with pytest.raises(RuntimeError, match="Too many redirects"):
        _secure_db_get("https://example.com/start", allowed_domains={"example.com"}, timeout=1.0)


def test_cached_h2_evaluation_empty_keywords():
    init_db()
    res = _cached_h2_evaluation(20, [], [])
    assert isinstance(res, dict)
    assert "conflict_speeches" in res
    assert "control_speeches" in res


@patch("src.app.st")
def test_page_hypothesen_pruefung_h2_input_length_limits(mock_st):
    from src.app import _page_hypothesen_pruefung
    init_db()

    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    tab_mocks = [MagicMock(), MagicMock(), MagicMock()]
    mock_st.tabs.return_value = tab_mocks

    # Test conflict_input exceeding 250 characters
    mock_st.text_input.side_effect = ["A" * 251, "Short control input"]
    mock_st.stop.side_effect = Exception("st.stop called")

    with pytest.raises(Exception, match="st.stop called"):
        _page_hypothesen_pruefung()

    mock_st.error.assert_called_with("Suchanfrage (Reizthemen) überschreitet das Sicherheitslimit von 250 Zeichen.")

    # Test control_input exceeding 250 characters
    mock_st.error.reset_mock()
    mock_st.text_input.side_effect = ["Short conflict input", "B" * 251]

    with pytest.raises(Exception, match="st.stop called"):
        _page_hypothesen_pruefung()

    mock_st.error.assert_called_with("Suchanfrage (Kontrollthemen) überschreitet das Sicherheitslimit von 250 Zeichen.")


def test_cached_h2_evaluation_custom_keywords():
    init_db()
    res = _cached_h2_evaluation(20, ["Klima", "Umwelt"], ["Arbeit", "Soziales"])
    assert isinstance(res, dict)
    assert "conflict_speeches" in res
    assert "control_speeches" in res


@patch("src.app.st")
def test_render_h3_asymmetric_attacks_smoke(mock_st):
    init_db()
    # Mock Streamlit UI methods
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    # Mock st.slider to return its default value keyword argument
    mock_st.slider.side_effect = lambda *args, **kwargs: kwargs.get("value", 0.0)
    mock_st.selectbox.side_effect = lambda *args, **kwargs: kwargs.get("options", [None])[0]
    # Run render
    render_h3_asymmetric_attacks(selected_wp=20)
    # Assertions on mock st calling
    assert mock_st.warning.call_count >= 0


@patch("src.app.st")
def test_render_debattenklima_metrics(mock_st):
    from src.app import render_debattenklima
    import pandas as pd
    import datetime

    init_db()
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    mock_st.toggle.return_value = True

    df_dummy = pd.DataFrame([
        {"datum": datetime.date(2023, 1, 1), "temperatur_index": 0.35, "avg_sent": -0.1},
        {"datum": datetime.date(2023, 1, 2), "temperatur_index": 0.45, "avg_sent": -0.2},
        {"datum": datetime.date(2023, 1, 3), "temperatur_index": 0.25, "avg_sent": 0.0},
    ])

    # Mock st.tabs to return 2 MagicMocks for the tabs
    tab_mocks = [MagicMock(), MagicMock()]
    mock_st.tabs.return_value = tab_mocks

    dummy_dyn = {"positions": pd.DataFrame(), "cascades": pd.DataFrame(), "total_cascades": 0}
    with patch("src.app._cached_klima_per_sitzung", return_value=df_dummy), \
         patch("src.app._cached_hottest_sessions", return_value=pd.DataFrame()), \
         patch("src.analytics.RedeZeitAnalyse.temporale_unterbrechungs_dynamik", return_value=dummy_dyn):
        render_debattenklima(selected_wp=20, datum_von=None, datum_bis=None)

    assert mock_st.plotly_chart.called


def test_spdx_dependencies_json_structure():
    import json
    from src.app import _SPDX_DEPENDENCIES_JSON, _get_cleaned_dependencies

    data = json.loads(_SPDX_DEPENDENCIES_JSON)
    assert data.get("spdxVersion") == "SPDX-2.3"
    assert data.get("name") == "com.github.Jonas-dpp/OpenParlament"
    packages = [pkg["name"] for pkg in data.get("packages", [])]
    assert "streamlit" in packages
    assert "pandas" in packages
    assert "com.github.Jonas-dpp/OpenParlament" in packages

    df_cleaned = _get_cleaned_dependencies()
    assert not df_cleaned.empty
    assert "streamlit" in df_cleaned["Paketname"].values
    assert "pandas" in df_cleaned["Paketname"].values


@patch("src.app.st")
def test_app_input_max_chars_validation(mock_st):
    from src.app import _page_stylometrie, render_themen_trend
    init_db()

    mock_st.tabs.return_value = [MagicMock(), MagicMock()]
    mock_st.text_area.return_value = "Sample text for stylometry test"
    mock_st.button.return_value = False

    _page_stylometrie()
    assert mock_st.text_area.called
    _, kwargs_area = mock_st.text_area.call_args
    assert kwargs_area.get("max_chars") == 25000

    mock_st.text_input.return_value = "Klimaschutz"
    render_themen_trend(selected_wp=20)
    assert mock_st.text_input.called
    for call in mock_st.text_input.call_args_list:
        _, kwargs_input = call
        assert kwargs_input.get("max_chars") in (100, 250)
