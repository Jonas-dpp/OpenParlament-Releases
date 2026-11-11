import pandas as pd
from unittest.mock import MagicMock, patch
import src.app as app


def test_render_redezeit_analyse_metrics(monkeypatch):
    """Verifies that render_redezeit_analyse displays statutory metrics (norm_mean_1, norm_mean_2, mean_diff, r_squared, p_value)."""
    fake_df = pd.DataFrame({
        "fraktion": ["SPD", "CDU/CSU"],
        "total_worte": [10000, 12000],
        "wort_anteil_pct": [45.0, 55.0],
        "sitz_anteil_pct": [40.0, 60.0],
        "fairness_index": [1.125, 0.916],
        "ueber_unterrepraesentation": ["Überrepräsentiert", "Proportional"]
    })
    monkeypatch.setattr(app, "RedeZeitAnalyse", MagicMock(return_value=MagicMock(
        fairness_index=MagicMock(return_value=fake_df),
        top_redselige_redner=MagicMock(return_value=pd.DataFrame()),
        wortanzahl_vs_zwischenrufe=MagicMock(return_value=pd.DataFrame()),
    )))
    mock_metrics = []
    def make_mock_col():
        col = MagicMock()
        col.metric.side_effect = lambda label, value, *a, **kw: mock_metrics.append((label, value))
        return col
    def mock_columns(spec, *args, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [make_mock_col() for _ in range(count)]
    monkeypatch.setattr("streamlit.columns", mock_columns)
    with patch("src.app.get_session"):
        app.render_redezeit_analyse(20, None, None)
    labels = [m[0] for m in mock_metrics]
    assert any("norm_mean_1" in label for label in labels)
    assert any("norm_mean_2" in label for label in labels)
    assert any("mean_diff" in label for label in labels)
    assert any("R²" in label for label in labels)
    assert any("p_value" in label for label in labels)
