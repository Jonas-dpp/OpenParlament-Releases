import pytest
import numpy as np
import pandas as pd
from src.analytics import StylometryPredictor, StylometryAnalyzer

class DummyModel:
    def __init__(self):
        self.classes_ = np.array(["CDU/CSU", "SPD", "Bündnis 90/Die Grünen"])
        self.feature_names_in_ = np.array(["avg_sentence_length", "passive_ratio", "type_token_ratio", "fill_word_ratio"])

    def predict(self, X):
        return np.array(["SPD"])

    def predict_proba(self, X):
        return np.array([[0.1, 0.7, 0.2]])

def test_stylometry_predictor_mock_model():
    predictor = StylometryPredictor()
    predictor.model = DummyModel()

    text = "Wir müssen heute wichtige Entscheidungen treffen. Das ist von entscheidender Bedeutung für unser Land."

    # Test preprocess_features
    df_features = predictor.preprocess_features(text)
    assert isinstance(df_features, pd.DataFrame)
    for col in ["avg_sentence_length", "passive_ratio", "type_token_ratio", "fill_word_ratio"]:
        assert col in df_features.columns

    # Test predict
    pred = predictor.predict(text)
    assert pred == "SPD"

    # Test predict_proba
    probs = predictor.predict_proba(text)
    assert isinstance(probs, dict)
    assert "SPD" in probs
    assert probs["SPD"] == 0.7
    assert probs["Bündnis 90/Die Grünen"] == 0.2
    assert probs["CDU/CSU"] == 0.1

    # Test explain
    explanations = predictor.explain(text)
    assert isinstance(explanations, list)
    assert len(explanations) > 0

    # Test metadata
    meta = predictor.metadata()
    assert isinstance(meta, dict)


def test_stylometry_analyzer_predict_enriched():
    analyzer = StylometryAnalyzer(session=None)
    analyzer.model = DummyModel()

    text = "Ganz kurze Sätze. Hallo. Tschüss."
    res = analyzer.predict_enriched(text)

    assert res["prediction"] == "SPD"
    assert res["confidence"] == 0.7
    assert res["uncertain"] is False
    assert isinstance(res["probabilities"], dict)
    assert isinstance(res["explanations"], list)
    assert "+ extremely short, snappy sentence structures" in res["explanations"]


def test_stylometry_security_length_limit():
    """Verify that StylometryPredictor enforces the length limit and prevents DoS."""
    predictor = StylometryPredictor()
    predictor.model = DummyModel()

    # Create a string of 25,001 characters
    long_text = "a" * 25001

    with pytest.raises(ValueError, match="Input text exceeds the maximum safe limit"):
        predictor.preprocess_features(long_text)


def test_stylometry_security_path_traversal():
    """Verify that StylometryPredictor rejects paths outside of safe roots to prevent directory traversal."""
    # 1. Unsafe model_path
    with pytest.raises(ValueError, match="Unsafe model path rejected."):
        StylometryPredictor(model_path="/etc/passwd")

    with pytest.raises(ValueError, match="Unsafe model path rejected."):
        StylometryPredictor(model_path="../../malicious.pkl")

    # 2. Unsafe speaker_averages_path
    with pytest.raises(ValueError, match="Unsafe speaker averages path rejected."):
        StylometryPredictor(speaker_averages_path="/etc/hosts")

    with pytest.raises(ValueError, match="Unsafe speaker averages path rejected."):
        StylometryPredictor(speaker_averages_path="../outside_averages.json")

    # 3. Safe paths should not raise a path validation ValueError (it might raise other errors if the file doesn't exist, which is fine, but not the security ValueError)
    try:
        StylometryPredictor(model_path="models/nonexistent_safe.pkl")
    except ValueError as e:
        assert "Unsafe" not in str(e)
    except Exception:
        pass


@pytest.mark.filterwarnings("ignore")
def test_load_data_and_extract_features_telemetry(caplog):
    """Test that load_data_and_extract_features prints progress logs with ETA for speech filtering."""
    from unittest.mock import patch, MagicMock
    import logging
    from scripts.train_stylometry import load_data_and_extract_features, TextLayer
    from datetime import datetime

    with patch("scripts.train_stylometry.get_session") as mock_get_session, \
         patch("scripts.train_stylometry.StylometryEngine") as mock_engine_cls:

        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session

        # 1005 dummy rows to trigger total_rows > 1000 logging logic
        dummy_date = datetime(2023, 1, 1)
        dummy_rows = [
            ("Text", "SPD", 1, 20, dummy_date, 10)
        ] * 1005
        mock_session.execute.return_value.fetchall.return_value = dummy_rows

        # Mock the engine feature extraction so it doesn't process text and returns correctly sized batches
        def mock_extract_features_batch(texts, batch_size=256):
            return [{"some_feature": 1.0}] * len(texts)

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.extract_features_batch.side_effect = mock_extract_features_batch

        with caplog.at_level(logging.INFO, logger="train_stylometry"):
            df = load_data_and_extract_features(TextLayer.CLEAN)

        assert not df.empty

        # Check that our custom SpeechFiltering telemetry log is present
        filtering_logs = [record.message for record in caplog.records if "SpeechFiltering:" in record.message]
        assert len(filtering_logs) > 0
        # Assert format: SpeechFiltering: X/Y | Z% | ETA: MM:SS (or HH:MM:SS)
        assert "SpeechFiltering: 1/1005" in filtering_logs[0]


def test_stylometry_analyzer_security_path_traversal():
    """Verify that StylometryAnalyzer rejects paths outside of safe roots to prevent directory traversal."""
    # 1. Unsafe model_path
    with pytest.raises(ValueError, match="Unsafe model path rejected."):
        StylometryAnalyzer(session=None, model_path="/etc/passwd")

    with pytest.raises(ValueError, match="Unsafe model path rejected."):
        StylometryAnalyzer(session=None, model_path="../../malicious.pkl")

    # 2. Safe paths should not raise a path validation ValueError (it might raise other errors if the file doesn't exist, which is fine, but not the security ValueError)
    try:
        StylometryAnalyzer(session=None, model_path="models/nonexistent_safe.pkl")
    except ValueError as e:
        assert "Unsafe" not in str(e)
    except Exception:
        pass
