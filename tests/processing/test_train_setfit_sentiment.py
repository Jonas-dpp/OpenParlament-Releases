"""
OpenParlament - Streamlit Dashboard
Copyright (C) 2026 Jonas-dpp

Tests for scripts/train_setfit_sentiment.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train_setfit_sentiment import (
    evaluate_baseline,
    generate_gold_standard_csv,
    map_continuous_score_to_class,
    run_pipeline,
    train_and_evaluate_setfit,
)


def test_map_continuous_score_to_class():
    assert map_continuous_score_to_class(0.8, -0.2, 0.2) == 1
    assert map_continuous_score_to_class(-0.8, -0.2, 0.2) == -1
    assert map_continuous_score_to_class(0.0, -0.2, 0.2) == 0
    assert map_continuous_score_to_class(0.2, -0.2, 0.2) == 0
    assert map_continuous_score_to_class(-0.2, -0.2, 0.2) == 0
    assert map_continuous_score_to_class(0.21, -0.2, 0.2) == 1
    assert map_continuous_score_to_class(-0.21, -0.2, 0.2) == -1


def test_generate_gold_standard_csv(tmp_path):
    csv_file = tmp_path / "test_gold_standard.csv"
    df = generate_gold_standard_csv(csv_file)

    assert csv_file.exists()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"text", "sentiment_raw", "true_label"}
    assert len(df) > 50
    assert set(df["true_label"].unique()) == {-1, 0, 1}


def test_evaluate_baseline(tmp_path):
    data = [
        {"text": "Zur Geschäftsordnung!", "sentiment_raw": -0.8, "true_label": 0},
        {"text": "Sehr gut!", "sentiment_raw": 0.9, "true_label": 1},
        {"text": "Pfui!", "sentiment_raw": -0.9, "true_label": -1},
    ]
    df = pd.DataFrame(data)
    res = evaluate_baseline(df)

    assert "classification_report" in res
    assert "confusion_matrix" in res
    assert "recall_neutral" in res
    assert "pred_labels" in res
    assert len(res["pred_labels"]) == 3


def test_train_and_evaluate_setfit_mocked(tmp_path):
    data = [
        {"text": f"Neutral text {i}", "sentiment_raw": -0.5, "true_label": 0}
        for i in range(10)
    ] + [
        {"text": f"Positive text {i}", "sentiment_raw": 0.8, "true_label": 1}
        for i in range(10)
    ] + [
        {"text": f"Negative text {i}", "sentiment_raw": -0.8, "true_label": -1}
        for i in range(10)
    ]
    df = pd.DataFrame(data)

    mock_model = MagicMock()
    mock_model.predict.return_value = [1] * 9  # 9 test samples (30% of 30)

    mock_dataset_cls = MagicMock()

    with patch("scripts.train_setfit_sentiment._SETFIT_AVAILABLE", True), \
         patch("scripts.train_setfit_sentiment.Dataset", mock_dataset_cls), \
         patch("scripts.train_setfit_sentiment.SetFitModel.from_pretrained", return_value=mock_model), \
         patch("scripts.train_setfit_sentiment.Trainer") as mock_trainer_cls:
        mock_trainer_obj = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_obj

        res = train_and_evaluate_setfit(
            df,
            model_name="dummy_model",
            output_dir=tmp_path / "model_out",
            num_epochs=1,
            num_iterations=2,
            batch_size=4,
        )

        assert "recall_neutral" in res
        assert "confusion_matrix" in res
        assert mock_trainer_obj.train.called
        assert mock_model.save_pretrained.called


def test_run_pipeline(tmp_path):
    csv_file = tmp_path / "gold_standard.csv"
    model_dir = tmp_path / "setfit_model"

    mock_model = MagicMock()
    # 36 test samples (30% of 120 dataset)
    mock_model.predict.return_value = [0] * 12 + [1] * 12 + [2] * 12

    mock_dataset_cls = MagicMock()

    with patch("scripts.train_setfit_sentiment._SETFIT_AVAILABLE", True), \
         patch("scripts.train_setfit_sentiment.Dataset", mock_dataset_cls), \
         patch("scripts.train_setfit_sentiment.SetFitModel.from_pretrained", return_value=mock_model), \
         patch("scripts.train_setfit_sentiment.Trainer") as mock_trainer_cls:
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        base_metrics, setfit_metrics = run_pipeline(
            csv_path=csv_file,
            output_dir=model_dir,
            model_name="dummy_model",
            lower_threshold=-0.2,
            upper_threshold=0.2,
            generate_if_missing=True,
        )

        assert csv_file.exists()
        assert "recall_neutral" in base_metrics
        assert "recall_neutral" in setfit_metrics
