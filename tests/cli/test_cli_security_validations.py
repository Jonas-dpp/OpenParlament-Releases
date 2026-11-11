import pytest
from unittest.mock import patch, MagicMock
import argparse

# Import the main functions from the scripts
from scripts.run_scraper import main as run_scraper_main
from scripts.run_nlp_cli import main as run_nlp_cli_main
from scripts.precompute_framing import main as precompute_framing_main
from scripts.precompute_complexity import main as precompute_complexity_main
from scripts.precompute_h3 import main as precompute_h3_main
from scripts.export_dashboard_plots import main as export_dashboard_plots_main
from scripts.migrate_text_processing import parse_args as migrate_text_processing_parse_args
from scripts.train_stylometry import train_model as train_stylometry_main
from scripts.extract_sarcasm_case_studies import main as extract_sarcasm_case_studies_main
from scripts.train_setfit_sentiment import (
    main as train_setfit_sentiment_main,
    map_continuous_score_to_class,
)

def test_run_scraper_validations():
    # 1a. Empty wahlperiode
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=[],
            max_pages=100,
            delay=1.0,
            redownload=False,
            nlp=False,
            nlp_batch_size=100,
            nlp_cuda=False,
            nlp_device_index=None,
            nlp_use_ner=False,
            nlp_fp16=False,
            commit_interval=10,
        )
        with pytest.raises(ValueError, match="At least one wahlperiode must be specified"):
            run_scraper_main()

    # 1b. Invalid wahlperiode (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=[20, 0],
            max_pages=100,
            delay=1.0,
            redownload=False,
            nlp=False,
            nlp_batch_size=100,
            nlp_cuda=False,
            nlp_device_index=None,
            nlp_use_ner=False,
            nlp_fp16=False,
            commit_interval=10,
        )
        with pytest.raises(ValueError, match="wahlperiode values must be positive integers"):
            run_scraper_main()

    # 2. Invalid max_pages (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=[20],
            max_pages=0,
            delay=1.0,
            redownload=False,
            nlp=False,
            nlp_batch_size=100,
            nlp_cuda=False,
            nlp_device_index=None,
            nlp_use_ner=False,
            nlp_fp16=False,
            commit_interval=10,
        )
        with pytest.raises(ValueError, match="max-pages must be a positive integer"):
            run_scraper_main()

    # 3. Invalid delay (< 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=[20],
            max_pages=100,
            delay=-1.0,
            redownload=False,
            nlp=False,
            nlp_batch_size=100,
            nlp_cuda=False,
            nlp_device_index=None,
            nlp_use_ner=False,
            nlp_fp16=False,
            commit_interval=10,
        )
        with pytest.raises(ValueError, match="delay must be a non-negative float/int"):
            run_scraper_main()

    # 4. Invalid nlp_batch_size (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=[20],
            max_pages=100,
            delay=1.0,
            redownload=False,
            nlp=False,
            nlp_batch_size=0,
            nlp_cuda=False,
            nlp_device_index=None,
            nlp_use_ner=False,
            nlp_fp16=False,
            commit_interval=10,
        )
        with pytest.raises(ValueError, match="nlp-batch-size must be a positive integer"):
            run_scraper_main()

    # 5. Invalid commit_interval (< 1)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=[20],
            max_pages=100,
            delay=1.0,
            redownload=False,
            nlp=False,
            nlp_batch_size=100,
            nlp_cuda=False,
            nlp_device_index=None,
            nlp_use_ner=False,
            nlp_fp16=False,
            commit_interval=0,
        )
        with pytest.raises(ValueError, match="commit-interval must be >= 1"):
            run_scraper_main()


def test_run_nlp_cli_validations():
    # 1. Invalid batch_size (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            target="zwischenrufe",
            batch_size=0,
            device=-1,
            cuda=False,
            device_index=None,
            use_ner=False,
            fp16=False,
            dry_run=True,
            only_unscored=False,
            limit=0,
            sentiment_field="sentiment_score",
            tone_label_field="ton_label",
            tone_scores_field="tone_scores",
            addressee_field="adressaten",
            no_sentiment=False,
            no_tone=False,
            no_addressee=False,
        )
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            run_nlp_cli_main()

    # 2. Invalid device (< -1)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            target="zwischenrufe",
            batch_size=100,
            device=-2,
            cuda=False,
            device_index=None,
            use_ner=False,
            fp16=False,
            dry_run=True,
            only_unscored=False,
            limit=0,
            sentiment_field="sentiment_score",
            tone_label_field="ton_label",
            tone_scores_field="tone_scores",
            addressee_field="adressaten",
            no_sentiment=False,
            no_tone=False,
            no_addressee=False,
        )
        with pytest.raises(ValueError, match="device must be >= -1"):
            run_nlp_cli_main()

    # 3. Invalid limit (< 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            target="zwischenrufe",
            batch_size=100,
            device=-1,
            cuda=False,
            device_index=None,
            use_ner=False,
            fp16=False,
            dry_run=True,
            only_unscored=False,
            limit=-1,
            sentiment_field="sentiment_score",
            tone_label_field="ton_label",
            tone_scores_field="tone_scores",
            addressee_field="adressaten",
            no_sentiment=False,
            no_tone=False,
            no_addressee=False,
        )
        with pytest.raises(ValueError, match="limit must be non-negative"):
            run_nlp_cli_main()


def test_precompute_framing_validations():
    # 1. Invalid wahlperiode (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=0,
            all=False,
            n_terms=500,
        )
        with pytest.raises(ValueError, match="wahlperiode must be a positive integer"):
            precompute_framing_main()

    # 2. Invalid n_terms (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=20,
            all=False,
            n_terms=0,
        )
        with pytest.raises(ValueError, match="n-terms must be a positive integer"):
            precompute_framing_main()


def test_precompute_complexity_validations():
    # 1. Invalid wahlperiode (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=0,
            all=False,
            limit=None,
            debug=False,
        )
        with pytest.raises(ValueError, match="wahlperiode must be a positive integer"):
            precompute_complexity_main()

    # 2. Invalid limit (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=20,
            all=False,
            limit=0,
            debug=False,
        )
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            precompute_complexity_main()


def test_precompute_h3_validations():
    # 1. Invalid wahlperiode (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            wahlperiode=0,
            all=False,
        )
        with pytest.raises(ValueError, match="wahlperiode must be a positive integer"):
            precompute_h3_main()


def test_export_dashboard_plots_validations():
    # 1. Invalid max_pages (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            output_dir=MagicMock(),
            format="both",
            max_pages=0,
            max_configs=10,
        )
        with pytest.raises(ValueError, match="max-pages must be a positive integer"):
            export_dashboard_plots_main()

    # 2. Invalid max_configs (< 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            output_dir=MagicMock(),
            format="both",
            max_pages=5,
            max_configs=-1,
        )
        with pytest.raises(ValueError, match="max-configs must be a non-negative integer"):
            export_dashboard_plots_main()


def test_migrate_text_processing_validations():
    # 1. Invalid batch_size (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            batch_size=0,
            workers=1,
            limit=None,
            resume=False,
            force=False,
            dry_run=True,
            verbose=False,
            stats_only=False,
        )
        with pytest.raises(ValueError, match="batch-size must be a positive integer"):
            migrate_text_processing_parse_args()

    # 2. Invalid limit (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            batch_size=500,
            workers=1,
            limit=0,
            resume=False,
            force=False,
            dry_run=True,
            verbose=False,
            stats_only=False,
        )
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            migrate_text_processing_parse_args()


def test_train_stylometry_validations():
    # 1. Invalid text_layer
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            text_layer="invalid_layer",
            model_type="all",
        )
        with pytest.raises(ValueError, match="text-layer must be one of"):
            train_stylometry_main()

    # 2. Invalid model_type
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            text_layer="clean",
            model_type="invalid_model",
        )
        with pytest.raises(ValueError, match="model-type must be one of"):
            train_stylometry_main()


def test_extract_sarcasm_case_studies_validations():
    # 1. Invalid limit (<= 0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            db="./data/openparlament.db",
            limit=0,
            output_csv="./data/sarcasm_case_studies.csv",
            output_md="./data/sarcasm_case_studies.md",
        )
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            extract_sarcasm_case_studies_main()


def test_train_setfit_sentiment_validations():
    # 1. Invalid lower >= upper in main argument validation
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            csv_path="data/gold_standard.csv",
            output_dir="models/setfit_sentiment_model",
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            lower_threshold=0.5,
            upper_threshold=0.2,
            generate_gold_standard=False,
        )
        with pytest.raises(ValueError, match="Invalid sentiment threshold boundaries"):
            train_setfit_sentiment_main()

    # 2. Out of bounds lower threshold (< -1.0)
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            csv_path="data/gold_standard.csv",
            output_dir="models/setfit_sentiment_model",
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            lower_threshold=-1.5,
            upper_threshold=0.2,
            generate_gold_standard=False,
        )
        with pytest.raises(ValueError, match="Invalid sentiment threshold boundaries"):
            train_setfit_sentiment_main()

    # 3. Direct helper function map_continuous_score_to_class defense-in-depth check
    with pytest.raises(ValueError, match="Invalid sentiment threshold boundaries"):
        map_continuous_score_to_class(0.0, lower_threshold=0.3, upper_threshold=0.1)
