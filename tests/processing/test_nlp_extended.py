import pytest
from unittest.mock import patch
from src.nlp import (
    _canonicalise_fraktion,
    _load_party_normalization,
    SentimentEngine,
    ToneClassifier,
    AddresseeDetector,
    StylometryEngine,
    get_cached_spacy_model,
    get_cached_blank_spacy_model,
)


def test_canonicalise_fraktion_edge_cases():
    # Test line 300
    assert _canonicalise_fraktion(None) == ""
    assert _canonicalise_fraktion("") == ""
    assert _canonicalise_fraktion("   ") == "N/A"  # maps to N/A
    assert _canonicalise_fraktion("spd") == "SPD"
    assert _canonicalise_fraktion("cdu") == "CDU/CSU"


def test_load_party_normalization_exception():
    # Test line 50-51 exceptions in _load_party_normalization
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", side_effect=Exception("Read failure")):
        res = _load_party_normalization()
        assert res == {}


def test_engines_extra_kwargs(caplog):
    # Test line 333, 395, 458: extra kwargs logging
    import logging
    with caplog.at_level(logging.DEBUG):
        _ = SentimentEngine(extra_ignored_arg="test")
        _ = ToneClassifier(extra_ignored_arg="test")
        _ = AddresseeDetector(extra_ignored_arg="test")
    assert any("SentimentEngine.__init__ ignoring extra kwargs" in record.message for record in caplog.records)
    assert any("ToneClassifier.__init__ ignoring extra kwargs" in record.message for record in caplog.records)
    assert any("AddresseeDetector.__init__ ignoring extra kwargs" in record.message for record in caplog.records)


def test_engines_no_transformers():
    # Test line 356, 418, 540: RuntimeError when _TRANSFORMERS_AVAILABLE is False
    with patch("src.nlp._TRANSFORMERS_AVAILABLE", False):
        s_engine = SentimentEngine()
        with pytest.raises(RuntimeError, match="Transformers/Torch required."):
            s_engine._get_pipeline()
        t_engine = ToneClassifier()
        with pytest.raises(RuntimeError, match="Transformers/Torch required."):
            t_engine._get_pipeline()
        a_detector = AddresseeDetector(use_ner=True)
        with pytest.raises(RuntimeError, match="Transformers/Torch required."):
            a_detector._get_ner_pipeline()


def test_ner_pipeline_already_failed():
    # Test line 535: raise RuntimeError if _NER_FAILED is True
    a_detector = AddresseeDetector(use_ner=True)
    with patch("src.nlp._NER_FAILED", True):
        with pytest.raises(RuntimeError, match="NER pipeline already failed – skipping reload"):
            a_detector._get_ner_pipeline()


def test_ner_pipeline_fails_permanently_on_load():
    # Test line 557-560: pipeline sets _NER_FAILED to True on load exception
    a_detector = AddresseeDetector(use_ner=True)
    with patch("src.nlp._NER_FAILED", False), \
         patch("src.nlp._NER_PIPELINE", None), \
         patch("src.nlp.pipeline", side_effect=ValueError("Model load failed")):
        with pytest.raises(ValueError, match="Model load failed"):
            a_detector._get_ner_pipeline()
        import src.nlp
        assert src.nlp._NER_FAILED is True


def test_ner_detect_persons_exception_handling():
    # Test line 575: return [] when _get_ner_pipeline throws exception inside _ner_detect_persons
    a_detector = AddresseeDetector(use_ner=True)
    with patch.object(a_detector, "_get_ner_pipeline", side_effect=Exception("Pipeline call error")):
        res = a_detector._ner_detect_persons("Herr Scholz")
        assert res == []


def test_stylometry_engine_mtld_hdd_edge_cases():
    engine = StylometryEngine()
    # mtld on empty list (line 703)
    assert engine._mtld([]) == 0.0
    # mtld factor_count zero fallback
    # with tokens that are all identical so current_ttr remains 1.0
    assert engine._mtld(["word"] * 5) > 0.0
    # hdd empty or T < N (line 735)
    assert engine._hdd(["word"] * 5, N=42) == 0.0
    # hdd T - c < N (line 740)
    assert engine._hdd(["a", "b", "c"], N=3) == 3.0
    # hdd T - c >= N (line 740-745)
    assert engine._hdd(["a", "b", "c", "d"], N=2) > 0.0


def test_spacy_cache_loading():
    # Test line 661: get_cached_spacy_model without disable list
    nlp_model = get_cached_spacy_model("de_core_news_sm")
    assert nlp_model is not None
    # Test blank model
    nlp_blank = get_cached_blank_spacy_model("de")
    assert nlp_blank is not None


def test_stylometry_engine_extract_batch_empty_nlp():
    # Test line 1270: extract_features_batch when nlp is None
    engine = StylometryEngine()
    with patch.object(engine, "_ensure_spacy", return_value=None):
        res = engine.extract_features_batch(["test"], batch_size=2)
        assert len(res) == 1
        assert isinstance(res[0], dict)
        assert res[0]["readability_flesch"] == 0.0


def test_stylometry_engine_extreme_lexicon_ratios():
    engine = StylometryEngine()
    text = (
        "Herr Präsident! Sehr geehrte Kolleginnen und Kollegen! "
        "Bitte danke! Gehen Sie! Hört! Man sie ihnen 20% im Jahr 2026. "
        "Eins zwei drei. hätte wäre würde. : ; - \""
    )
    feats = engine.extract_features(text)
    assert feats is not None
    assert feats["punctuation_freq"]["?"] >= 0
    assert feats["pron_man_ratio"] >= 0.0
    assert feats["pron_sie_ratio"] >= 0.0
    assert feats["num_percent_ratio"] >= 0.0
    assert feats["num_year_ratio"] >= 0.0
    assert feats["num_words_ratio"] >= 0.0
    assert feats["politeness_ratio"] >= 0.0
    assert feats["direct_address_ratio"] >= 0.0
