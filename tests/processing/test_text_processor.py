from __future__ import annotations

import pytest
from src.database import get_session, init_db
from src.models import Rede, Redner, Sitzung
from src.text_processing.processor import TextProcessor

def test_text_processor_split_raw_text():
    processor = TextProcessor()

    raw = (
        "Präsident: Das Wort hat der Abgeordnete.\n"
        "Herr Präsident! Meine Damen und Herren!\n"
        "Wir wollen Fortschritt wagen. (Beifall bei der SPD und dem BÜNDNIS 90/DIE GRÜNEN)\n"
        "Sitzung unterbrochen.\n"
        "Aber das kostet Geld. (Zuruf von der CDU/CSU - Lachen)\n"
        "Vielen Dank."
    )

    clean, metadata, counts = processor.split_raw_text(raw)

    # Assert president instructions and stage directions are extracted to metadata
    assert "Präsident: Das Wort hat der Abgeordnete." in metadata
    assert "Sitzung unterbrochen." in metadata

    # Assert inline interjections in parentheses are removed from clean text
    assert "(Beifall bei der SPD und dem BÜNDNIS 90/DIE GRÜNEN)" in metadata
    assert "(Zuruf von der CDU/CSU - Lachen)" in metadata

    # Check that speech-style headers remain intact
    assert "Herr Präsident! Meine Damen und Herren!" in clean
    assert "Wir wollen Fortschritt wagen." in clean
    assert "Aber das kostet Geld." in clean
    assert "Vielen Dank." in clean

    # Check that removed interjections are not present in clean
    assert "Beifall bei der SPD" not in clean
    assert "Zuruf von der CDU" not in clean

    # Check counts
    assert counts["president"] == 1
    assert counts["applause"] == 1
    assert counts["heckling"] == 1
    assert counts["laughter"] == 1


def test_text_processor_normalize_text():
    processor = TextProcessor()

    raw = """  Umlauts:  ä,   ö,   ü,   ß.   Spacings: \t\t multiple   spaces.


 Done. """
    normalized = processor.normalize_text(raw)

    assert "ä, ö, ü, ß" in normalized
    assert "Spacings: multiple spaces." in normalized
    assert "\n\n" in normalized
    assert "\n\n\n" not in normalized


def test_text_processor_extreme_inputs():
    processor = TextProcessor()

    # Empty inputs
    processed_empty = processor.process("")
    assert processed_empty.clean_text == ""
    assert processed_empty.normalized_text == ""
    assert processed_empty.metadata_text == ""
    assert processed_empty.metrics.char_count_raw == 0

    # Extrem size inputs
    large_text = "Redebeitrag.\n" * 10000
    processed_large = processor.process(large_text)
    assert processed_large.metrics.char_count_raw == len(large_text)
    assert processed_large.metrics.char_count_clean > 0

    # Payload exceeding 500,000 character limit
    too_large_text = "Redebeitrag.\n" * 45000  # 45000 * 13 = 585,000 chars
    with pytest.raises(ValueError, match="Input text exceeds the maximum safe limit of 500,000 characters."):
        processor.process(too_large_text)

    with pytest.raises(ValueError, match="Input text exceeds the maximum safe limit of 500,000 characters."):
        processor.split_raw_text(too_large_text)


def test_text_processor_unicode_edge_cases():
    processor = TextProcessor()

    # Unicode accents, non-breaking spaces, hyphen variants
    unicode_input = "Café\u00a0de Paris – we have \u2014 hyphens."
    processed = processor.process(unicode_input)
    assert "Café" in processed.normalized_text
    # Non-breaking space should be normalized to standard space
    assert "Café de Paris" in processed.normalized_text


def test_migration_cli_dry_run_and_help(tmp_path, monkeypatch):
    # Isolated DB configuration for CLI migration test
    db_file = tmp_path / "test_migration_cli.db"
    db_url = f"sqlite:///{db_file}"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", db_url)

    from src.database import reset_db_state
    reset_db_state()
    init_db()

    # Run the migration CLI using python subprocess or directly via importing main
    # Here we import and invoke main with mocked CLI arguments to test dry-run, stats, force, and resume features
    import sys
    from scripts.migrate_text_processing import main

    # Insert a dummy speech that is unmigrated
    with get_session() as session:
        # Seed test sitzung and redner
        from datetime import date
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2023, 1, 1))
        redner = Redner(vorname="Max", nachname="Mustermann", fraktion="CDU/CSU")
        session.add_all([sitzung, redner])
        session.commit()

        speech = Rede(
            sitzung_id=sitzung.sitzungs_id,
            redner_id=redner.redner_id,
            text="Präsident: Hallo.\nIch rede hier. (Beifall)",
            wortanzahl=4,
            raw_text=None, # unmigrated
            processing_version=0
        )
        session.add(speech)
        session.commit()

    # Mock CLI arguments to do a dry-run first
    monkeypatch.setattr(sys, "argv", ["migrate_text_processing.py", "--dry-run", "--verbose"])
    main()

    # Check that speech is still unmigrated after dry-run
    with get_session() as session:
        s = session.query(Rede).first()
        assert s.raw_text is None
        assert s.processing_version == 0

    # Now run for real with --limit 1
    monkeypatch.setattr(sys, "argv", ["migrate_text_processing.py", "--limit", "1", "--verbose"])
    main()

    processor = TextProcessor()
    # Verify the speech has been successfully migrated
    with get_session() as session:
        s = session.query(Rede).first()
        assert s.raw_text == "Präsident: Hallo.\nIch rede hier. (Beifall)"
        assert s.clean_text == "Ich rede hier."
        assert s.metadata_text == "Präsident: Hallo.\n(Beifall)"
        assert s.processing_version == processor.version
        assert s.count_applause == 1
        assert s.count_president == 1
        assert s.char_count_raw > 0
        assert s.char_count_clean > 0
        assert s.char_count_metadata > 0

    # Test resume option (should process nothing as it's already migrated)
    monkeypatch.setattr(sys, "argv", ["migrate_text_processing.py", "--resume", "--verbose"])
    main()

    # Test force option (should re-migrate everything)
    monkeypatch.setattr(sys, "argv", ["migrate_text_processing.py", "--force", "--verbose"])
    main()

    # Cleanup
    reset_db_state()


def test_stylometry_engine_batching_and_ner_disabled():
    from src.nlp import StylometryEngine
    engine = StylometryEngine()

    # 1. Assert NER is in fact disabled on the spacy model
    nlp = engine._ensure_spacy()
    assert "ner" not in nlp.pipe_names, "SpaCy NER component should be disabled."

    # 2. Test batch extraction
    texts = [
        "Das ist der erste Satz.",
        "Und das ist ein zweiter, längerer Satz mit mehr Wörtern.",
        ""
    ]

    batch_features = engine.extract_features_batch(texts, batch_size=2)
    assert len(batch_features) == 3

    # Ensure empty string yields standard zero metrics
    assert batch_features[2]["avg_sentence_length"] == 0.0
    assert batch_features[2]["type_token_ratio"] == 0.0

    # First and second texts should have positive sentence metrics
    assert batch_features[0]["avg_sentence_length"] > 0.0
    assert batch_features[1]["avg_sentence_length"] > 0.0
    assert isinstance(batch_features[0]["punctuation_freq"], dict)


def test_extended_stylometry_features():
    from src.nlp import StylometryEngine
    engine = StylometryEngine()

    # Test Syllable Counter
    assert engine._count_syllables("Schnee") == 1
    assert engine._count_syllables("Liebe") == 2
    assert engine._count_syllables("Bundestagsabgeordneter") >= 5
    assert engine._count_syllables("") == 0

    text = (
        "Herr Präsident! Meine Damen und Herren! "
        "Wir müssen heute über den Klimaschutz debattieren. Es ist unsere Pflicht. "
        "Wir wollen Fortschritt wagen und die Wirtschaft stärken! "
        "Ist das nicht absolut offensichtlich? (Beifall)"
    )

    feats = engine.extract_features(text)

    # 1. Readability
    assert feats["readability_lix"] > 0.0
    assert feats["readability_wiener_sachtext"] != 0.0
    assert feats["readability_flesch"] != 0.0

    # 2. Lexical diversity helper indices
    assert feats["root_ttr"] > 0.0
    assert feats["corrected_ttr"] > 0.0
    assert feats["mtld"] >= 0.0
    assert feats["hdd"] >= 0.0
    assert feats["hapax_legomena_ratio"] >= 0.0
    assert feats["hapax_dislegomena_ratio"] >= 0.0

    # 3. Pronoun ratios
    assert feats["pron_wir_ratio"] >= 0.0
    assert feats["pron_ich_ratio"] >= 0.0

    # 4. POS ratios
    assert feats["pos_noun_ratio"] > 0.0
    assert feats["pos_verb_ratio"] > 0.0
    assert feats["pos_adj_ratio"] >= 0.0

    # 5. Punctuation ratio checks
    assert feats["punct_period_ratio"] > 0.0
    assert feats["punct_question_ratio"] > 0.0
    assert feats["punct_exclamation_ratio"] > 0.0

    # 6. Structure & Rhetoric
    assert feats["num_paragraphs"] >= 1
    assert feats["sentence_paragraph_ratio"] > 0.0
    assert feats["rhetorical_question_ratio"] > 0.0
    assert feats["exclamatory_sentence_ratio"] > 0.0

    # Match batch vs single extraction
    single = engine.extract_features(text)
    batch = engine.extract_features_batch([text], batch_size=1)[0]

    for key in single.keys():
        if key == "punctuation_freq":
            assert single[key] == batch[key]
        else:
            assert abs(single[key] - batch[key]) < 1e-9


def test_stylometry_empty_inputs_handling():
    from src.nlp import StylometryEngine
    engine = StylometryEngine()

    feats = engine.extract_features("")
    assert feats["avg_sentence_length"] == 0.0
    assert feats["type_token_ratio"] == 0.0
    assert feats["mtld"] == 0.0
    assert feats["hdd"] == 0.0
    assert feats["pos_noun_ratio"] == 0.0
    assert feats["readability_flesch"] == 0.0


def test_wiener_sachtextformel_mathematical_correctness():
    from src.nlp import StylometryEngine
    engine = StylometryEngine()

    # Case 1: All monosyllabic words
    # Text: "Ich bin hier."
    # Tokens: ["ich", "bin", "hier"] (len=3)
    # ms_val: 0.0 (no word >= 3 syllables)
    # w6_val: 0.0 (no word > 6 chars)
    # es_val: 100.0 (all words are 1 syllable)
    # avg_sentence_length: 3.0
    # Expected WSTF: 0.1935 * 0 + 0.1672 * 3.0 + 0.1297 * 0 - 0.0327 * 100.0 - 0.875 = 0.5016 - 3.27 - 0.875 = -3.6434
    feats_mono = engine.extract_features("Ich bin hier.")
    assert feats_mono["readability_wiener_sachtext"] == pytest.approx(-3.6434, abs=1e-4)

    # Case 2: All words longer than 6 letters, multi-syllabic, no monosyllabic
    # Text: "Präsidenten debattierten." (assuming "präsidenten" -> 4 syllables, "debattierten" -> 4 syllables, both length > 6)
    # Tokens: ["präsidenten", "debattierten"] (len=2)
    # ms_val: 100.0
    # w6_val: 100.0
    # es_val: 0.0
    # avg_sentence_length: 2.0
    # Expected WSTF: 0.1935 * 100.0 + 0.1672 * 2.0 + 0.1297 * 100.0 - 0.0327 * 0.0 - 0.875 = 19.35 + 0.3344 + 12.97 - 0.875 = 31.7794
    feats_long = engine.extract_features("Präsidenten debattierten.")
    assert feats_long["readability_wiener_sachtext"] == pytest.approx(31.7794, abs=1e-4)


def test_wiener_sachtextformel_edge_cases():
    from src.nlp import StylometryEngine
    engine = StylometryEngine()

    # 1. Empty Text
    feats_empty = engine.extract_features("")
    assert feats_empty["readability_wiener_sachtext"] == 0.0

    # 2. Punctuation-only Text
    feats_punct = engine.extract_features("!!! ??? ...")
    assert feats_punct["readability_wiener_sachtext"] == 0.0

    # 3. One-word Text
    # Text: "Ja."
    # Tokens: ["ja"] (len=1)
    # ms_val: 0.0, w6_val: 0.0, es_val: 100.0, avg_sentence_length: 1.0
    # Expected WSTF: 0.1935 * 0 + 0.1672 * 1.0 + 0.1297 * 0 - 0.0327 * 100.0 - 0.875 = 0.1672 - 3.27 - 0.875 = -3.9778
    feats_one = engine.extract_features("Ja.")
    assert feats_one["readability_wiener_sachtext"] == pytest.approx(-3.9778, abs=1e-4)

    # 4. Mixed realistic text
    # Text: "Wir müssen heute den sehr komplexen Bericht debattieren."
    # Tokens: ["wir", "müssen", "heute", "den", "sehr", "komplexen", "bericht", "debattieren"] (len=8)
    # Words with >= 3 syllables: "komplexen" (kom-ple-xen -> 3), "debattieren" (de-bat-tie-ren -> 4). Total 2 words. ms_val = 2/8 * 100 = 25.0
    # Words with exactly 1 syllable: "wir", "den", "sehr". Total 3 words. es_val = 3/8 * 100 = 37.5
    # Words with length > 6: "komplexen" (9), "bericht" (7), "debattieren" (12). Total 3 words. w6_val = 3/8 * 100 = 37.5
    # Sentence length: 8.0
    # Expected WSTF: 0.1935 * 25.0 + 0.1672 * 8.0 + 0.1297 * 37.5 - 0.0327 * 37.5 - 0.875
    # = 4.8375 + 1.3376 + 4.86375 - 1.22625 - 0.875 = 8.9376
    feats_mixed = engine.extract_features("Wir müssen heute den sehr komplexen Bericht debattieren.")
    assert feats_mixed["readability_wiener_sachtext"] == pytest.approx(8.9376, abs=1e-4)
    
    # 5. Text with numbers
    # Text: "Ich bin 123."
    # With our fix, "123" is excluded from both numerator and denominator of ms_val and es_val.
    # Tokens: ["ich", "bin"] (len=2)
    # Words with >= 3 syllables: 0. ms_val = 0.0
    # Words with exactly 1 syllable: "ich", "bin". Total 2 words. es_val = 100.0
    # Words with length > 6: 0. w6_val = 0.0
    # Sentence length: 2.0 (since 123 is excluded from tokens)
    # Expected WSTF: 0.1935 * 0 + 0.1672 * 2.0 + 0.1297 * 0 - 0.0327 * 100.0 - 0.875 = 0.3344 - 3.27 - 0.875 = -3.8106
    feats_num = engine.extract_features("Ich bin 123.")
    assert feats_num["readability_wiener_sachtext"] == pytest.approx(-3.8106, abs=1e-4)


def test_wiener_sachtextformel_numeric_token_symmetry():
    from src.nlp import StylometryEngine
    engine = StylometryEngine()

    # Text containing numbers / numerical tokens: "1 bin"
    # Under previous buggy implementation:
    #   "1" (like_num=True) is counted in words_1_syllable but excluded from num_tokens.
    #   es_val = (2 * 100) / 1 = 200.0% (domain violation!)
    # Under corrected implementation:
    #   "1" is excluded from both numerator and denominator.
    #   num_tokens = 1 ("bin"), words_1_syllable = 1 ("bin").
    #   es_val = 100.0% (exact, stable, within bounds!)
    #   ms_val = 0.0%
    #   w6_val = 0.0%
    #   avg_sentence_length = 1.0
    #   Expected WSTF: 0.1935 * 0 + 0.1672 * 1.0 + 0.1297 * 0 - 0.0327 * 100.0 - 0.875 = 0.1672 - 3.27 - 0.875 = -3.9778
    feats = engine.extract_features("1 bin")
    assert feats["readability_wiener_sachtext"] == pytest.approx(-3.9778, abs=1e-4)
