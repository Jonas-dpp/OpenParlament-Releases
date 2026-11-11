import pytest
import pandas as pd
from datetime import date
from src.database import get_session, init_db, drop_db
from src.models import Rede, Redner, Sitzung
from src.analytics import FramingAnalysis

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

@pytest.fixture(autouse=True)
def _use_in_memory_db(monkeypatch, tmp_path):
    """Redirect all DB access to a per-test temporary SQLite file.

    Using a file-backed SQLite DB avoids multiple-connection semantics
    that break tests when using ``sqlite:///:memory:`` across separate
    SQLAlchemy sessions. Use the public ``reset_db_state`` helper rather
    than poking internal module attributes.
    """
    db_file = tmp_path / "test_framing.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file}")

    import src.database as db_module
    db_module.reset_db_state()
    init_db()
    yield
    drop_db()
    db_module.reset_db_state()


def test_load_documents_grouping():
    """Verify that speeches are loaded, canonicalised, and joined per faction."""
    with get_session() as session:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 10, 26))
        redner1 = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
        redner2 = Redner(vorname="Lars", nachname="Klingbeil", fraktion="SPD")
        redner3 = Redner(vorname="Friedrich", nachname="Merz", fraktion="CDU/CSU")
        
        rede1 = Rede(sitzung=sitzung, redner=redner1, text="Das ist Rede eins.", wortanzahl=4)
        rede2 = Rede(sitzung=sitzung, redner=redner2, text="Das ist Rede zwei.", wortanzahl=4)
        rede3 = Rede(sitzung=sitzung, redner=redner3, text="Das ist Rede drei.", wortanzahl=4)
        
        session.add_all([sitzung, redner1, redner2, redner3, rede1, rede2, rede3])
        session.commit()

    with get_session() as session:
        fa = FramingAnalysis(session)
        df_docs = fa.load_documents(wahlperiode=20)
        
        assert len(df_docs) == 2
        
        spd_doc = df_docs[df_docs["fraktion"] == "SPD"].iloc[0]["text"]
        assert "Das ist Rede eins." in spd_doc
        assert "Das ist Rede zwei." in spd_doc
        
        cdu_doc = df_docs[df_docs["fraktion"] == "CDU/CSU"].iloc[0]["text"]
        assert "Das ist Rede drei." in cdu_doc


def test_preprocessing():
    """Verify preprocessing removes punctuation, stopwords, custom stopwords, and does lemmatization."""
    with get_session() as session:
        fa = FramingAnalysis(
            session,
            custom_stopwords={"antrag", "ausschuss", "tagesordnungspunkt"}
        )
        
        # Test input containing:
        # - punctuation and numbers ("!")
        # - spaCy stopword ("und", "ist")
        # - custom stopword ("Antrag", "ausschuss" - check case insensitivity)
        # - normal German nouns that should be lemmatized ("Abgeordneten")
        test_text = "Die Abgeordneten debattieren über den Antrag und den Ausschuss! Das ist Nummer 1."
        preprocessed = fa.preprocess(test_text)
        
        # "Abgeordneten" lemma is "Abgeordnete" (proper casing in German, capitalized noun)
        # "debattieren" lemma is "debattieren"
        assert "Abgeordnete" in preprocessed or "abgeordneter" in preprocessed.lower()
        assert "debattieren" in preprocessed
        
        # Stopwords and punctuation should be removed
        assert "und" not in preprocessed.lower()
        assert "ist" not in preprocessed.lower()
        assert "antrag" not in preprocessed.lower()
        assert "ausschuss" not in preprocessed.lower()
        assert "nummer" in preprocessed.lower()
        assert "1" not in preprocessed


def test_proper_nouns_filtering():
    """Verify that proper nouns (PROPN) can be filtered or retained."""
    with get_session() as session:
        # Test sentence with proper noun (Berlin, Scholz)
        test_text = "Berlin und Olaf Scholz sind in Deutschland."
        
        # 1. Filter proper nouns (default)
        fa_filter = FramingAnalysis(session, filter_propn=True)
        preprocessed_filtered = fa_filter.preprocess(test_text)
        assert "berlin" not in preprocessed_filtered.lower()
        assert "scholz" not in preprocessed_filtered.lower()
        assert "deutschland" not in preprocessed_filtered.lower()
        
        # 2. Keep proper nouns
        fa_keep = FramingAnalysis(session, filter_propn=False)
        preprocessed_kept = fa_keep.preprocess(test_text)
        assert "berlin" in preprocessed_kept.lower()
        assert "scholz" in preprocessed_kept.lower()
        assert "deutschland" in preprocessed_kept.lower()


def test_parliamentary_fluff_stopwords_filtering():
    """Verify that ritual parliamentary salutations and procedural fluff ('Kolleginnen und Kollegen', etc.) are removed."""
    with get_session() as session:
        fa = FramingAnalysis(session)
        sample = (
            "Sehr geehrte Frau Präsidentin! Liebe Kolleginnen und Kollegen! "
            "Meine sehr verehrten Damen und Herren! Herzlichen Dank für Ihre Aufmerksamkeit."
        )
        preprocessed = fa.preprocess(sample)
        # Should not contain any of the fluff tokens or lemmas
        for token in [
            "kollege", "kollegin", "kollegen", "kolleginnen",
            "präsidentin", "präsident",
            "dame", "damen", "herr", "herren",
            "dank", "aufmerksamkeit", "frau", "geehrt", "verehrt"
        ]:
            assert token not in preprocessed.lower().split()


def test_faction_self_references_and_other_party_preservation():
    """Verify that a faction's self-references are pruned while competitors' names are preserved."""
    with get_session() as session:
        fa = FramingAnalysis(session)
        sample = (
            "Die Grünen kritisieren die AfD, die SPD und die Union scharf für ihre Blockadehaltung."
        )

        # 1. Spoken by BÜNDNIS 90/DIE GRÜNEN:
        # 'Grünen' should be removed as self-reference, but 'AfD', 'SPD', 'Union' should stay
        prep_gruene = fa.preprocess(sample, fraktion="BÜNDNIS 90/DIE GRÜNEN")
        assert "grüne" not in prep_gruene.lower().split()
        assert "grünen" not in prep_gruene.lower().split()
        assert "grün" not in prep_gruene.lower().split()
        assert "afd" in prep_gruene.lower().split()
        assert "spd" in prep_gruene.lower().split()
        assert "union" in prep_gruene.lower().split()

        # 2. Spoken by SPD:
        # 'SPD' should be removed as self-reference, but 'Grünen', 'AfD', 'Union' should stay
        prep_spd = fa.preprocess(sample, fraktion="SPD")
        assert "spd" not in prep_spd.lower().split()
        assert "sozialdemokrat" not in prep_spd.lower().split()
        assert "afd" in prep_spd.lower().split()
        assert "union" in prep_spd.lower().split()
        assert any(t in prep_spd.lower().split() for t in ["grüne", "grünen", "grün"])

        # 3. Spoken by CDU/CSU:
        # 'Union' should be removed as self-reference, but 'AfD', 'SPD', 'Grünen' should stay
        prep_cdu = fa.preprocess(sample, fraktion="CDU/CSU")
        assert "union" not in prep_cdu.lower().split()
        assert "cdu" not in prep_cdu.lower().split()
        assert "csu" not in prep_cdu.lower().split()
        assert "afd" in prep_cdu.lower().split()
        assert "spd" in prep_cdu.lower().split()
        assert any(t in prep_cdu.lower().split() for t in ["grüne", "grünen", "grün"])


def test_compute_tfidf_synthetic():
    """Verify TF-IDF returns expected terms with highest scores on a small dataset."""
    with get_session() as session:
        # Using FramingAnalysis, run on synthetic documents.
        # We bypass load_documents and feed df directly to compute_tfidf.
        df_docs = pd.DataFrame([
            {"fraktion": "SPD", "text": "digitalisierung breitband glasfaser zukunft digitalisierung"},
            {"fraktion": "CDU/CSU", "text": "sicherheit bundeswehr verteidigung armee sicherheit"}
        ])
        
        fa = FramingAnalysis(session, min_df=1, max_df=1.0)
        tfidf_matrix, vectorizer, factions = fa.compute_tfidf(df_docs)
        
        df_top = fa.top_terms(tfidf_matrix, vectorizer, factions, n_terms=3)
        
        # SPD top term should be digitalisierung
        spd_terms = df_top[df_top["fraktion"] == "SPD"]
        assert spd_terms.iloc[0]["term"] == "digitalisierung"
        
        # CDU top term should be sicherheit
        cdu_terms = df_top[df_top["fraktion"] == "CDU/CSU"]
        assert cdu_terms.iloc[0]["term"] == "sicherheit"


def test_empty_dataset():
    """Verify that empty datasets are handled gracefully without crashing."""
    with get_session() as session:
        fa = FramingAnalysis(session)
        # Empty wahlperiode (no data in DB)
        df_terms, vectorizer = fa.run(wahlperiode=99)
        
        assert df_terms.empty
        assert "fraktion" in df_terms.columns
        assert "term" in df_terms.columns
        assert "score" in df_terms.columns
        
        # Fitting empty document list
        df_empty = pd.DataFrame(columns=["fraktion", "text"])
        tfidf_matrix, vec, factions = fa.compute_tfidf(df_empty)
        assert tfidf_matrix.shape[0] == 0


def test_missing_values():
    """Verify that speeches with missing text or missing factions do not crash analysis."""
    with get_session() as session:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 10, 26))
        redner1 = Redner(vorname="Olaf", nachname="Scholz", fraktion=None) # missing faction
        redner2 = Redner(vorname="Lars", nachname="Klingbeil", fraktion="SPD")
        redner3 = Redner(vorname="Friedrich", nachname="Merz", fraktion="CDU/CSU")
        
        rede1 = Rede(sitzung=sitzung, redner=redner1, text="Keine Fraktion.", wortanzahl=2)
        rede2 = Rede(sitzung=sitzung, redner=redner2, text=None, wortanzahl=0) # missing text
        rede3 = Rede(sitzung=sitzung, redner=redner3, text="Gültige Rede.", wortanzahl=2)
        
        session.add_all([sitzung, redner1, redner2, redner3, rede1, rede2, rede3])
        session.commit()

    with get_session() as session:
        fa = FramingAnalysis(session)
        df_docs = fa.load_documents(wahlperiode=20)
        
        # Only CDU/CSU has a valid faction AND valid text
        assert len(df_docs) == 1
        assert df_docs.iloc[0]["fraktion"] == "CDU/CSU"
        assert df_docs.iloc[0]["text"] == "Gültige Rede."


def test_db_framing_query():
    """Verify that the SQL query for precomputed framing works correctly."""
    from src.models import FramingResult
    from src.app import _cached_framing

    with get_session() as session:
        # Add dummy framing results
        results = [
            FramingResult(wahlperiode=20, fraktion="SPD", term="Digitalisierung", score=0.9),
            FramingResult(wahlperiode=20, fraktion="SPD", term="Zukunft", score=0.8),
            FramingResult(wahlperiode=20, fraktion="SPD", term="Innovation", score=0.7),
            FramingResult(wahlperiode=20, fraktion="CDU/CSU", term="Sicherheit", score=0.95),
            FramingResult(wahlperiode=19, fraktion="SPD", term="Altlast", score=0.5),
        ]
        session.add_all(results)
        session.commit()

    # Test fetching with limit
    df = _cached_framing(wahlperiode=20, n_terms=2)

    # Should have 2 terms for SPD and 1 for CDU/CSU
    assert len(df) == 3

    spd = df[df["fraktion"] == "SPD"]
    assert len(spd) == 2
    assert spd.iloc[0]["term"] == "Digitalisierung"
    assert spd.iloc[1]["term"] == "Zukunft"

    cdu = df[df["fraktion"] == "CDU/CSU"]
    assert len(cdu) == 1
    assert cdu.iloc[0]["term"] == "Sicherheit"

    # Test wahlperiode isolation
    df_19 = _cached_framing(wahlperiode=19)
    assert len(df_19) == 1
    assert df_19.iloc[0]["term"] == "Altlast"

    # Test segment-filtered query
    with get_session() as session:
        session.add_all([
            FramingResult(wahlperiode=20, fraktion="SPD", segment="speech", variant="global", term="redeterm", score=0.88),
            FramingResult(wahlperiode=20, fraktion="SPD", segment="interjection", variant="global", term="rufterm", score=0.92),
        ])
        session.commit()

    df_seg_sp = _cached_framing(wahlperiode=20, n_terms=5, segment="speech")
    assert any(df_seg_sp["term"] == "redeterm")
    assert not any(df_seg_sp["term"] == "rufterm")


# ─────────────────────────────────────────────────────────────────────────────
# Segmented TF-IDF tests (FramingAnalyzer)
# ─────────────────────────────────────────────────────────────────────────────

from src.analytics import FramingAnalyzer
from src.models import Zwischenruf


def test_framing_analyzer_load_segments():
    """Verify that FramingAnalyzer loads speeches, interjections and all segments correctly."""
    with get_session() as session:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=10, datum=date(2022, 1, 15))
        redner_spd = Redner(vorname="Lars", nachname="Klingbeil", fraktion="SPD")
        redner_cdu = Redner(vorname="Friedrich", nachname="Merz", fraktion="CDU/CSU")

        rede1 = Rede(sitzung=sitzung, redner=redner_spd, text="Wir investieren in moderne Infrastruktur.", wortanzahl=5)
        rede2 = Rede(sitzung=sitzung, redner=redner_cdu, text="Die Bundesregierung verfehlt ihre Haushaltsziele.", wortanzahl=5)

        zr1 = Zwischenruf(rede=rede1, fraktion="CDU/CSU", text="Das ist doch reine Illusion!", kategorie="Widerspruch")
        zr2 = Zwischenruf(rede=rede2, fraktion="SPD", text="Hören Sie doch erst einmal zu!", kategorie="Zwischenruf")

        session.add_all([sitzung, redner_spd, redner_cdu, rede1, rede2, zr1, zr2])
        session.commit()

    with get_session() as session:
        analyzer = FramingAnalyzer(session)

        # 1. Speeches segment
        df_speech = analyzer.load_segment_documents(20, segment="speech")
        assert len(df_speech) == 2
        assert "SPD" in df_speech["fraktion"].values
        assert "Infrastruktur" in df_speech[df_speech["fraktion"] == "SPD"].iloc[0]["text"]

        # 2. Interjections segment
        df_interj = analyzer.load_segment_documents(20, segment="interjection")
        assert len(df_interj) == 2
        assert "CDU/CSU" in df_interj["fraktion"].values
        assert "Illusion" in df_interj[df_interj["fraktion"] == "CDU/CSU"].iloc[0]["text"]

        # 3. All segment
        df_all = analyzer.load_segment_documents(20, segment="all")
        assert len(df_all) == 2
        spd_all = df_all[df_all["fraktion"] == "SPD"].iloc[0]["text"]
        assert "Infrastruktur" in spd_all
        assert "Hören Sie" in spd_all


def test_framing_analyzer_global_vs_segmented_variants():
    """Verify mathematical and operational behavior of Variant A (global) vs Variant B (segmented)."""
    docs_speech = pd.DataFrame([
        {"fraktion": "SPD", "text": "klimaschutz transformation energiewende industrie"},
        {"fraktion": "CDU/CSU", "text": "wirtschaftswachstum steuersenkung schuldenbremse haushalt"},
    ])
    docs_interj = pd.DataFrame([
        {"fraktion": "SPD", "text": "unsinn heuchelei widerspruch skandal"},
        {"fraktion": "CDU/CSU", "text": "zwischenfrage antwort ministerruecktritt peinlich"},
    ])

    analyzer = FramingAnalyzer(min_df=1, max_df=1.0)
    # Fit global model on all combined
    docs_all = pd.concat([docs_speech, docs_interj], ignore_index=True)
    analyzer.fit(docs_all, segment="all")

    # Variant A: Global IDF
    df_global_speech = analyzer.analyze_segment(segment="speech", variant="global", top_n=5, documents=docs_speech)
    assert not df_global_speech.empty
    assert (df_global_speech["variant"] == "global").all()
    assert (df_global_speech["segment"] == "speech").all()

    # Variant B: Segmented IDF
    df_seg_speech = analyzer.analyze_segment(segment="speech", variant="segmented", top_n=5, documents=docs_speech)
    assert not df_seg_speech.empty
    assert (df_seg_speech["variant"] == "segmented").all()


def test_framing_analyzer_compare_segments():
    """Verify comparison of terms between speeches and interjections."""
    docs_speech = pd.DataFrame([
        {"fraktion": "SPD", "text": "klimaschutz investition transformation"},
        {"fraktion": "CDU/CSU", "text": "haushalt schuldenbremse investition"},
    ])
    docs_interj = pd.DataFrame([
        {"fraktion": "SPD", "text": "unsinn heuchelei skandal"},
        {"fraktion": "CDU/CSU", "text": "ruecktritt heuchelei lautstark"},
    ])

    analyzer = FramingAnalyzer(min_df=1, max_df=1.0)

    # Compare speeches (a) and interjections (b) with global IDF
    df_comp = analyzer.compare_segments(
        segments=("speech", "interjection"),
        top_n=10,
        variant="global",
        documents_a=docs_speech,
        documents_b=docs_interj,
    )

    assert not df_comp.empty
    assert "term" in df_comp.columns
    assert "diff" in df_comp.columns
    assert "dominant_segment" in df_comp.columns

    # Speeches should have positive diff for speech terms (e.g. investition / transformation)
    speech_terms = df_comp[df_comp["dominant_segment"] == "speech"]["term"].tolist()
    assert any(t in speech_terms for t in ["klimaschutz", "transformation", "investition", "haushalt", "schuldenbremse"])

    # Interjections should have negative diff for heckling terms (e.g. unsinn / heuchelei)
    interj_terms = df_comp[df_comp["dominant_segment"] == "interjection"]["term"].tolist()
    assert any(t in interj_terms for t in ["unsinn", "heuchelei", "skandal", "ruecktritt"])


def test_framing_analysis_backward_compatibility():
    """Ensure legacy FramingAnalysis inherits from FramingAnalyzer and runs without regressions."""
    from src.analytics import FramingAnalysis

    with get_session() as session:
        sitzung = Sitzung(wahlperiode=20, sitzungsnr=12, datum=date(2022, 2, 1))
        redner = Redner(vorname="Robert", nachname="Habeck", fraktion="BÜNDNIS 90/DIE GRÜNEN")
        rede = Rede(sitzung=sitzung, redner=redner, text="Erneuerbare Energien sichern Wohlstand.", wortanzahl=5)
        session.add_all([sitzung, redner, rede])
        session.commit()

    with get_session() as session:
        legacy_fa = FramingAnalysis(session, min_df=1, max_df=1.0)
        df_docs = legacy_fa.load_documents(wahlperiode=20)
        assert len(df_docs) == 1
        assert "BÜNDNIS 90/DIE GRÜNEN" in df_docs["fraktion"].values

        terms_df, vec = legacy_fa.run(wahlperiode=20, n_terms=5)
        assert not terms_df.empty
        assert "term" in terms_df.columns
        assert "score" in terms_df.columns
