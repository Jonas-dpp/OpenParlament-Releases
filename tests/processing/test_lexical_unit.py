
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import Base, Sitzung, Redner, Rede
from src.analytics import LexicalAnalysis
from datetime import date

def test_lexical_analysis_metrics():
    # Setup in-memory DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed data
    sz = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2023, 1, 1), gesamtwortzahl=100)
    r1 = Redner(vorname="Olaf", nachname="Scholz", fraktion="SPD")
    session.add_all([sz, r1])
    session.commit()

    text = "Dies ist ein Test. Wir testen die lexikalische Vielfalt und Dichte. Ein Test ist gut."
    # Total tokens (simplified): Dies, ist, ein, Test, Wir, testen, die, lexikalische, Vielfalt, und, Dichte, Ein, Test, ist, gut (15)
    # Types: dies, ist, ein, test, wir, testen, die, lexikalische, vielfalt, und, dichte, gut (12)
    # Content words (Noun, Verb, Adj, Adv): Test (N), testen (V), lexikalische (Adj), Vielfalt (N), Dichte (N), Test (N), gut (Adj) -> 7 approx

    rd = Rede(sitzung_id=sz.sitzungs_id, redner_id=r1.redner_id, text=text, wortanzahl=15)
    session.add(rd)
    session.commit()

    analyzer = LexicalAnalysis(session)
    metrics = analyzer.calculate_metrics(text)

    assert metrics["total_tokens"] > 0
    assert metrics["ttr"] > 0
    assert metrics["lexical_density"] > 0
    assert "msttr" in metrics

    # Test run
    df = analyzer.run(20)
    assert not df.empty
    assert "fraktion" in df.columns
    assert df.iloc[0]["fraktion"] == "SPD"
    assert df.iloc[0]["ttr"] > 0

    session.close()


def test_lexical_analysis_metrics_segment_size_guard():
    # Setup in-memory DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    analyzer = LexicalAnalysis(session)
    text = "Dies ist ein Test. Wir testen die lexikalische Vielfalt und Dichte. Ein Test ist gut."

    # Test with non-positive segment_size (0 and negative)
    metrics_zero = analyzer.calculate_metrics(text, segment_size=0)
    metrics_neg = analyzer.calculate_metrics(text, segment_size=-50)

    # Verify both execute without raising errors and fall back to standard calculations
    assert metrics_zero["total_tokens"] > 0
    assert metrics_zero["msttr"] > 0
    assert metrics_neg["total_tokens"] > 0
    assert metrics_neg["msttr"] > 0

    session.close()
