"""Unit tests for the 6 advanced statistical methods implemented in src/analytics.py."""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.analytics import (
    FramingAnalysis,
    InteraktionsNetzwerk,
    RednerVergleich,
    SpeechComplexityAnalysis,
    StylometryAnalyzer,
)


def test_kruskal_wallis_complexity_test():
    sca = SpeechComplexityAnalysis(MagicMock())
    rng = np.random.default_rng(42)

    # Create dummy DataFrame with non-normal complexity scores per faction
    df = pd.DataFrame(
        {
            "fraktion": ["SPD"] * 20 + ["CDU/CSU"] * 20 + ["AfD"] * 20,
            "wstf_score": (
                list(rng.normal(8.0, 1.0, 20))
                + list(rng.normal(12.0, 1.0, 20))
                + list(rng.normal(6.0, 1.0, 20))
            ),
        }
    )

    res = sca.kruskal_wallis_complexity_test(df, metric="wstf_score", group_col="fraktion")
    assert "statistic" in res
    assert "p_value" in res
    assert res["is_significant"] is True
    assert len(res["groups"]) == 3

    # Empty / Edge Cases
    empty_res = sca.kruskal_wallis_complexity_test(pd.DataFrame())
    assert empty_res["statistic"] == 0.0
    assert empty_res["is_significant"] is False


def test_logistic_regression_faction_classifier():
    sa = StylometryAnalyzer(MagicMock())
    rng = np.random.default_rng(42)

    # Create dummy DataFrame for faction classification
    df = pd.DataFrame(
        {
            "fraktion": ["SPD"] * 15 + ["CDU/CSU"] * 15,
            "wstf_score": list(rng.normal(8.0, 0.5, 15)) + list(rng.normal(12.0, 0.5, 15)),
            "sentence_length": list(rng.normal(12, 1, 15)) + list(rng.normal(22, 1, 15)),
        }
    )

    res = sa.logistic_regression_faction_classifier(df, target_col="fraktion")
    assert "accuracy" in res
    assert res["accuracy"] > 0.8
    assert "coefficients" in res
    assert not res["coefficients"].empty
    assert "SPD" in res["classes"] or "CDU/CSU" in res["classes"]

    # Empty edge case
    empty_res = sa.logistic_regression_faction_classifier(pd.DataFrame())
    assert empty_res["accuracy"] == 0.0


def test_chi2_independence_test():
    fa = FramingAnalysis(MagicMock())

    # Create a 2x2 contingency matrix
    contingency = pd.DataFrame(
        {"SPD": [100, 10], "AfD": [10, 100]},
        index=["Klimaschutz", "Migration"],
    )

    res = fa.chi2_independence_test(contingency)
    assert "chi2" in res
    assert res["chi2"] > 0
    assert res["is_significant"] is True
    assert res["dof"] == 1

    # Empty edge case
    empty_res = fa.chi2_independence_test(pd.DataFrame())
    assert empty_res["chi2"] == 0.0
    assert empty_res["is_significant"] is False


def test_hypergeometric_keyword_enrichment():
    fa = FramingAnalysis(MagicMock())

    # Test term enrichment
    res = fa.hypergeometric_keyword_enrichment(
        term_count_in_subcorpus=50,
        subcorpus_total_words=1000,
        term_count_in_global_corpus=100,
        global_corpus_total_words=100000,
    )

    assert "p_value" in res
    assert res["p_value"] < 0.001
    assert res["enrichment_factor"] > 10.0
    assert res["is_significant"] is True

    # Invalid input edge case
    invalid_res = fa.hypergeometric_keyword_enrichment(0, 0, 0, 0)
    assert invalid_res["p_value"] == 1.0


def test_ergm_attack_network_analysis():
    inet = InteraktionsNetzwerk(MagicMock())

    # Create dummy adjacency matrix (rows: speaker, cols: interruptor)
    factions = ["CDU/CSU", "SPD", "AfD", "DIE LINKE"]
    adj_matrix = pd.DataFrame(
        [
            [0, 5, 25, 10],  # CDU speaker interjected by AfD/LINKE
            [5, 0, 20, 12],  # SPD speaker interjected by AfD/LINKE
            [2, 2, 0, 5],    # AfD speaker
            [1, 1, 4, 0],    # LINKE speaker
        ],
        index=factions,
        columns=factions,
    )

    res = inet.ergm_attack_network_analysis(
        adj_matrix,
        government_factions={"CDU/CSU", "SPD"},
        opposition_factions={"AfD", "DIE LINKE"},
    )

    assert "log_odds" in res
    assert "odds_ratio" in res
    assert res["odds_ratio"] > 1.0 or res["is_significant"] is True
    assert "density" in res
    assert "reciprocity" in res

    # Non-square matrix edge case (e.g., 4 rows x 6 columns)
    factions_rows = ["CDU/CSU", "SPD", "AfD", "DIE LINKE"]
    factions_cols = ["CDU/CSU", "SPD", "AfD", "DIE LINKE", "FDP", "BSW"]
    non_square_adj = pd.DataFrame(
        [
            [0, 5, 25, 10, 2, 1],
            [5, 0, 20, 12, 1, 0],
            [2, 2, 0, 5, 0, 0],
            [1, 1, 4, 0, 0, 0],
        ],
        index=factions_rows,
        columns=factions_cols,
    )
    non_square_res = inet.ergm_attack_network_analysis(
        non_square_adj,
        government_factions={"CDU/CSU", "SPD", "FDP"},
        opposition_factions={"AfD", "DIE LINKE", "BSW"},
    )
    assert "log_odds" in non_square_res
    assert non_square_res["p_value"] <= 1.0
    assert "density" in non_square_res

    # Empty matrix edge case
    empty_res = inet.ergm_attack_network_analysis(pd.DataFrame())
    assert empty_res["p_value"] == 1.0


def test_cosine_similarity_permutation_test():
    rv = RednerVergleich(MagicMock())

    vec1 = np.array([1.0, 2.0, 3.0, 0.5, 0.0])
    vec2 = np.array([1.1, 1.9, 3.1, 0.4, 0.1])

    res = rv.cosine_similarity_permutation_test(vec1, vec2, num_permutations=200, random_state=42)

    assert "cosine_similarity" in res
    assert res["cosine_similarity"] > 0.95
    assert "p_value" in res
    # Phipson & Smyth (2010): p-value = (k + 1) / (B + 1) > 0.0 (strictly non-zero)
    assert 0.0 < res["p_value"] < 0.05
    assert res["is_significant"] is True

    # Test non-positive num_permutations falls back to default
    fallback_res = rv.cosine_similarity_permutation_test(vec1, vec2, num_permutations=-10, random_state=42)
    assert 0.0 < fallback_res["p_value"] < 0.05

    # Empty / Mismatched vector edge case
    empty_res = rv.cosine_similarity_permutation_test([], [1, 2, 3])
    assert empty_res["cosine_similarity"] == 0.0
