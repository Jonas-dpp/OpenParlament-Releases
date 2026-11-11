#!/usr/bin/env python3
"""
compute_standard_deviations.py

Computes sample standard deviations (s with Bessel's correction ddof=1)
and robust distribution metrics across multiple parliamentary metrics in openparlament.db:
1. Interjection sentiment overall and across electoral terms (WP 18-21) with strict clipping & audit.
2. Speech sentiment across factions and electoral terms.
3. Speech complexity metrics (WSTF, Flesch FRE, ASL, TTR) across factions.

Output is logged and saved to data/standard_deviations_report.json for audit.
"""

import argparse
import json
import logging
import os
import sqlite3
import time
import pandas as pd

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("compute_standard_deviations")


def compute_metrics(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    results = {}

    # 1. Zwischenruf Sentiment Overall & per WP (mit Clipping & Audit-Metriken)
    df_zr = pd.read_sql_query("""
        SELECT z.sentiment_score, s.wahlperiode, z.text
        FROM zwischenrufe z
        JOIN reden r ON z.rede_id = r.rede_id
        JOIN sitzungen s ON r.sitzung_id = s.sitzungs_id
        WHERE z.sentiment_score IS NOT NULL
    """, conn)

    df_zr["sentiment_score"] = pd.to_numeric(df_zr["sentiment_score"], errors="coerce")
    df_zr["sentiment_score_clipped"] = df_zr["sentiment_score"].clip(-1.0, 1.0)

    def get_distribution_stats(series):
        clean_s = series.dropna()
        if len(clean_s) == 0:
            return {"count": 0, "mean": 0.0, "std_s": 0.0, "median": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": int(len(clean_s)),
            "mean": float(round(clean_s.mean(), 4)),
            "std_s": float(round(clean_s.std(ddof=1), 4)) if len(clean_s) > 1 else 0.0,
            "median": float(round(clean_s.median(), 4)),
            "iqr": float(round(clean_s.quantile(0.75) - clean_s.quantile(0.25), 4)),
            "min": float(round(clean_s.min(), 4)),
            "max": float(round(clean_s.max(), 4)),
        }

    results["zwischenruf_sentiment_overall"] = get_distribution_stats(df_zr["sentiment_score_clipped"])

    # Parser-Audit: Extremwerte abspeichern zur händischen Validierung
    top_pos = df_zr.nlargest(5, "sentiment_score")[["sentiment_score", "text"]].to_dict(orient="records")
    top_neg = df_zr.nsmallest(5, "sentiment_score")[["sentiment_score", "text"]].to_dict(orient="records")
    results["zwischenruf_audit_extremes"] = {"top_positive": top_pos, "top_negative": top_neg}

    zr_per_wp = {}
    for wp, group in df_zr.groupby("wahlperiode"):
        zr_per_wp[f"WP_{wp}"] = get_distribution_stats(group["sentiment_score_clipped"])
    results["zwischenruf_sentiment_per_wp"] = zr_per_wp

    # 2. Speech Sentiment by Faction
    df_reden = pd.read_sql_query("""
        SELECT r.sentiment_score, red.fraktion, s.wahlperiode
        FROM reden r
        JOIN redner red ON r.redner_id = red.redner_id
        JOIN sitzungen s ON r.sitzung_id = s.sitzungs_id
        WHERE r.sentiment_score IS NOT NULL AND red.fraktion IS NOT NULL AND red.fraktion != ''
    """, conn)

    df_reden["sentiment_score"] = pd.to_numeric(df_reden["sentiment_score"], errors="coerce")
    df_reden["sentiment_score_clipped"] = df_reden["sentiment_score"].clip(-1.0, 1.0)
    df_reden["fraktion_clean"] = df_reden["fraktion"].str.replace("\xa0", " ")

    reden_per_faction = {}
    for fraktion, group in df_reden.groupby("fraktion_clean"):
        if len(group) >= 50:
            reden_per_faction[fraktion] = get_distribution_stats(group["sentiment_score_clipped"])
    results["speech_sentiment_per_faction"] = reden_per_faction

    # 3. Sample Speech Complexity (using src.nlp.StylometryEngine if available)
    try:
        from src.nlp import StylometryEngine
        engine = StylometryEngine()

        df_sample = pd.read_sql_query("""
            SELECT r.rede_id, r.clean_text, r.text, red.fraktion, s.wahlperiode
            FROM reden r
            JOIN redner red ON r.redner_id = red.redner_id
            JOIN sitzungen s ON r.sitzung_id = s.sitzungs_id
            WHERE red.fraktion IS NOT NULL AND red.fraktion != ''
            ORDER BY r.rede_id
            LIMIT 1500
        """, conn)

        total_sample = len(df_sample)
        if total_sample > 0:
            logger.info("Extracting speech complexity features for %d sample speeches...", total_sample)
        start_time = time.time()
        log_step = max(10, total_sample // 20)

        comp_records = []
        for i, row in enumerate(df_sample.itertuples(index=False), start=1):
            txt = getattr(row, "clean_text", None) or getattr(row, "text", None) or ""
            if len(txt) > 200:
                f = engine.extract_features(txt)
                fraktion_val = getattr(row, "fraktion", "") or ""
                comp_records.append({
                    "fraktion": fraktion_val.replace("\xa0", " "),
                    "wstf": f.get("readability_wiener_sachtext", 0.0),
                    "flesch": f.get("readability_flesch", 0.0),
                    "asl": f.get("avg_sentence_length", 0.0),
                    "ttr": f.get("type_token_ratio", 0.0),
                })

            if total_sample > 0 and (i % log_step == 0 or i == total_sample):
                elapsed = time.time() - start_time
                eta = (elapsed / i) * (total_sample - i)
                pct = int((i / total_sample) * 100)
                eta_fmt = time.strftime("%M:%S", time.gmtime(eta)) if eta < 3600 else time.strftime("%H:%M:%S", time.gmtime(eta))
                logger.info(
                    f"SpeechComplexityExtraction: {i}/{total_sample} | {pct}% | ETA: {eta_fmt}"
                )

        df_comp = pd.DataFrame(comp_records)
        comp_per_faction = {}
        if not df_comp.empty and "fraktion" in df_comp.columns:
            for fraktion, group in df_comp.groupby("fraktion"):
                if len(group) >= 10:
                    comp_per_faction[fraktion] = {
                        "count": int(len(group)),
                        "wstf": {"mean": float(round(group["wstf"].mean(), 2)), "std_s": float(round(group["wstf"].std(ddof=1), 2))},
                        "flesch": {"mean": float(round(group["flesch"].mean(), 2)), "std_s": float(round(group["flesch"].std(ddof=1), 2))},
                        "asl": {"mean": float(round(group["asl"].mean(), 2)), "std_s": float(round(group["asl"].std(ddof=1), 2))},
                        "ttr": {"mean": float(round(group["ttr"].mean(), 3)), "std_s": float(round(group["ttr"].std(ddof=1), 3))},
                    }
        results["speech_complexity_per_faction_sample"] = comp_per_faction
    except Exception as e:
        logger.warning("Speech complexity extraction skipped or failed: %s", e)
        results["speech_complexity_error"] = str(e)

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute robust standard deviations for OpenParlament corpus.")
    parser.add_argument("--db", default="./data/openparlament.db", help="Path to SQLite database")
    parser.add_argument("--output", default="./data/standard_deviations_report.json", help="Path to output JSON")
    args = parser.parse_args()

    logger.info("Computing robust standard deviations and distribution metrics...")
    results = compute_metrics(args.db)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Robust report successfully saved to %s", args.output)


if __name__ == "__main__":
    main()
