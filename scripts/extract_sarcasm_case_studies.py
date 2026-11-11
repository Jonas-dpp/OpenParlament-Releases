#!/usr/bin/env python3
"""
extract_sarcasm_case_studies.py

Extracts top positive interjections (> 0.95) called by Opposition factions towards
Government speakers, along with direct speech textual context (5 sentences before
and 3 sentences after the interjection) and annotation fields for qualitative evaluation.

Outputs:
- CSV report: data/sarcasm_case_studies.csv
- Excel report: data/sarcasm_case_studies.xlsx
- Markdown report: data/sarcasm_case_studies.md
"""

import os
import sqlite3
import argparse
import logging
import time
import pandas as pd
import re
from typing import Tuple, Set, Optional, List

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("extract_sarcasm_case_studies")


# Government and opposition factions per Wahlperiode (aligned with OpenParlament analytical definitions)
GOV_OPP_FACTIONS_BY_WP = {
    0: ({"CDU/CSU", "SPD", "FDP"}, {"BÜNDNIS 90/DIE GRÜNEN", "DIE LINKE"}),
    1: ({"CDU/CSU", "FDP"}, {"SPD"}),
    2: ({"CDU/CSU", "FDP"}, {"SPD"}),
    3: ({"CDU/CSU"}, {"SPD", "FDP"}),
    4: ({"CDU/CSU", "FDP"}, {"SPD"}),
    5: ({"CDU/CSU", "SPD"}, {"FDP"}),
    6: ({"SPD", "FDP"}, {"CDU/CSU"}),
    7: ({"SPD", "FDP"}, {"CDU/CSU"}),
    8: ({"SPD", "FDP"}, {"CDU/CSU"}),
    9: ({"CDU/CSU", "FDP"}, {"SPD"}),
    10: ({"CDU/CSU", "FDP"}, {"SPD", "BÜNDNIS 90/DIE GRÜNEN"}),
    11: ({"CDU/CSU", "FDP"}, {"SPD", "BÜNDNIS 90/DIE GRÜNEN"}),
    12: ({"CDU/CSU", "FDP"}, {"SPD", "BÜNDNIS 90/DIE GRÜNEN", "DIE LINKE"}),
    13: ({"CDU/CSU", "FDP"}, {"SPD", "BÜNDNIS 90/DIE GRÜNEN", "DIE LINKE"}),
    14: ({"SPD", "BÜNDNIS 90/DIE GRÜNEN"}, {"CDU/CSU", "FDP", "DIE LINKE"}),
    15: ({"SPD", "BÜNDNIS 90/DIE GRÜNEN"}, {"CDU/CSU", "FDP"}),
    16: ({"CDU/CSU", "SPD"}, {"FDP", "DIE LINKE", "BÜNDNIS 90/DIE GRÜNEN"}),
    17: ({"CDU/CSU", "FDP"}, {"SPD", "DIE LINKE", "BÜNDNIS 90/DIE GRÜNEN"}),
    18: ({"CDU/CSU", "SPD"}, {"BÜNDNIS 90/DIE GRÜNEN", "DIE LINKE"}),
    19: ({"CDU/CSU", "SPD"}, {"AFD", "FDP", "BÜNDNIS 90/DIE GRÜNEN", "DIE LINKE"}),
    20: ({"SPD", "BÜNDNIS 90/DIE GRÜNEN", "FDP"}, {"CDU/CSU", "AFD", "DIE LINKE", "BSW"}),
    21: ({"CDU/CSU", "SPD"}, {"AFD", "BÜNDNIS 90/DIE GRÜNEN", "DIE LINKE", "BSW"}),
}


def canonicalise_faction(faction: Optional[str]) -> str:
    """Canonicalise political faction string representations."""
    if not faction or not isinstance(faction, str):
        return ""

    f = faction.strip()
    upper = f.upper()

    if "SPD" in upper:
        return "SPD"
    if "CDU" in upper or "CSU" in upper or "UNION" in upper:
        return "CDU/CSU"
    if "GRÜN" in upper or "GRUEN" in upper or "BÜNDNIS 90" in upper:
        return "BÜNDNIS 90/DIE GRÜNEN"
    if "FDP" in upper:
        return "FDP"
    if "AFD" in upper:
        return "AfD"
    if "LINKE" in upper:
        return "DIE LINKE"
    if "BSW" in upper:
        return "BSW"
    if "SSW" in upper:
        return "SSW"

    return f


def get_gov_opp_factions(wp: int) -> Tuple[Set[str], Set[str]]:
    """Returns canonical sets of government and opposition factions for a given Wahlperiode."""
    gov, opp = GOV_OPP_FACTIONS_BY_WP.get(wp, GOV_OPP_FACTIONS_BY_WP[max(GOV_OPP_FACTIONS_BY_WP.keys())])
    gov_canon = {canonicalise_faction(f) for f in gov}
    opp_canon = {canonicalise_faction(f) for f in opp}
    return gov_canon, opp_canon


def is_opposition_to_government(z_faction: str, speaker_faction: str, wp: int) -> bool:
    """Checks if an interjection originated from Opposition directed at a Government speaker."""
    z_canon = canonicalise_faction(z_faction)
    spk_canon = canonicalise_faction(speaker_faction)

    if not z_canon or not spk_canon:
        return False

    gov_factions, opp_factions = get_gov_opp_factions(wp)
    return (z_canon in opp_factions) and (spk_canon in gov_factions)


def split_into_sentences(text: str) -> List[str]:
    """Splits text into non-empty sentences preserving sentence punctuation."""
    if not text:
        return []

    # Clean multi-spaces and normalise linebreaks
    clean_t = re.sub(r"\s+", " ", text.strip())
    raw_sents = re.split(r"(?<=[.!?])\s+", clean_t)
    return [s.strip() for s in raw_sents if s.strip()]


def extract_context_from_speech(speech_text: str, interjection_text: str, before_count: int = 5, after_count: int = 3) -> Tuple[str, str]:
    """
    Extracts the 5 sentences of the main speaker before the interjection
    and 3 sentences after the interjection.

    Returns:
        Tuple[str, str]: (context_before_5_sents, context_after_3_sents)
    """
    if not speech_text:
        return "", ""

    # Clean raw text and remove interjection comment if present in speech_text
    clean_speech = speech_text
    if interjection_text and interjection_text in clean_speech:
        parts = clean_speech.split(interjection_text, 1)
        before_text = parts[0]
        after_text = parts[1] if len(parts) > 1 else ""

        sents_before = split_into_sentences(before_text)
        sents_after = split_into_sentences(after_text)

        context_before = " ".join(sents_before[-before_count:]) if sents_before else ""
        context_after = " ".join(sents_after[:after_count]) if sents_after else ""
        return context_before, context_after

    # Fallback: Exact interjection position could not be determined.
    logger.debug(
        "Could not locate interjection in speech text: %s",
        interjection_text[:100]
    )
    return "", ""


def extract_case_studies(
    db_path: str,
    limit: int = 50,
    min_sentiment: float = 0.95
) -> pd.DataFrame:
    """Queries SQLite DB and extracts opposition -> government positive extreme interjections with context."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not isinstance(min_sentiment, (int, float)) or isinstance(min_sentiment, bool) or min_sentiment < -1.0 or min_sentiment > 1.0:
        raise ValueError("min_sentiment must be a float between -1.0 and 1.0")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            z.ruf_id AS interjection_id,
            z.text AS zwischenruf_text,
            z.sentiment_score AS sentiment_score,
            z.fraktion AS zwischenruf_fraktion,
            red.vorname || ' ' || red.nachname AS hauptredner,
            red.fraktion AS hauptredner_fraktion,
            s.wahlperiode AS wahlperiode,
            s.sitzungsnr AS sitzungsnr,
            s.datum AS sitzung_datum,
            COALESCE(r.raw_text, r.clean_text, r.text, '') AS rede_raw_text
        FROM zwischenrufe z
        JOIN reden r ON z.rede_id = r.rede_id
        JOIN redner red ON r.redner_id = red.redner_id
        JOIN sitzungen s ON r.sitzung_id = s.sitzungs_id
        WHERE z.sentiment_score >= ?
        ORDER BY z.sentiment_score DESC, z.ruf_id ASC
    """

    df_raw = pd.read_sql_query(query, conn, params=(min_sentiment,))
    conn.close()

    if df_raw.empty:
        logger.warning(f"No interjections found with sentiment_score >= {min_sentiment}")
        cols = [
            "interjection_id",
            "zwischenruf_text",
            "sentiment_score",
            "zwischenruf_fraktion",
            "hauptredner",
            "hauptredner_fraktion",
            "wahlperiode",
            "sitzungsnr",
            "sitzung_datum",
            "kontext_5_saetze_vorher",
            "kontext_3_satz_nachher",
            "manual_label",
            "error_category",
            "notes",
        ]
        return pd.DataFrame(columns=cols)

    # Filter for Opposition -> Government interjections
    matching_rows = []
    for row in df_raw.itertuples():
        z_frak = getattr(row, "zwischenruf_fraktion", "")
        spk_frak = getattr(row, "hauptredner_fraktion", "")
        wp = getattr(row, "wahlperiode", 20)

        if is_opposition_to_government(z_frak, spk_frak, wp):
            matching_rows.append(row)

        if len(matching_rows) >= limit:
            break

    if not matching_rows:
        logger.warning("No Opposition -> Government interjections matched the criteria.")
        df_matched = pd.DataFrame()
    else:
        df_matched = pd.DataFrame(matching_rows)

    if df_matched.empty:
        cols = [
            "interjection_id",
            "zwischenruf_text",
            "sentiment_score",
            "zwischenruf_fraktion",
            "hauptredner",
            "hauptredner_fraktion",
            "wahlperiode",
            "sitzungsnr",
            "sitzung_datum",
            "kontext_5_saetze_vorher",
            "kontext_3_satz_nachher",
            "manual_label",
            "error_category",
            "notes",
        ]
        return pd.DataFrame(columns=cols)

    # Process speech context sentences with progress tracking and ETA calculation
    total_items = len(df_matched)
    start_time = time.time()
    before_contexts = []
    after_contexts = []

    log_step = max(1, total_items // 10)
    for i, row in enumerate(df_matched.itertuples(), 1):
        speech_text = getattr(row, "rede_raw_text", "")
        z_text = getattr(row, "zwischenruf_text", "")

        ctx_before, ctx_after = extract_context_from_speech(speech_text, z_text, before_count=5, after_count=3)
        before_contexts.append(ctx_before)
        after_contexts.append(ctx_after)

        if i == 1 or i % log_step == 0 or i == total_items:
            elapsed = time.time() - start_time
            pct = int((i / total_items) * 100)
            eta = (elapsed / i) * (total_items - i)
            eta_str = time.strftime("%M:%S", time.gmtime(eta))
            logger.info(f"SarcasmContextExtraction: {i}/{total_items} | {pct}% | ETA: {eta_str}")

    df_matched["kontext_5_saetze_vorher"] = before_contexts
    df_matched["kontext_3_satz_nachher"] = after_contexts

    # Add manual annotation fields for qualitative evaluation
    df_matched["manual_label"] = ""
    df_matched["error_category"] = ""
    df_matched["notes"] = ""

    # Canonicalise displayed faction names
    df_matched["zwischenruf_fraktion"] = df_matched["zwischenruf_fraktion"].apply(canonicalise_faction)
    df_matched["hauptredner_fraktion"] = df_matched["hauptredner_fraktion"].apply(canonicalise_faction)

    cols = [
        "interjection_id",
        "zwischenruf_text",
        "sentiment_score",
        "zwischenruf_fraktion",
        "hauptredner",
        "hauptredner_fraktion",
        "wahlperiode",
        "sitzungsnr",
        "sitzung_datum",
        "kontext_5_saetze_vorher",
        "kontext_3_satz_nachher",
        "manual_label",
        "error_category",
        "notes",
    ]

    return df_matched[cols].copy()


def generate_markdown(df: pd.DataFrame) -> str:
    """Generates a Markdown document for qualitative evaluation and sarcasm case studies."""
    md_lines = [
        "# Qualitative Sarcasm Case Studies (NLP Evaluation Framework)",
        "",
        "This dataset contains top positive interjections (> 0.95 sentiment score) called by ",
        "Opposition factions towards Government speakers in the German Bundestag. ",
        "Each record includes textual context (5 sentences before and 3 sentences after) for manual annotation.",
        "",
    ]

    if df.empty:
        md_lines.append("_No interjections matching the criteria were found._")
        return "\n".join(md_lines)

    for idx, row in df.iterrows():
        md_lines.append(f"### {idx + 1}. Interjection #{row['interjection_id']}: „{row['zwischenruf_text']}“")
        md_lines.append(f"- **Sentiment Score:** `{row['sentiment_score']:.4f}`")
        md_lines.append(f"- **Interjection Faction (Opposition):** {row['zwischenruf_fraktion']}")
        md_lines.append(f"- **Main Speaker (Government):** {row['hauptredner']} ({row['hauptredner_fraktion']})")
        md_lines.append(f"- **Session:** Wahlperiode {row['wahlperiode']}, Sitzung {row['sitzungsnr']} (Date: {row['sitzung_datum']})")
        md_lines.append(f"- **Context Before (5 sentences):** „*{row['kontext_5_saetze_vorher']}*“")
        md_lines.append(f"- **Context After (3 sentences):** „*{row['kontext_3_satz_nachher']}*“")
        md_lines.append("- **Manual Annotation:**")
        md_lines.append("  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_")
        md_lines.append("  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_")
        md_lines.append("  - `notes`: _[]_")
        md_lines.append("")

    return "\n".join(md_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract top positive extreme interjections from Opposition to Government for sarcasm NLP evaluation."
    )
    parser.add_argument("--db", default="./data/openparlament.db", help="Path to SQLite database")
    parser.add_argument("--limit", type=int, default=50, help="Number of extreme positive interjections to extract (default: 50)")
    parser.add_argument("--min-sentiment", type=float, default=0.95, help="Minimum sentiment score threshold (default: 0.95)")
    parser.add_argument("--output-csv", default="./data/sarcasm_case_studies.csv", help="Path to CSV output file")
    parser.add_argument("--output-excel", default="./data/sarcasm_case_studies.xlsx", help="Path to Excel output file")
    parser.add_argument("--output-md", default="./data/sarcasm_case_studies.md", help="Path to Markdown output file")
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("limit must be a positive integer")
    if args.min_sentiment < -1.0 or args.min_sentiment > 1.0:
        raise ValueError("min_sentiment must be between -1.0 and 1.0")

    df_report = extract_case_studies(args.db, limit=args.limit, min_sentiment=args.min_sentiment)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)

    df_report.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    logger.info("CSV report successfully saved to %s (%d rows)", args.output_csv, len(df_report))

    try:
        df_report.to_excel(args.output_excel, index=False, engine="openpyxl")
        logger.info("Excel report successfully saved to %s", args.output_excel)
    except Exception as exc:
        logger.warning("Could not export Excel file: %s", exc)

    md_content = generate_markdown(df_report)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Markdown report successfully saved to %s", args.output_md)


if __name__ == "__main__":
    main()
