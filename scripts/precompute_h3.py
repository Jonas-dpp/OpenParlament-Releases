"""
OpenParlament - Precomputation Script for Hypothese 3 (Asymmetric Attacks Network)
Copyright (C) 2026 Jules

This script extracts plenary interjection data, cleans and canonicalises faction names,
and precomputes the networks of interjections. Results are written to a compressed
Parquet file, ensuring the Streamlit app can load them near-instantaneously.

Usage:
    python scripts/precompute_h3.py --all
    python scripts/precompute_h3.py --wahlperiode 20
"""

import argparse
import sys
import logging
import time
import uuid
from pathlib import Path
from typing import List

# Allow importing from src/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from sqlalchemy import select, func
from src.database import get_session, init_db
from src.models import Sitzung, Rede, Redner, Zwischenruf
from src.analytics import _canonicalise_faction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("precomp_h3")


def normalize_faction_h3(name: str) -> str:
    """Map raw faction name to a highly consistent set of main Bundestag factions."""
    if not name:
        return ""
    name_clean = name.replace("\xa0", " ").strip()
    canonical = _canonicalise_faction(name_clean)
    if not canonical:
        return ""

    cu = canonical.upper()
    if cu in {"BÜNDNIS 90/DIE GRÜNEN", "GRÜNE", "BÜNDNIS90/DIE GRÜNEN", "BÜNDNIS 90 / DIE GRÜNEN", "B90/GRÜNE"}:
        return "BÜNDNIS 90/DIE GRÜNEN"
    if cu in {"DIE LINKE", "LINKE", "PDS", "DIE LINKE."}:
        return "Die Linke"
    if cu in {"CDU", "CSU", "CDU/CSU", "UNION"}:
        return "CDU/CSU"
    if cu in {"SPD"}:
        return "SPD"
    if cu in {"FDP"}:
        return "FDP"
    if cu in {"AFD", "ADD"}:
        return "AfD"
    if cu in {"BSW"}:
        return "BSW"
    if cu in {"SSW"}:
        return "SSW"
    return canonical


def precompute_data(wahlperioden: List[int]) -> pd.DataFrame:
    """Query and clean data for the specified list of wahlperioden."""
    logger.info("Initializing database connection...")
    init_db()

    all_records = []

    with get_session() as session:
        # Check total count first
        total_count = session.query(func.count(Zwischenruf.ruf_id)).scalar()
        logger.info(f"Total interjections in DB: {total_count}")

        if total_count == 0:
            logger.warning("Database contains no interjections! Please import data first.")
            return pd.DataFrame()

        # Build query
        stmt = (
            select(
                Sitzung.wahlperiode.label("wp"),
                Zwischenruf.fraktion.label("sender"),
                Redner.fraktion.label("speaker"),
                Zwischenruf.sentiment_score.label("sentiment")
            )
            .join(Rede, Rede.rede_id == Zwischenruf.rede_id)
            .join(Redner, Redner.redner_id == Rede.redner_id)
            .join(Sitzung, Sitzung.sitzungs_id == Rede.sitzung_id)
            .where(Sitzung.wahlperiode.in_(wahlperioden))
            .where(Zwischenruf.fraktion.isnot(None))
            .where(Redner.fraktion.isnot(None))
        )

        logger.info("Executing database query to fetch raw interjection rows...")
        start_time = time.time()
        rows = session.execute(stmt).fetchall()
        logger.info(f"Query returned {len(rows)} matching rows in {time.time() - start_time:.2f}s.")

        if not rows:
            logger.warning("No rows found matching the criteria.")
            return pd.DataFrame()

        total_rows = len(rows)
        logger.info("Processing and normalising rows...")

        # We define a list of main factions we want to include for the network
        main_factions = {"CDU/CSU", "SPD", "BÜNDNIS 90/DIE GRÜNEN", "FDP", "AfD", "Die Linke", "BSW", "SSW"}

        # Dynamic throttling step to prevent terminal silence for smaller datasets
        # while guarding against performance-degrading I/O overhead (targets ~20 log statements)
        log_step = max(500, total_rows // 20)

        processing_start = time.time()
        for i, row in enumerate(rows):
            wp, sender, speaker, sentiment = row

            s_norm = normalize_faction_h3(sender)
            t_norm = normalize_faction_h3(speaker)

            if s_norm in main_factions and t_norm in main_factions:
                all_records.append({
                    "wp": int(wp),
                    "sender": s_norm,
                    "speaker": t_norm,
                    "sentiment": float(sentiment) if sentiment is not None else 0.0
                })

            # Progress/ETA Telemetry logging (throttled)
            if (i + 1) == 1 or (i + 1) % log_step == 0 or (i + 1) == total_rows:
                elapsed = time.time() - processing_start
                pct = int(((i + 1) / total_rows) * 100)
                avg_time = elapsed / (i + 1)
                eta = avg_time * (total_rows - (i + 1))
                eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
                logger.info(f"H3DBPrepare: {i + 1}/{total_rows} | {pct}% | ETA: {eta_str}")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    logger.info(f"Extracted and cleaned {len(df)} records for the main factions.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Precompute Hypothese 3 interjection network data.")
    parser.add_argument(
        "--wahlperiode", type=int, help="Specific positive Wahlperiode to precompute."
    )
    parser.add_argument(
        "--all", action="store_true", help="Precompute for all Wahlperioden in the DB."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose DEBUG logging."
    )

    args = parser.parse_args()

    # Post-parsing strict validation
    if args.wahlperiode is not None and args.wahlperiode <= 0:
        raise ValueError(f"wahlperiode must be a positive integer, got {args.wahlperiode}")

    if not args.all and args.wahlperiode is None:
        logger.error("Please specify either --wahlperiode <num> or --all.")
        parser.print_help()
        sys.exit(1)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    logger.info("Starting Hypothese 3 Precomputation...")

    if args.all:
        # Fetch distinct wahlperioden from Sitzung
        with get_session() as session:
            init_db()
            wps = session.execute(
                select(Sitzung.wahlperiode).where(Sitzung.wahlperiode.is_not(None)).distinct()
            ).scalars().all()
            if not wps:
                logger.error("No Wahlperioden found in database. Exiting.")
                sys.exit(1)
            target_wps = sorted(wps)
            logger.info(f"Target Wahlperioden from DB: {target_wps}")
    else:
        target_wps = [args.wahlperiode]

    df_results = precompute_data(target_wps)

    # Output file path
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "h3_precomputed.parquet"

    # If the recomputed target data is empty, do not destroy or modify existing results
    if df_results.empty:
        logger.warning("No data computed for target Wahlperioden %s. Existing Parquet file will not be overwritten.", target_wps)
        sys.exit(0)

    # Support incremental updates: when precomputing a single WP, merge with existing data
    if not args.all and out_path.exists():
        try:
            existing_df = pd.read_parquet(out_path)
        except Exception as exc:
            logger.error("Failed to read existing Parquet file at %s: %s. Aborting to protect data integrity.", out_path, exc)
            raise RuntimeError(f"Unreadable existing Parquet file: {out_path}") from exc

        other_wps_df = existing_df[~existing_df["wp"].isin(target_wps)]
        df_results = pd.concat([other_wps_df, df_results], ignore_index=True)

    # Save as compressed Parquet atomically using a unique UUID temporary file
    tmp_path = data_dir / f"h3_precomputed_{uuid.uuid4().hex}.tmp.parquet"
    try:
        logger.info(f"Writing precomputed results to {out_path} via temporary file {tmp_path.name}...")
        df_results.to_parquet(tmp_path, compression="snappy", index=False)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    logger.info(f"Successfully wrote {len(df_results)} records. File size: {out_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info("Hypothese 3 precomputation complete!")


if __name__ == "__main__":
    main()
