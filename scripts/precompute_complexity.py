"""
OpenParlament - Precomputation Script for Speech Complexity and Reaction Gaps
Copyright (C) 2026 Jules

This script performs heavy-lifting Speech Complexity and Reaction Gap computations offline.
Results are stored in the 'speech_complexity_results' and 'reaction_gap_results' tables,
allowing the Streamlit dashboard to load them near-instantaneously via simple SQL queries.

Usage:
    python scripts/precompute_complexity.py --wahlperiode 20
    python scripts/precompute_complexity.py --all
"""

import argparse
import sys
import logging
import time
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("precomp_comp")

# Allow importing from src/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import delete, select, func
from src.database import get_session, init_db
from src.models import SpeechComplexityResult, ReactionGapResult, Sitzung, Rede
from src.analytics import SpeechComplexityAnalysis, ReactionGapAnalysis


def precompute_for_wp(session, wp: int, limit: Optional[int] = None):
    """Run Speech Complexity and Reaction Gap analysis for a given wahlperiode and save to DB."""
    
    # ── 0. Fast Pre-Check ──
    speech_count = session.execute(
        select(func.count(Rede.rede_id))
        .join(Sitzung)
        .where(Sitzung.wahlperiode == wp)
    ).scalar()

    if not speech_count:
        logger.info(f"Skipping Wahlperiode {wp}: Keine Reden in der Datenbank gefunden.")
        return False

    logger.info(f"--- Starting Processing for Wahlperiode {wp} ({speech_count} Reden) ---")
    start_time_wp = time.time()

    # ── 1. Speech Complexity Analysis ──
    logger.info("Computing Speech Complexity results...")
    sc_start = time.time()
    sc_analyzer = SpeechComplexityAnalysis(session)
    df_complexity = sc_analyzer.run(wp, limit=limit)
    logger.info(f"Speech Complexity computation finished in {time.time() - sc_start:.2f}s. Found {len(df_complexity)} records.")

    # ── 2. Reaction Gap Analysis ──
    logger.info("Computing Reaction Gap results...")
    rg_start = time.time()
    rg_analyzer = ReactionGapAnalysis(session)
    df_gap = rg_analyzer.run(wp, limit=limit)
    logger.info(f"Reaction Gap computation finished in {time.time() - rg_start:.2f}s. Found {len(df_gap)} records.")

    try:
        with session.begin_nested():
            # --- Speech Complexity results ---
            if not df_complexity.empty:
                logger.info(f"Cleaning existing Speech Complexity results for WP {wp}...")
                result = session.execute(delete(SpeechComplexityResult).where(SpeechComplexityResult.wahlperiode == wp))
                logger.debug(f"Deleted {result.rowcount} old Speech Complexity records.")

                logger.info(f"Preparing insertion of {len(df_complexity)} Speech Complexity records...")
                sc_results = []
                total_sc = len(df_complexity)
                start_time_sc = time.time()
                for idx, (_, row) in enumerate(df_complexity.iterrows(), start=1):
                    sc_results.append(
                        SpeechComplexityResult(
                            rede_id=int(row["rede_id"]),
                            wahlperiode=wp,
                            redner_id=int(row["redner_id"]),
                            wstf_score=float(row.get("wstf_score", 0.0)),
                            flesch_score=float(row.get("flesch_score", 0.0)),
                            avg_sentence_length=float(row["avg_sentence_length"]),
                            subordinate_clause_ratio=float(row["subordinate_clause_ratio"]),
                            avg_word_length=float(row["avg_word_length"]),
                            long_word_ratio=float(row["long_word_ratio"]),
                            lexical_diversity=float(row["lexical_diversity"]),
                        )
                    )
                    if idx == 1 or idx % 1000 == 0 or idx == total_sc:
                        elapsed = time.time() - start_time_sc
                        avg_time = elapsed / idx
                        eta = avg_time * (total_sc - idx)
                        eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
                        logger.info(
                            f"SpeechComplexityDBPrepare: {idx}/{total_sc} | {int((idx/total_sc)*100)}% | ETA: {eta_str}"
                        )
                session.add_all(sc_results)
                logger.info(f"Added {len(sc_results)} Speech Complexity records to the session.")

            # --- Reaction Gap results ---
            if not df_gap.empty:
                logger.info(f"Cleaning existing Reaction Gap results for WP {wp}...")
                subq = select(Rede.rede_id).join(Sitzung).where(Sitzung.wahlperiode == wp)
                result = session.execute(delete(ReactionGapResult).where(ReactionGapResult.rede_id.in_(subq)))
                logger.debug(f"Deleted {result.rowcount} old Reaction Gap records.")

                logger.info(f"Preparing insertion of {len(df_gap)} Reaction Gap records...")
                rg_results = []
                total_rg = len(df_gap)
                start_time_rg = time.time()
                for idx, (_, row) in enumerate(df_gap.iterrows(), start=1):
                    rg_results.append(
                        ReactionGapResult(
                            rede_id=int(row["rede_id"]),
                            zwischenruf_id=int(row["zwischenruf_id"]),
                            topic_id=row["topic_id"],
                            speaker_sentiment=float(row["speaker_sentiment"]),
                            reaction_sentiment=float(row["reaction_sentiment"]),
                            gap=float(row["gap"]),
                            time_to_first_interjection=float(row["time_to_first_interjection"]),
                        )
                    )
                    if idx == 1 or idx % 1000 == 0 or idx == total_rg:
                        elapsed = time.time() - start_time_rg
                        avg_time = elapsed / idx
                        eta = avg_time * (total_rg - idx)
                        eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
                        logger.info(
                            f"ReactionGapDBPrepare: {idx}/{total_rg} | {int((idx/total_rg)*100)}% | ETA: {eta_str}"
                        )
                session.add_all(rg_results)
                logger.info(f"Added {len(rg_results)} Reaction Gap records to the session.")

        session.commit()
        total_time = time.time() - start_time_wp
        logger.info(f"Successfully committed all precomputed results for WP {wp}. Total time: {total_time:.2f}s.")
        return True
    except Exception:
        session.rollback()
        logger.error(f"Error during precomputation for WP {wp}. Transaction rolled back.", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Precompute Speech Complexity and Reaction Gaps.")
    parser.add_argument(
        "--wahlperiode", type=int, help="Specific Wahlperiode to process."
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all Wahlperioden in the DB."
    )
    parser.add_argument(
        "--limit", type=int, help="Limit on the number of speeches/reactions to process per analysis (for testing)."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose DEBUG logging."
    )

    args = parser.parse_args()

    # Post-parsing parameter validations for security and robustness
    if args.wahlperiode is not None and args.wahlperiode <= 0:
        raise ValueError(f"wahlperiode must be a positive integer, got {args.wahlperiode}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {args.limit}")

    # Adjust log level if debug is passed
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    logger.info("Initializing database connection...")
    init_db()

    with get_session() as session:
        if args.all:
            logger.info("Fetching all Wahlperioden from the database...")
            wps = session.execute(
                select(Sitzung.wahlperiode).where(Sitzung.wahlperiode.is_not(None)).distinct()
            ).scalars().all()
            if not wps:
                logger.warning("No Wahlperioden found in database. Exiting.")
                return
            
            sorted_wps = sorted(wps)
            total_wps = len(sorted_wps)
            processed_count = 0
            start_time = time.time()

            logger.info(f"Found {total_wps} Wahlperioden: {sorted_wps}")
            for i, wp in enumerate(sorted_wps, 1):
                success = precompute_for_wp(session, wp, limit=args.limit)
                if success:
                    processed_count += 1

                # Progress / ETA tracking across wahlperioden
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining_wps = total_wps - i
                eta = avg_time * remaining_wps

                if eta >= 3600:
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                else:
                    eta_str = time.strftime('%M:%S', time.gmtime(eta))

                logger.info(f"PrecomputeComplexityWahlperioden: {i}/{total_wps} | {int((i/total_wps)*100)}% | ETA: {eta_str}")

        elif args.wahlperiode:
            precompute_for_wp(session, args.wahlperiode, limit=args.limit)
        else:
            logger.error("No valid arguments provided. Please specify --wahlperiode or --all.")
            parser.print_help()


if __name__ == "__main__":
    main()