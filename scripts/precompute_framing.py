"""
OpenParlament - Precomputation Script for Framing Analysis (TF-IDF)
Copyright (C) 2026 Jonas-dpp

This script performs heavy-lifting spaCy preprocessing and scikit-learn TF-IDF
scoring offline. Results are stored in the 'framing_results' table, allowing
the Streamlit dashboard to load them near-instantaneously via simple SQL queries.

Usage:
    python scripts/precompute_framing.py --wahlperiode 20
    python scripts/precompute_framing.py --all
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Allow importing from src/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("precompute_framing")

from sqlalchemy import delete, select
from src.database import get_session, init_db
from src.models import FramingResult, LexicalResult, Sitzung
from src.analytics import FramingAnalysis, LexicalAnalysis


def precompute_for_wp(session, wp: int, n_terms: int = 500, limit: Optional[int] = None):
    """Run Framing and Lexical analysis for a given wahlperiode and save to DB."""
    logger.info(f"--- Processing Wahlperiode {wp} ---")

    # ── 1. Framing Analysis (TF-IDF) ──
    logger.info("Computing TF-IDF Framing terms across all segments...")
    fa_analyzer = FramingAnalysis(session)
    import pandas as pd
    segment_frames = []

    # 1.1 Load complete 'all' corpus (speeches AND interjections) and fit shared global vectorizer
    docs_all = fa_analyzer.load_segment_documents(wp, segment="all", limit=limit)
    if not docs_all.empty:
        fa_analyzer.fit(docs_all, segment="all")
        # Global variant for segment 'all'
        df_all_global = fa_analyzer.analyze_segment(
            segment="all",
            variant="global",
            top_n=n_terms,
            wahlperiode=wp,
            documents=docs_all,
        )
        if not df_all_global.empty:
            df_all_global["segment"] = "all"
            df_all_global["variant"] = "global"
            segment_frames.append(df_all_global[["fraktion", "term", "score", "segment", "variant"]])

        # Segmented variant for segment 'all'
        df_all_seg = fa_analyzer.analyze_segment(
            segment="all",
            variant="segmented",
            top_n=n_terms,
            wahlperiode=wp,
            documents=docs_all,
        )
        if not df_all_seg.empty:
            df_all_seg["segment"] = "all"
            df_all_seg["variant"] = "segmented"
            segment_frames.append(df_all_seg[["fraktion", "term", "score", "segment", "variant"]])

    # 1.2 Segment-specific terms: speech, interjection, short_intervention, question
    segments_to_compute = ["speech", "interjection", "short_intervention", "question"]

    for seg in segments_to_compute:
        try:
            logger.info(f"Computing TF-IDF terms for segment '{seg}'...")
            docs_seg = fa_analyzer.load_segment_documents(wp, segment=seg, limit=limit)
            if not docs_seg.empty:
                # Variant A: Global IDF model (shared across all communication forms)
                df_global = fa_analyzer.analyze_segment(
                    segment=seg,
                    variant="global",
                    top_n=n_terms,
                    wahlperiode=wp,
                    documents=docs_seg,
                )
                if not df_global.empty:
                    df_global["segment"] = seg
                    df_global["variant"] = "global"
                    segment_frames.append(df_global[["fraktion", "term", "score", "segment", "variant"]])

                # Variant B: Segment-specific IDF model (isolated within this communication form)
                df_segmented = fa_analyzer.analyze_segment(
                    segment=seg,
                    variant="segmented",
                    top_n=n_terms,
                    wahlperiode=wp,
                    documents=docs_seg,
                )
                if not df_segmented.empty:
                    df_segmented["segment"] = seg
                    df_segmented["variant"] = "segmented"
                    segment_frames.append(df_segmented[["fraktion", "term", "score", "segment", "variant"]])
        except Exception as e:
            logger.error(f"Error computing segment '{seg}' for WP {wp}; aborting precomputation: {e}", exc_info=True)
            raise

    all_framing_df = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame(columns=["fraktion", "term", "score", "segment", "variant"])

    # ── 2. Lexical Analysis (Diversity & Density) ──
    logger.info("Computing Lexical metrics...")
    lex_analyzer = LexicalAnalysis(session)
    df_lexical = lex_analyzer.run(wp)

    if all_framing_df.empty and df_lexical.empty:
        logger.info(f"No data found for Wahlperiode {wp}. Skipping.")
        return

    try:
        with session.begin_nested():
            # --- Framing results ---
            if not all_framing_df.empty:
                logger.info(f"Cleaning existing framing results for WP {wp}...")
                session.execute(delete(FramingResult).where(FramingResult.wahlperiode == wp))

                logger.info(f"Inserting {len(all_framing_df)} precomputed terms...")
                f_results = []
                total_f = len(all_framing_df)
                start_time_f = time.time()
                for idx, (_, row) in enumerate(all_framing_df.iterrows(), start=1):
                    f_results.append(
                        FramingResult(
                            wahlperiode=wp,
                            fraktion=row["fraktion"],
                            segment=str(row.get("segment", "all")),
                            variant=str(row.get("variant", "global")),
                            term=row["term"],
                            score=float(row["score"]),
                        )
                    )
                    if idx == 1 or idx % 500 == 0 or idx == total_f:
                        elapsed = time.time() - start_time_f
                        avg_time = elapsed / idx
                        eta = avg_time * (total_f - idx)
                        eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
                        logger.info(
                            f"FramingDBPrepare: {idx}/{total_f} | {int((idx/total_f)*100)}% | ETA: {eta_str}"
                        )
                session.add_all(f_results)

            # --- Lexical results ---
            if not df_lexical.empty:
                logger.info(f"Cleaning existing lexical results for WP {wp}...")
                session.execute(delete(LexicalResult).where(LexicalResult.wahlperiode == wp))

                logger.info(f"Inserting {len(df_lexical)} lexical records...")
                l_results = []
                total_l = len(df_lexical)
                start_time_l = time.time()
                for idx, (_, row) in enumerate(df_lexical.iterrows(), start=1):
                    l_results.append(
                        LexicalResult(
                            wahlperiode=wp,
                            fraktion=row["fraktion"],
                            ttr=float(row["ttr"]),
                            msttr=float(row["msttr"]),
                            lexical_density=float(row["lexical_density"]),
                            total_tokens=int(row["total_tokens"]),
                            total_types=int(row["total_types"]),
                        )
                    )
                    if idx == 1 or idx % 5 == 0 or idx == total_l:
                        elapsed = time.time() - start_time_l
                        avg_time = elapsed / idx
                        eta = avg_time * (total_l - idx)
                        eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
                        logger.info(
                            f"LexicalDBPrepare: {idx}/{total_l} | {int((idx/total_l)*100)}% | ETA: {eta_str}"
                        )
                session.add_all(l_results)

        session.commit()
        logger.info(f"Successfully precomputed WP {wp}.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error during precomputation for WP {wp}: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Precompute TF-IDF framing terms.")
    parser.add_argument(
        "--wahlperiode", type=int, help="Specific Wahlperiode to process."
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all Wahlperioden in the DB."
    )
    parser.add_argument(
        "--n-terms", type=int, default=500, help="Number of terms to precompute per faction (default: 500)."
    )
    parser.add_argument(
        "--limit", type=int, help="Limit on the number of speeches/reactions to process (for testing)."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose DEBUG logging."
    )

    args = parser.parse_args()

    # Post-parsing parameter validations for security and robustness
    if args.wahlperiode is not None and args.wahlperiode <= 0:
        raise ValueError(f"wahlperiode must be a positive integer, got {args.wahlperiode}")
    if args.n_terms <= 0:
        raise ValueError(f"n-terms must be a positive integer, got {args.n_terms}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {args.limit}")

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    # Ensure DB is initialised (table exists)
    init_db()

    with get_session() as session:
        if args.all:
            wps = session.execute(
                select(Sitzung.wahlperiode).where(Sitzung.wahlperiode.is_not(None)).distinct()
            ).scalars().all()
            if not wps:
                logger.warning("No Wahlperioden found in database.")
                return

            sorted_wps = sorted(wps)
            total_wps = len(sorted_wps)
            start_time = time.time()

            logger.info(f"Found {total_wps} Wahlperioden to process: {sorted_wps}")
            for i, wp in enumerate(sorted_wps, 1):
                precompute_for_wp(session, wp, n_terms=args.n_terms, limit=args.limit)

                # Progress / ETA tracking
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining_wps = total_wps - i
                eta = avg_time * remaining_wps

                if eta >= 3600:
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                else:
                    eta_str = time.strftime('%M:%S', time.gmtime(eta))

                logger.info(f"PrecomputeWahlperioden: {i}/{total_wps} | {int((i/total_wps)*100)}% | ETA: {eta_str}")

        elif args.wahlperiode:
            precompute_for_wp(session, args.wahlperiode, n_terms=args.n_terms, limit=args.limit)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
