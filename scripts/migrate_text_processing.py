#!/usr/bin/env python3
"""
OpenParlament - Streamlit Dashboard
Copyright (C) 2026 Jonas-dpp

CLI script to migrate/backfill existing speeches in the database
with the new more-layered text processing model.

Usage:
  python scripts/migrate_text_processing.py [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import select, func
from src.database import get_session, init_db
from src.models import Rede
from src.text_processing.processor import TextProcessor

# Configure standard telemetry logging format
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("migrate_text_processing")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate text processing layers for existing speeches.")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for processing database records.")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (unused, kept for CLI compatibility).")
    parser.add_argument("--limit", type=int, default=None, help="Limit total number of speeches to process (for testing).")
    parser.add_argument("--resume", action="store_true", help="Resume migration, processing only speeches with older versions or unmigrated (raw_text is NULL).")
    parser.add_argument("--force", action="store_true", help="Force processing all speeches even if they match current version.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry-run (no DB writes).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug/verbose logging.")
    parser.add_argument("--stats-only", action="store_true", help="Only compute and display stats, do not run the migration.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be a positive integer")
    if args.workers <= 0:
        raise ValueError("workers must be a positive integer")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be a positive integer")

    return args

def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    init_db()
    processor = TextProcessor()
    current_version = processor.version

    logger.info("Starting text processing migration (Target Version: %d)...", current_version)
    if args.dry_run:
        logger.info("--- DRY RUN MODE (No changes will be saved to database) ---")

    with get_session() as session:
        # Check current DB stats
        total_stmt = select(func.count(Rede.rede_id))
        total_count = session.execute(total_stmt).scalar() or 0

        unmigrated_stmt = select(func.count(Rede.rede_id)).where(Rede.raw_text.is_(None))
        unmigrated_count = session.execute(unmigrated_stmt).scalar() or 0

        older_version_stmt = select(func.count(Rede.rede_id)).where((Rede.raw_text.isnot(None)) & (Rede.processing_version < current_version))
        older_count = session.execute(older_version_stmt).scalar() or 0

        logger.info("Total speeches in DB: %d", total_count)
        logger.info("Unprocessed speeches (raw_text is NULL): %d", unmigrated_count)
        logger.info("Speeches with older processing version (< %d): %d", current_version, older_count)

        if args.stats_only:
            logger.info("Stats computed. Exiting as requested by --stats-only.")
            return

        # Determine which records to load
        stmt = select(Rede)
        if args.resume:
            logger.info("Resume option active: targeting unprocessed or older-version speeches.")
            stmt = stmt.where((Rede.raw_text.is_(None)) | (Rede.processing_version < current_version))
        elif not args.force:
            # Default behavior: process unmigrated speeches
            logger.info("Default behavior active: targeting unprocessed speeches.")
            stmt = stmt.where(Rede.raw_text.is_(None))
        else:
            logger.info("Force option active: reprocessing ALL speeches.")

        # Order by rede_id to keep it sequential and support limit
        stmt = stmt.order_by(Rede.rede_id)
        if args.limit is not None:
            logger.info("Limiting query to %d speeches.", args.limit)
            stmt = stmt.limit(args.limit)

        speeches = session.execute(stmt).scalars().all()
        to_process_count = len(speeches)
        logger.info("Loaded %d speeches for migration.", to_process_count)

        if to_process_count == 0:
            logger.info("No speeches require processing. Migration complete!")
            return

        processed_count = 0
        start_time = time.perf_counter()

        # Batch iteration
        for start_idx in range(0, to_process_count, args.batch_size):
            batch_speeches = speeches[start_idx:start_idx + args.batch_size]

            # Atomic transaction context
            with session.begin_nested() if hasattr(session, 'begin_nested') else session:
                for rede in batch_speeches:
                    # If Rede's raw_text is NULL, we populate it with the legacy `text` content as fallback
                    text_to_process = rede.raw_text if rede.raw_text is not None else rede.text
                    if not text_to_process:
                        text_to_process = ""

                    processed = processor.process(text_to_process)

                    if not args.dry_run:
                        rede.text = processed.clean_text # Keep backward compatibility
                        rede.raw_text = processed.raw_text
                        rede.clean_text = processed.clean_text
                        rede.normalized_text = processed.normalized_text
                        rede.metadata_text = processed.metadata_text
                        rede.processing_version = current_version
                        rede.processed_at = datetime.now()

                        # Populate quality metrics
                        rede.char_count_raw = processed.metrics.char_count_raw
                        rede.char_count_clean = processed.metrics.char_count_clean
                        rede.char_count_metadata = processed.metrics.char_count_metadata
                        rede.removed_chars = processed.metrics.removed_chars
                        rede.removed_percent = processed.metrics.removed_percent
                        rede.count_applause = processed.metrics.count_applause
                        rede.count_heckling = processed.metrics.count_heckling
                        rede.count_laughter = processed.metrics.count_laughter
                        rede.count_president = processed.metrics.count_president
                        rede.processing_time_ms = processed.metrics.processing_time_ms

                if not args.dry_run:
                    session.commit()

            processed_count += len(batch_speeches)

            # Throughput & ETA metrics
            elapsed_total = time.perf_counter() - start_time
            throughput = processed_count / elapsed_total if elapsed_total > 0 else 0.0
            remaining_count = to_process_count - processed_count
            eta_seconds = remaining_count / throughput if throughput > 0 else 0.0

            if eta_seconds >= 3600:
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            else:
                eta_str = time.strftime('%M:%S', time.gmtime(eta_seconds))

            logger.info(
                "TextMigration: %d/%d | %d%% | ETA: %s (Throughput: %.2f speeches/sec)",
                processed_count, to_process_count, int((processed_count / to_process_count) * 100),
                eta_str, throughput
            )

        total_elapsed = time.perf_counter() - start_time
        logger.info(
            "Migration finished! Processed %d speeches in %.2f seconds (Average: %.2f speeches/sec).",
            processed_count, total_elapsed, processed_count / total_elapsed if total_elapsed > 0 else 0.0
        )

if __name__ == "__main__":
    main()
