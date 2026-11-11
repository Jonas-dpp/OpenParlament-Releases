#!/usr/bin/env python3
"""
OpenParlament - Streamlit Dashboard
Copyright (C) 2026 Jonas-dpp

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

--------------------------------------------------------------------------
db_init.py – Initialise the OpenParlament SQLite database.

Run from the project root:
    python scripts/db_init.py

Options:
    --reset           Drops all tables before initializing (WARNING: Deletes all data).
    --reset-framing   Drops/clears framing and lexical precomputed analysis results.
    --reset-nlp       Resets sentiment, tone, and addressee columns on speeches and
                      interjections to NULL and clears reaction gap results.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_engine, init_db, reset_framing, reset_nlp
from src.models import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise the OpenParlament database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables before initializing (WARNING: Deletes all data)."
    )
    parser.add_argument(
        "--reset-framing",
        action="store_true",
        help="Reset framing analysis: drop all precomputed framing and lexical records."
    )
    parser.add_argument(
        "--reset-nlp",
        action="store_true",
        help="Reset NLP analysis: reset sentiment, tone, and addressees on speeches/interjections to NULL and drop reaction gap records."
    )
    args = parser.parse_args()

    engine = get_engine()

    if args.reset:
        logger.warning("Resetting database: Dropping all existing tables...")
        Base.metadata.drop_all(engine)
        logger.info("All tables dropped.")
    else:
        # Initialize schema first to ensure tables/columns exist before partial deletes/updates
        init_db()
        if args.reset_framing:
            logger.warning("Resetting framing data: Clearing framing_results and lexical_results...")
            reset_framing()
            logger.info("Framing and lexical records reset successfully.")

        if args.reset_nlp:
            logger.warning("Resetting NLP data: Clearing sentiment, tone, addressee fields and reaction_gap_results...")
            reset_nlp()
            logger.info("NLP fields and reaction gap records reset successfully.")

    logger.info("Initialising database...")
    init_db()
    
    db_url = str(engine.url)
    logger.info("Database ready at: %s", db_url)
    
    with engine.connect() as conn:
        tables = list(engine.dialect.get_table_names(conn))
        logger.info("Tables in database: %s", tables)


if __name__ == "__main__":
    main()