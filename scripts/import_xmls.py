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
Import all XML files from data/xml using the project's ProtocolImporter.
Also backfills datum and wochentag for any already-imported Sitzungen that
are still missing those values (e.g. imported before the parser was fixed).

Usage:
  python scripts/import_xmls.py
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

# Configure logging to match project standards and include modular context
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("import_xmls")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import select
from src.database import init_db, get_session
from src.models import Sitzung
from src.parser import BundestagXMLParser, weekday_german
from scripts.run_scraper import ProtocolImporter


def _backfill_missing_dates(xml_dir: Path) -> None:
    """Re-parse local XML files to fill in datum / wochentag where NULL."""
    parser = BundestagXMLParser()

    with get_session() as session:
        rows = session.execute(
            select(Sitzung).where(
                (Sitzung.datum.is_(None)) | (Sitzung.wochentag.is_(None))
            )
        ).scalars().all()

        total = len(rows)
        if not total:
            logger.info("All Sitzungen already have datum and wochentag.")
            return

        logger.info(
            "Backfilling datum/wochentag for %d Sitzung(en) with missing data…",
            total,
        )
        updated = 0
        start_time = time.time()

        for idx, sitzung in enumerate(rows, start=1):
            xml_file = xml_dir / f"{sitzung.wahlperiode:02d}{sitzung.sitzungsnr:03d}.xml"
            if not xml_file.exists():
                continue
            try:
                result = parser.parse_file(xml_file)
                changed = False
                if sitzung.datum is None and result.sitzung.datum is not None:
                    sitzung.datum = result.sitzung.datum
                    changed = True
                if sitzung.wochentag is None and sitzung.datum is not None:
                    sitzung.wochentag = weekday_german(sitzung.datum)
                    changed = True
                if changed:
                    updated += 1
                    logger.debug(
                        "  WP%d/%d → datum=%s  wochentag=%s",
                        sitzung.wahlperiode, sitzung.sitzungsnr,
                        sitzung.datum, sitzung.wochentag,
                    )
            except Exception:
                logger.exception("Failed to re-parse %s", xml_file.name)

            # Throttling logging: Log on first/last, and periodically
            if idx % 5 == 0 or idx == total:
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                eta = avg_time * (total - idx)
                eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
                logger.info(
                    f"BackfillSitzungen: {idx}/{total} | {int((idx/total)*100)}% | ETA: {eta_str}"
                )

        session.commit()
        logger.info("Backfilled %d/%d session(s).", updated, total)


def main() -> None:
    init_db()
    importer = ProtocolImporter()
    xml_dir = _PROJECT_ROOT / "data" / "xml"
    if not xml_dir.exists():
        logger.error("XML directory not found: %s", xml_dir)
        return

    files = sorted(xml_dir.glob("*.xml"))
    total_files = len(files)
    if not total_files:
        logger.error("No XML files found in %s", xml_dir)
        return

    # ── Import new files ───────────────────────────────────────────────────────
    imported = 0
    start_time = time.time()

    for idx, p in enumerate(files, start=1):
        logger.debug("Importing %s", p.name)
        try:
            if importer.import_file(p):
                imported += 1
        except Exception:
            logger.exception("Failed to import %s", p.name)

        # Throttling logging to match standard format
        if idx % 10 == 0 or idx == total_files:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            eta = avg_time * (total_files - idx)
            eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
            logger.info(
                f"ImportXMLs: {idx}/{total_files} | {int((idx/total_files)*100)}% | ETA: {eta_str}"
            )

    logger.info("Imported %d/%d files.", imported, total_files)

    # ── Backfill any sessions that are still missing datum / wochentag ─────────
    _backfill_missing_dates(xml_dir)


if __name__ == "__main__":
    main()
