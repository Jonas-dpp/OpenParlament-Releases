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
run_scraper.py – Download Bundestag plenary protocols and import them into the DB.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_session, init_db
from src.models import Redner, Sitzung, Zwischenruf
from src.parser import BundestagXMLParser
from src.scraper import BundestagScraper
from sqlalchemy import select, update

# NLP helpers shared with run_nlp_cli.py
from scripts.run_nlp_cli import (
    _pk_attr,
    _serialize_for_column,
    is_column_attr,
    tone_scores_to_columns,
)
from scripts.nlp_session import NLPSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_scraper")


# ─────────────────────────────────────────────────────────────────────────────
# Importer
# ─────────────────────────────────────────────────────────────────────────────

class ProtocolImporter:
    def __init__(self) -> None:
        self._parser = BundestagXMLParser()

    @staticmethod
    def _redner_key(r) -> str | None:
        if r.bundestag_id:
            return str(r.bundestag_id)
        return r.vollname.lower().strip()


    def import_file(self, path: Path) -> bool:
        fname = path.stem
        result = None
        wp, snr = None, None

        # 1. Determine IDs
        if fname.isdigit() and len(fname) >= 3:
            wp, snr = int(fname[:2]), int(fname[2:])
        else:
            result = None

        with get_session() as session:
            try:
                # 1. Existenz-Check (Sitzung)
                existing = session.execute(
                    select(Sitzung).where(Sitzung.wahlperiode == wp, Sitzung.sitzungsnr == snr)
                ).scalar_one_or_none()

                if existing:
                    logger.info("Skipping already-imported Sitzung %d/%d.", wp, snr)
                    return False

                # 2. Parsen (Hier passiert die Magie durch dein Parser-Update!)
                # Der Parser nutzt jetzt session.get(), um Duplikate in der Session zu vermeiden.
                if result is None:
                    result = self._parser.parse_file(path, session=session)
                    wp, snr = result.sitzung.wahlperiode, result.sitzung.sitzungsnr

                # 3. Redner-Map aus Parser-Ergebnis (Instanzen bereits in Session)
                redner_map: dict[str, Redner] = {}
                for r in result.redner:
                    key = self._redner_key(r)
                    if not key:
                        continue
                    redner_map[key] = r

                # 4. Persistence (Sitzung & Reden verknüpfen)
                session.add(result.sitzung)
                session.flush()  # ID für Sitzung generieren

                for rede in result.reden:
                    rede.sitzung = result.sitzung
                    session.add(rede)

                for zwr in result.zwischenrufe:
                    session.add(zwr)

                session.commit()
                logger.info("Imported Sitzung %d/%d.", wp, snr)
                return True
            except Exception as e:
                session.rollback()
                logger.error(
                    "Import failed for %s/%s (%s): %s\n%s",
                    wp, snr, path.name, e, traceback.format_exc(),
                )
                raise

# ─────────────────────────────────────────────────────────────────────────────
# NLP scoring
# ─────────────────────────────────────────────────────────────────────────────

def run_nlp_scoring(
    batch_size: int = 100,
    *,
    use_cuda: bool = False,
    device_index: int | None = None,
    use_ner: bool = False,
    fp16: bool = False,
    commit_interval: int = 10,
) -> None:
    """Run sentiment, tone, and addressee scoring on all unscored Zwischenrufe.

    Uses :class:`NLPSession` so that all three NLP engines are initialised in
    the same way as in ``run_nlp_cli.py`` (fp16 support, CUDA device selection,
    lazy transformers import).  Results are written back via a single bulk
    UPDATE per ORM class per *commit_interval* batches (or on the final batch)
    rather than after every single batch, reducing DB round-trips.

    :param commit_interval: Number of batches to accumulate before issuing a
        ``COMMIT``.  Lower values are safer (less re-work on failure) but
        produce more DB round-trips.  Default: 10.
    """
    if commit_interval < 1:
        raise ValueError(f"commit_interval must be >= 1, got {commit_interval}")

    with NLPSession(
        use_cuda=use_cuda,
        device_index=device_index,
        batch_size=batch_size,
        use_ner=use_ner,
        fp16=fp16,
    ) as agent:
        sentiment_engine = agent.sentiment_engine
        tone_classifier = agent.tone_classifier
        addressee_detector = agent.addressee_detector

        with get_session() as session:
            rows = session.execute(
                select(Zwischenruf).where(Zwischenruf.sentiment_score.is_(None))
            ).scalars().all()

            if not rows:
                logger.info("All interjections already scored.")
                return

            num_batches = (len(rows) + batch_size - 1) // batch_size
            # Accumulated bulk-update mappings across batches within the current
            # commit window, keyed by ORM class.
            pending_rows: Dict[type, List[dict]] = defaultdict(list)
            # Track the range of rows covered by the pending window for logging.
            pending_start: int = 1

            start_time = time.time()
            for batch_num, batch_start in enumerate(range(0, len(rows), batch_size), start=1):
                # Throttled logging of progress with ETA
                if batch_num == 1 or batch_num % 10 == 0 or batch_num == num_batches:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / batch_num
                    eta = avg_time * (num_batches - batch_num)
                    eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
                    logger.info(
                        f"NLPScoring: {batch_num}/{num_batches} | {int((batch_num/num_batches)*100)}% | ETA: {eta_str}"
                    )

                batch_rows: List[Zwischenruf] = rows[batch_start:batch_start + batch_size]
                texts = [z.text or "" for z in batch_rows]

                sentiments = []
                if sentiment_engine:
                    try:
                        sentiments = sentiment_engine.score_batch(texts)
                    except Exception as exc:
                        logger.error("Sentiment engine failed: %s", exc)
                        sentiments = [0.0] * len(batch_rows)

                tones = []
                if tone_classifier:
                    try:
                        tones = tone_classifier.classify_batch(texts)
                    except Exception as exc:
                        logger.error("Tone classifier failed: %s", exc)
                        tones = [("Neutral", {}) for _ in batch_rows]

                addressees = []
                if addressee_detector:
                    try:
                        addressees = addressee_detector.detect_batch(texts)
                    except Exception as exc:
                        logger.error("Addressee detection failed: %s", exc)
                        addressees = [[] for _ in batch_rows]

                # Accumulate bulk-update mappings for this batch.
                for i, obj in enumerate(batch_rows):
                    cls = obj.__class__
                    pk_name = _pk_attr(cls)
                    row_map: dict = {pk_name: getattr(obj, pk_name)}

                    if sentiments and is_column_attr(cls, "sentiment_score"):
                        row_map["sentiment_score"] = float(sentiments[i])

                    if tones and is_column_attr(cls, "ton_label"):
                        label, scores = tones[i]
                        row_map["ton_label"] = label
                        if is_column_attr(cls, "tone_scores"):
                            row_map["tone_scores"] = _serialize_for_column(cls, "tone_scores", scores)
                        for col, value in tone_scores_to_columns(scores).items():
                            if is_column_attr(cls, col):
                                row_map[col] = value

                    if addressees and is_column_attr(cls, "adressaten"):
                        row_map["adressaten"] = _serialize_for_column(cls, "adressaten", addressees[i])

                    if len(row_map) > 1:
                        pending_rows[cls].append(row_map)

                is_last_batch = batch_num == num_batches
                should_commit = (batch_num % commit_interval == 0) or is_last_batch

                if should_commit and pending_rows:
                    try:
                        for cls, mappings in pending_rows.items():
                            session.execute(
                                update(cls),
                                mappings,
                                execution_options={"synchronize_session": False},
                            )
                        session.commit()
                        session.expire_all()
                        logger.info(
                            "Committed rows %d–%d (batches %d–%d).",
                            pending_start,
                            batch_start + len(batch_rows),
                            (pending_start - 1) // batch_size + 1,
                            batch_num,
                        )
                        pending_rows.clear()
                        pending_start = batch_start + len(batch_rows) + 1
                    except Exception as exc:
                        session.rollback()
                        logger.error(
                            "Failed to commit NLP rows %d–%d: %s  "
                            "— You can safely re-run this command; already committed rows will not be affected.",
                            pending_start,
                            batch_start + len(batch_rows),
                            exc,
                        )
                        raise

# ─────────────────────────────────────────────────────────────────────────────
# CLI / Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenParlament Scraper")
    p.add_argument(
        "--wahlperiode",
        type=int,
        nargs="+",
        default=[18, 19, 20, 21],
        metavar="WP",
        help="Eine oder mehrere Wahlperioden, z. B. --wahlperiode 18 19 20 21."
    )
    p.add_argument("--max-pages", type=int, default=100)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--redownload", action="store_true", 
                   help="Re-download files even if they already exist locally.")
    
    # NLP Arguments
    p.add_argument("--nlp", action="store_true",
                   help="Run NLP scoring (sentiment, tone, addressee) after import.")
    p.add_argument("--nlp-batch-size", type=int, default=100,
                   help="Batch size for NLP scoring.")
    p.add_argument("--nlp-cuda", action="store_true",
                   help="Use CUDA for NLP models.")
    p.add_argument("--nlp-device-index", type=int, default=None,
                   help="Explicit CUDA device index.")
    p.add_argument("--nlp-use-ner", action="store_true",
                   help="Enable NER in AddresseeDetector.")
    p.add_argument("--nlp-fp16", action="store_true",
                   help="Load NLP models in half-precision (GPU only).")
    p.add_argument("--commit-interval", type=int, default=10,
                   metavar="N",
                   help="Commit NLP scoring results to DB every N batches (default: 10). Must be >= 1.")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Post-parsing parameter validations for security and robustness
    if not args.wahlperiode:
        raise ValueError("At least one wahlperiode must be specified.")
    if any(wp <= 0 for wp in args.wahlperiode):
        raise ValueError(f"wahlperiode values must be positive integers, got {args.wahlperiode}")

    args.wahlperiode = list(dict.fromkeys(args.wahlperiode))

    if args.max_pages <= 0:
        raise ValueError(f"max-pages must be a positive integer, got {args.max_pages}")
    if args.delay < 0.0:
        raise ValueError(f"delay must be a non-negative float/int, got {args.delay}")
    if args.nlp_batch_size <= 0:
        raise ValueError(f"nlp-batch-size must be a positive integer, got {args.nlp_batch_size}")
    if args.commit_interval < 1:
        raise ValueError(f"commit-interval must be >= 1, got {args.commit_interval}")

    init_db()

    download_dir = _PROJECT_ROOT / "data" / "xml"
    scraper = BundestagScraper(
        download_dir=download_dir,
        request_delay=args.delay,
        skip_existing=not args.redownload,
    )
    importer = ProtocolImporter()

    urls = []
    for wahlperiode in args.wahlperiode:
        logger.info("Fetching protocol URLs for Wahlperiode %d...", wahlperiode)
        wp_urls = scraper.fetch_protocol_urls(
            wahlperiode=wahlperiode,
            max_pages=args.max_pages,
        )
        logger.info(
            "Found %d protocol URLs for Wahlperiode %d.",
            len(wp_urls),
            wahlperiode,
        )
        urls.extend(wp_urls)

    paths = scraper.download_all(urls)

    successes = 0
    failures = 0
    total_paths = len(paths)
    import_start_time = time.time()

    for idx, p in enumerate(paths, start=1):
        try:
            if importer.import_file(p): successes += 1
        except Exception as exc:
            failures += 1
            logger.error("Failed to import protocol file %s: %s", p, exc)

        # Throttling logging: log every 5 protocols or at the end to match project standards
        if idx % 5 == 0 or idx == total_paths:
            elapsed = time.time() - import_start_time
            avg_time = elapsed / idx
            eta = avg_time * (total_paths - idx)
            eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))
            logger.info(
                "ProtocolImport: %d/%d | %d%% | ETA: %s",
                idx,
                total_paths,
                int((idx / total_paths) * 100),
                eta_str
            )

    if failures > 0:
        logger.warning("Imported %d new session(s); %d file(s) failed.", successes, failures)
    else:
        logger.info("Imported %d new session(s).", successes)

    if args.nlp:
        run_nlp_scoring(
            batch_size=args.nlp_batch_size,
            use_cuda=args.nlp_cuda,
            device_index=args.nlp_device_index,
            use_ner=args.nlp_use_ner,
            fp16=args.nlp_fp16,
            commit_interval=args.commit_interval,
        )

if __name__ == "__main__":
    main()