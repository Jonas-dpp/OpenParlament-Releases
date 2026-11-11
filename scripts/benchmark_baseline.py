#!/usr/bin/env python3
"""Record baseline timings/memory for key OpenParlament workloads."""

from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analytics import ThemenKarriere
from src.database import get_session, init_db
from src.parser import BundestagXMLParser
from src.scraper import BundestagScraper


def _timer(label: str, fn):
    start = time.perf_counter()
    result = fn()
    dur = time.perf_counter() - start
    return label, dur, result


def main() -> None:
    p = argparse.ArgumentParser(description="Capture baseline benchmark metrics")
    p.add_argument("--db-url", default=os.environ.get("OPENPARLAMENT_DB_URL", ""))
    p.add_argument("--xml-file", default="")
    p.add_argument("--keyword", default="Klimaschutz")
    p.add_argument("--output", default=str(_PROJECT_ROOT / "tmp" / "benchmark_baseline.json"))
    args = p.parse_args()

    if args.db_url:
        os.environ["OPENPARLAMENT_DB_URL"] = args.db_url

    init_db()

    tracemalloc.start()
    metrics: dict[str, float] = {}

    with get_session() as session:
        tk = ThemenKarriere(session)
        k1, d1, _ = _timer("keyword_trend_seconds", lambda: tk.keyword_trend(args.keyword, wahlperiode=20))
        k2, d2, _ = _timer("multi_wp_keyword_trend_seconds", lambda: tk.multi_wp_keyword_trend(args.keyword))
        metrics[k1] = d1
        metrics[k2] = d2

    if args.xml_file:
        parser = BundestagXMLParser()
        _, d3, _ = _timer("parser_parse_file_seconds", lambda: parser.parse_file(Path(args.xml_file)))
        metrics["parser_parse_file_seconds"] = d3

    scraper = BundestagScraper(request_delay=0, probe_delay=0)
    _, d4, _ = _timer(
        "scraper_probe_50_sessions_seconds",
        lambda: scraper.probe_dserver_window(20, max_sessions=50),
    )
    metrics["scraper_probe_50_sessions_seconds"] = d4

    _current, peak = tracemalloc.get_traced_memory()
    metrics["memory_peak_mb"] = round(peak / (1024 * 1024), 2)
    tracemalloc.stop()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
