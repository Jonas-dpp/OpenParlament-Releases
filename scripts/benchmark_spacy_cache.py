#!/usr/bin/env python3
"""
Benchmark script to measure the performance improvement of cached vs. non-cached spaCy model loads.
"""

from __future__ import annotations

import time
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

def main() -> None:
    print("=== spaCy Loading Benchmark ===")

    # 1. Measure non-cached loading (simulated)
    print("\nRunning simulated non-cached loads...")
    import spacy

    start_time = time.perf_counter()
    print("Loading non-cached run 1...")
    _ = spacy.load("de_core_news_sm", disable=["ner"])
    run1_time = time.perf_counter() - start_time
    print(f"Non-cached run 1 took: {run1_time:.4f} seconds")

    start_time2 = time.perf_counter()
    print("Loading non-cached run 2...")
    _ = spacy.load("de_core_news_sm", disable=["ner"])
    run2_time = time.perf_counter() - start_time2
    print(f"Non-cached run 2 took: {run2_time:.4f} seconds")

    total_non_cached = run1_time + run2_time
    print(f"Total simulated non-cached loading time: {total_non_cached:.4f} seconds")

    # 2. Measure cached loading
    print("\nRunning cached loads...")
    from src.nlp import get_cached_spacy_model, _SPACY_CACHE
    _SPACY_CACHE.clear()  # Clear cache to ensure a fresh load

    start_cached = time.perf_counter()
    print("Loading cached run 1 (cache miss)...")
    _ = get_cached_spacy_model("de_core_news_sm", disable=["ner"])
    cached_run1_time = time.perf_counter() - start_cached
    print(f"Cached run 1 took: {cached_run1_time:.4f} seconds")

    start_cached2 = time.perf_counter()
    print("Loading cached run 2 (cache hit)...")
    _ = get_cached_spacy_model("de_core_news_sm", disable=["ner"])
    cached_run2_time = time.perf_counter() - start_cached2
    print(f"Cached run 2 took: {cached_run2_time:.4f} seconds")

    total_cached = cached_run1_time + cached_run2_time
    print(f"Total cached loading time: {total_cached:.4f} seconds")

    # 3. Summary
    speedup_run2 = (run2_time / cached_run2_time) if cached_run2_time > 0 else float('inf')
    total_speedup = (total_non_cached / total_cached) if total_cached > 0 else float('inf')

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Non-cached loading total: {total_non_cached:.4f}s")
    print(f"Cached loading total:     {total_cached:.4f}s")
    print(f"Total Speedup:            {total_speedup:.2f}x faster")
    print(f"Cache Hit Speedup (Run 2): {speedup_run2:.2f}x faster")

if __name__ == "__main__":
    main()
