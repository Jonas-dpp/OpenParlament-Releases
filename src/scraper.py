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
Bundestag Open Data scraper for OpenParlament.

Downloads plenary protocol XML files from the official Bundestag open-data
portal and stores them locally.

Usage
─────
    scraper = BundestagScraper(download_dir=Path("data/xml"))
    urls = scraper.fetch_protocol_urls(wahlperiode=20)
    scraper.download_all(urls)

URL discovery strategy (two sources)
─────────────────────────────────────
1. **dserver (primary / complete)**
   ``dserver.bundestag.de`` serves XML files at a fully predictable path::

       https://dserver.bundestag.de/btp/{wp}/{wp}{sn:03d}.xml

   Because the URL contains no opaque blob-ID, we can probe every session
   number systematically.  Each candidate is probed with a streaming GET
   request (HEAD is avoided because some servers return 405 or incorrect
   status codes for HEAD).  Probing stops automatically after
   ``_MAX_CONSECUTIVE_MISSES`` consecutive non-200 responses.  This is the
   *only* reliable way to enumerate all sessions – the filterlist API cannot
   page past its internal TYPO3 offset limit.

2. **Filterlist API (secondary / seed)**
   ``bundestag.de/ajax/filterlist/…`` returns recently published protocols.
   It is kept as a supplementary source to catch any file not yet propagated
   to the dserver mirror, but pagination is guarded by a **stall detector**:
   if a page contains no new URLs (all already seen from the dserver pass)
   the loop terminates immediately instead of burning through pointless
   requests.

To enumerate multiple Wahlperioden at once use ``fetch_all_wahlperioden``::

    scraper = BundestagScraper(download_dir=Path("data/xml"))
    urls = scraper.fetch_all_wahlperioden([18, 19, 20])

The scraper is deliberately conservative: it respects HTTP rate limits
(configurable delay between requests) and skips files that have already been
downloaded (idempotent).
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from lxml import etree

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_BASE_URL = "https://www.bundestag.de"

# The Bundestag open-data listing page (filter endpoint with XML protocol links).
# The page number is controlled by the `offset` query parameter.
# NOTE: No query parameters here – they are passed explicitly in _fetch_page()
# to avoid duplicate parameters in the final request URL.
#
# The path segment "866354-866354" is the CMS node ID of the Bundestag
# Plenarprotokolle open-data page.  It is stable across requests; only the
# known-good value "866354" is accepted by the server – shorter, longer, or
# otherwise different values return HTTP 400.
_OPENDATA_URL = (
    "https://www.bundestag.de/ajax/filterlist/de/services/opendata/"
    "866354-866354"
)

# Root XML element that identifies a genuine Bundestag plenary-protocol file.
# Any downloaded XML whose root tag differs from this value is rejected.
_EXPECTED_ROOT_TAG = "dbtplenarprotokoll"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "OpenParlament/1.1.0 (Automatisierte Analyse v. BT-Protokollen zur Untersuchung v. Kommunikation und Dynamiken; "
        "academic research)"
    ),
    "Accept-Language": "de-DE,de;q=0.9",
}

_XML_LINK_PATTERN = re.compile(r"\.xml$", re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# dserver constants
# ─────────────────────────────────────────────────────────────────────────────

# dserver.bundestag.de hosts plenary protocol XMLs at a predictable path:
#   /btp/{wahlperiode}/{wahlperiode}{sitzungsnr:03d}.xml
# Unlike the blob-storage URLs, no opaque CMS ID is required.
_DSERVER_BASE = "https://dserver.bundestag.de/btp"

# Stop probing dserver after this many consecutive HTTP 404 responses past any
# known seed session limit.
# 404 is treated as the only definitive "no such session" signal for the stop
# condition; transient 5xx/429/network failures are retried and do not advance
# the consecutive-miss stop counter.
_MAX_CONSECUTIVE_MISSES = 15

# Number of times to retry a dserver probe on a 5xx (server-side) error
# before giving up and counting the attempt as a miss.
_MAX_RETRIES = 3

# Connection timeout (seconds) for dserver probe requests.  Probes only need
# the HTTP status line – the response body is never read – so 5 seconds is
# more than sufficient.  Downloads use a separate, longer timeout (30 s).
_PROBE_TIMEOUT = 5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_url(url: str) -> str:
    """Return a sanitized version of the URL with query string, fragment, and credentials removed."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or "unknown"
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return f"{parsed.scheme}://{host}{parsed.path}"
    except Exception:
        return "[malformed URL]"


def _url_filename(url: str) -> str:
    """Return the lowercase bare filename of a URL for deduplication.

    Both ``https://dserver.bundestag.de/btp/20/20001.xml`` and
    ``https://www.bundestag.de/resource/blob/XXX/20001.xml`` resolve to the
    same session; comparing bare filenames prevents the same session from
    appearing twice when it is discovered via both sources.

    Uses :func:`urllib.parse.urlparse` to correctly handle query strings
    (``?download=true``), URL fragments (``#section``), encoded characters,
    and other edge cases that chained ``.split()`` calls would mishandle.

    :param url: Absolute URL of an XML file.
    :return: Lowercase filename with no path, query-string, or fragment.
    """
    return Path(urlparse(url).path).name.lower()


# ─────────────────────────────────────────────────────────────────────────────
# BundestagScraper
# ─────────────────────────────────────────────────────────────────────────────

class BundestagScraper:
    """Download plenary protocol XML files from the Bundestag open-data portal.

    Parameters
    ──────────
    download_dir : Path
        Directory where XML files are saved.  Created automatically.
    request_delay : float
        Seconds to wait between filterlist-API requests and between file
        downloads (be a good citizen).  Default: ``1.0``.
    probe_delay : float
        Seconds to wait between individual dserver probe requests.  This can
        be much shorter than *request_delay* because probes only read the HTTP
        status line (stream=True, body never transferred).  Default: ``0.05``
        (20 req/s), which is respectful without being prohibitively slow.
    skip_existing : bool
        When ``True`` (the default) files that already exist in *download_dir*
        are not re-downloaded.  Set to ``False`` to force re-downloading every
        file even if it was previously downloaded.
    session : requests.Session, optional
        Inject a custom session (useful for testing / mocking).
    """

    def __init__(
        self,
        download_dir: Path = Path("data/xml"),
        request_delay: float = 1.0,
        probe_delay: float = 0.05,
        skip_existing: bool = True,
        session: Optional[requests.Session] = None,
        allowed_domains: Optional[set[str] | list[str]] = None,
        max_consecutive_misses: int = _MAX_CONSECUTIVE_MISSES,
    ) -> None:
        if not isinstance(request_delay, (int, float)) or request_delay < 0:
            raise ValueError("request_delay must be a non-negative number.")
        if not isinstance(probe_delay, (int, float)) or probe_delay < 0:
            raise ValueError("probe_delay must be a non-negative number.")
        if not isinstance(max_consecutive_misses, int) or max_consecutive_misses <= 0:
            raise ValueError("max_consecutive_misses must be a positive integer.")
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.probe_delay = probe_delay
        self.skip_existing = skip_existing
        self.max_consecutive_misses = max_consecutive_misses
        raw_allowed = (
            set(allowed_domains)
            if allowed_domains is not None
            else {"www.bundestag.de", "dserver.bundestag.de"}
        )
        self.allowed_domains = {domain.lower().rstrip(".") for domain in raw_allowed}
        self._session = session or self._build_session()

    # ── Public interface ──────────────────────────────────────────────────────

    def fetch_protocol_urls(
        self,
        wahlperiode: int,
        max_pages: int = 100,
        max_consecutive_misses: Optional[int] = None,
    ) -> List[str]:
        """Return a deduplicated list of absolute XML download URLs for *wahlperiode*.

        Uses two complementary sources:

        1. **Filterlist API** (secondary / seed) – queries recently published
           files first to extract seed session numbers.
        2. **dserver** (primary) – probes the deterministic URL pattern on
           ``dserver.bundestag.de`` up to the seed maximum and beyond to enumerate
           *all* sessions without stopping prematurely on session numbering gaps.

        Deduplication is **filename-based**: the same session number discovered
        via dserver (``…/btp/20/20001.xml``) and via the filterlist
        (``…/resource/blob/XXX/20001.xml``) shares the same bare filename
        ``20001.xml`` and will therefore be collected only once, preferring the
        canonical dserver URL.

        :param wahlperiode: The Wahlperiode to scrape (e.g. 20).
        :param max_pages: Safety cap on filterlist pagination iterations.
        :param max_consecutive_misses: Consecutive 404 count required to terminate
            dserver probing past the seed maximum. Defaults to self.max_consecutive_misses.
        :return: Deduplicated list of absolute URLs pointing to XML files.
        """
        if not isinstance(wahlperiode, int) or wahlperiode <= 0:
            raise ValueError("Wahlperiode must be a positive integer.")
        if not isinstance(max_pages, int) or max_pages <= 0:
            raise ValueError("max_pages must be a positive integer.")

        miss_limit = (
            max_consecutive_misses
            if max_consecutive_misses is not None
            else self.max_consecutive_misses
        )
        if not isinstance(miss_limit, int) or miss_limit <= 0:
            raise ValueError("max_consecutive_misses must be a positive integer.")

        seen_filenames: set = set()
        urls: List[str] = []

        # ── 1. Secondary: Filterlist API (seed / recent uploads) ──────────────
        logger.info(
            "Fetching seed URLs from filterlist API for Wahlperiode %d…", wahlperiode
        )
        filterlist_urls: List[str] = []
        seed_sessions: set = set()
        wp_pattern = re.compile(rf"^{re.escape(str(wahlperiode))}(\d+)\.xml$", re.IGNORECASE)

        offset = 0
        limit = 10
        start_time = time.time()
        for page in range(max_pages):
            page_idx = page + 1
            page_urls = self._fetch_page(offset, limit, wahlperiode)
            is_break = False

            if not page_urls:
                logger.info(
                    "Filterlist returned no URLs at offset=%d; stopping.", offset
                )
                is_break = True
            else:
                new_found = 0
                for url in page_urls:
                    fname = _url_filename(url)
                    if fname not in seen_filenames:
                        seen_filenames.add(fname)
                        filterlist_urls.append(url)
                        new_found += 1
                        m = wp_pattern.match(fname)
                        if m:
                            try:
                                seed_sessions.add(int(m.group(1)))
                            except ValueError:
                                pass
                if new_found == 0:
                    logger.info(
                        "Filterlist stall detected at offset=%d (page %d); "
                        "all results already seen.",
                        offset,
                        page,
                    )
                    is_break = True

            # Progress / ETA tracking for Filterlist Query
            if page_idx == 1 or page_idx % 10 == 0 or page_idx == max_pages or is_break:
                elapsed = time.time() - start_time
                avg_time = elapsed / page_idx
                remaining_pages = 0 if is_break else (max_pages - page_idx)
                eta = avg_time * remaining_pages
                if eta >= 3600:
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                else:
                    eta_str = time.strftime('%M:%S', time.gmtime(eta))
                logger.info(
                    f"FilterlistQuery: {page_idx}/{max_pages} | {int((page_idx/max_pages)*100)}% | ETA: {eta_str}"
                )

            if is_break:
                break

            offset += limit
            time.sleep(self.request_delay)

        max_seed_sn = max(seed_sessions) if seed_sessions else 0

        # ── 2. Primary: dserver (complete, deterministic) ─────────────────────
        logger.info(
            "Probing dserver for Wahlperiode %d XML files (seed max session: %d)…",
            wahlperiode,
            max_seed_sn,
        )
        dserver_urls = self._fetch_via_dserver(
            wahlperiode,
            min_sessions=max_seed_sn,
            max_consecutive_misses=miss_limit,
        )

        # Dserver URLs are canonical — include all dserver results
        urls = list(dserver_urls)
        dserver_fnames = {_url_filename(u) for u in dserver_urls}

        # Supplement with filterlist URLs for sessions not found on dserver
        for url in filterlist_urls:
            fname = _url_filename(url)
            if fname not in dserver_fnames:
                urls.append(url)
                dserver_fnames.add(fname)

        logger.info(
            "Found %d total protocol URLs for Wahlperiode %d.", len(urls), wahlperiode
        )
        return urls

    def fetch_all_wahlperioden(
        self,
        wahlperioden: List[int],
        max_pages: int = 100,
        max_workers: Optional[int] = None,
    ) -> List[str]:
        if not isinstance(max_pages, int) or max_pages <= 0:
            raise ValueError("max_pages must be a positive integer.")
        if max_workers is not None:
            if not isinstance(max_workers, int) or max_workers <= 0:
                raise ValueError("max_workers must be a positive integer.")
        for wp in wahlperioden:
            if not isinstance(wp, int) or wp <= 0:
                raise ValueError("Wahlperiode must be a positive integer.")
        """Return deduplicated XML URLs for all given Wahlperioden.

        When *max_workers* is ``None`` (the default) the degree of parallelism
        is chosen automatically as ``min(8, len(wahlperioden))``.  For IO-bound
        HTTP work this provides a significant wall-clock speedup without
        saturating the server.  Pass ``max_workers=1`` to force strictly
        sequential execution (useful in tests or when a single WP is given).

        Each parallel worker thread gets its own :class:`BundestagScraper`
        instance (and therefore its own ``requests.Session``) because
        ``requests.Session`` is not thread-safe.

        :param wahlperioden: Ordered list of Wahlperioden to scrape
            (e.g. ``[18, 19, 20]``).
        :param max_pages: Safety cap forwarded to each
            :meth:`fetch_protocol_urls` call.
        :param max_workers: Number of threads for parallel Wahlperioden
            discovery.  ``None`` (default) auto-selects
            ``min(8, len(wahlperioden))``.
        :return: Flat, deduplicated list of XML download URLs.
        """
        for wp in wahlperioden:
            if not isinstance(wp, int) or wp <= 0:
                raise ValueError("Wahlperiode must be a positive integer.")

        effective_workers = (
            min(8, len(wahlperioden))
            if max_workers is None
            else max_workers
        )

        if effective_workers > 1:
            def _fetch_one_wahlperiode(wp: int) -> List[str]:
                # Each thread gets its own BundestagScraper (and therefore
                # its own requests.Session) to avoid session sharing across
                # threads, which is not safe.
                worker = BundestagScraper(
                    download_dir=self.download_dir,
                    request_delay=self.request_delay,
                    probe_delay=self.probe_delay,
                    skip_existing=self.skip_existing,
                    allowed_domains=self.allowed_domains,
                    max_consecutive_misses=self.max_consecutive_misses,
                )
                return worker.fetch_protocol_urls(wp, max_pages=max_pages)

            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                wp_results = list(executor.map(_fetch_one_wahlperiode, wahlperioden))
        else:
            wp_results = [
                self.fetch_protocol_urls(wp, max_pages=max_pages)
                for wp in wahlperioden
            ]

        seen_filenames: set = set()
        all_urls: List[str] = []

        for wp_urls in wp_results:
            for url in wp_urls:
                fname = _url_filename(url)
                if fname not in seen_filenames:
                    seen_filenames.add(fname)
                    all_urls.append(url)

        logger.info(
            "fetch_all_wahlperioden: %d total URLs across %d Wahlperioden.",
            len(all_urls),
            len(wahlperioden),
        )
        return all_urls

    def download_all(self, urls: List[str]) -> List[Path]:
        """Download all *urls* and return a list of local file paths.

        Already-downloaded files are skipped without any delay.  A summary of
        newly downloaded vs already-present files is printed at INFO level so
        the operator can see what happened without trawling through debug logs.

        :param urls: List of absolute XML download URLs.
        :return: Paths to the local XML files (both newly downloaded and
            pre-existing).
        """
        paths: List[Path] = []
        total = len(urls)
        downloaded = 0
        skipped = 0

        if total == 0:
            logger.info("Finished: 0/0 files ready (0 newly downloaded, 0 already on disk, 0 failed).")
            return paths

        start_time = time.time()
        for i, url in enumerate(urls, 1):
            # Security: Validate URL prior to any file resolution or disk exists check (defense-in-depth)
            parsed = urlparse(url)
            if parsed.username or parsed.password:
                logger.warning("Rejected URL with credentials in download_all")
                continue
            clean_hostname = (parsed.hostname or "").lower().rstrip(".")
            if (
                parsed.scheme != "https"
                or not clean_hostname
                or clean_hostname not in self.allowed_domains
            ):
                logger.warning("Rejected insecure or invalid URL scheme/domain in download_all: %s", _sanitize_url(url))
                continue

            filename = _url_filename(url)
            # Security: Path traversal prevention before .exists() check
            unquoted_filename = unquote(filename)
            if ".." in unquoted_filename or "/" in unquoted_filename or "\\" in unquoted_filename:
                logger.warning("Rejected potential path traversal filename in download_all: %s", _sanitize_url(url))
                continue

            dest = self.download_dir / (filename if filename.endswith(".xml") else filename + ".xml")
            # For statistics, treat a file as "already on disk" when it exists and
            # skip_existing is enabled. The actual XML-root validation and decision
            # to reuse or redownload the file is delegated to download_one().
            pre_existed = dest.exists() and self.skip_existing
            path = self.download_one(url)
            if path:
                if pre_existed:
                    skipped += 1
                else:
                    downloaded += 1
                paths.append(path)

            # Throttling logging: log first, every 10th, and last
            if i == 1 or i % 10 == 0 or i == total:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (total - i)
                if eta >= 3600:
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                else:
                    eta_str = time.strftime('%M:%S', time.gmtime(eta))
                logger.info(
                    f"ProtocolDownload: {i}/{total} | {int((i/total)*100)}% | ETA: {eta_str}"
                )
            logger.debug("[%d/%d] %s", i, total, url)

        logger.info(
            "Finished: %d/%d files ready  (%d newly downloaded, %d already on disk, %d failed).",
            len(paths), total, downloaded, skipped, total - len(paths),
        )
        return paths

    def download_one(self, url: str) -> Optional[Path]:
        """Download a single XML file.  Returns local path or None on failure.

        After downloading, the file content is validated: only files whose XML
        root element is ``dbtplenarprotokoll`` are kept.  Any other XML (or
        non-XML response) is discarded and ``None`` is returned.
        """
        # Security: URL scheme and domain validation (SSRF mitigation)
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            logger.warning("Rejected URL with credentials")
            return None
        if parsed.scheme != "https":
            logger.warning("Rejected insecure or invalid URL scheme: %s", _sanitize_url(url))
            return None
        clean_hostname = (parsed.hostname or "").lower().rstrip(".")
        if not clean_hostname or clean_hostname not in self.allowed_domains:
            logger.warning("Rejected URL with untrusted domain: %s", _sanitize_url(url))
            return None

        filename = _url_filename(url)
        # Security: Path traversal prevention
        unquoted_filename = unquote(filename)
        if ".." in unquoted_filename or "/" in unquoted_filename or "\\" in unquoted_filename:
            logger.warning("Rejected URL with potential path traversal: %s", _sanitize_url(url))
            return None

        if not filename.endswith(".xml"):
            filename = filename + ".xml"
        dest = self.download_dir / filename

        # Skip only if the existing file is readable and valid.
        if dest.exists() and self.skip_existing:
            try:
                existing = dest.read_bytes()
            except OSError as exc:
                logger.warning(
                    "Could not read existing file %s (%s); re-downloading.", dest, exc
                )
            else:
                if _is_dbtplenarprotokoll(existing):
                    logger.info("Skipping %s (already downloaded and valid)", _sanitize_url(url))
                    return dest
                logger.warning(
                    "Existing file %s is invalid (unexpected XML root); re-downloading.",
                    dest,
                )

        try:
            resp = self._secure_get(url, timeout=30)
            resp.raise_for_status()
            content = resp.content
        except Exception as exc:
            logger.error("Failed to download %s: %s", _sanitize_url(url), exc)
            return None

        if not _is_dbtplenarprotokoll(content):
            logger.warning(
                "Skipping %s: root element is not <%s>", _sanitize_url(url), _EXPECTED_ROOT_TAG
            )
            return None

        dest.write_bytes(content)
        logger.info("Downloaded: %s → %s", _sanitize_url(url), dest)
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        return dest

    def _secure_get(self, url: str, timeout: float, stream: bool = False) -> requests.Response:
        """Fetch a URL securely by manually validating every redirect target before making the request.

        This prevents SSRF attacks via open redirects to internal/private IP ranges or untrusted domains.
        """
        if not isinstance(url, str):
            raise ValueError("URL must be a string")
        max_redirects = 5
        current_url = url

        for _ in range(max_redirects + 1):
            if not isinstance(current_url, str):
                raise ValueError("URL must be a string")
            parsed = urlparse(current_url)
            if parsed.username or parsed.password:
                raise ValueError("URL with credentials rejected")
            if parsed.scheme != "https":
                raise ValueError(f"Insecure or invalid URL scheme: {_sanitize_url(current_url)}")
            clean_hostname = (parsed.hostname or "").lower().rstrip(".")
            if not clean_hostname or clean_hostname not in self.allowed_domains:
                raise ValueError(f"URL with untrusted domain rejected: {_sanitize_url(current_url)}")

            resp = self._session.get(current_url, timeout=timeout, allow_redirects=False, stream=stream)

            if isinstance(resp.status_code, int) and 300 <= resp.status_code < 400 and "Location" in resp.headers:
                next_url = resp.headers["Location"]
                if isinstance(next_url, str):
                    if not next_url.startswith("http"):
                        next_url = urljoin(str(current_url), next_url)
                    current_url = next_url
                else:
                    raise ValueError("Location header must be a string")
                if stream:
                    resp.close()
            else:
                return resp

        raise ValueError("Too many redirects")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_page(
        self, offset: int, limit: int, wahlperiode: int
    ) -> List[str]:
        """Scrape one paginated results page and return XML URLs."""
        if not isinstance(wahlperiode, int) or wahlperiode <= 0:
            raise ValueError("Wahlperiode must be a positive integer.")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

        params = {
            "limit": limit,
            "offset": offset,
            "noFilterSet": "true",
        }
        try:
            resp = self._session.get(_OPENDATA_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Error fetching listing page (offset=%d): %s", offset, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        urls: List[str] = []

        # Compile a precise pattern that matches "<wahlperiode><digits>.xml" only,
        # e.g. "20001.xml" for Wahlperiode 20.  This is more robust than a plain
        # startswith() check, which would accept arbitrary XML files whose names
        # happen to begin with the wahlperiode digits.
        wp_filename_pattern = re.compile(
            rf"^{re.escape(str(wahlperiode))}\d+\.xml$", re.IGNORECASE
        )

        for a_tag in soup.find_all("a", href=True):
            href: str = a_tag["href"]
            # Keep only XML links that match the target Wahlperiode.
            if _XML_LINK_PATTERN.search(href):
                abs_url = href if href.startswith("http") else urljoin(_BASE_URL, href)
                filename = _url_filename(abs_url)
                if wp_filename_pattern.match(filename):
                    urls.append(abs_url)

        return urls

    def _fetch_via_dserver(
        self,
        wahlperiode: int,
        max_sessions: int = 300,
        max_consecutive_misses: Optional[int] = None,
        min_sessions: int = 0,
    ) -> List[str]:
        """Probe the deterministic dserver URL space for *wahlperiode*.

        Sends a streaming GET request (not HEAD – HEAD can return 405 or
        unreliable status codes on some servers) for each candidate session
        number (``1`` to *max_sessions* or *min_sessions*).  Only the HTTP status
        code is inspected; the body is never read.  A URL is included in the result
        when the server responds with HTTP 200.  Probing stops early once
        *max_consecutive_misses* consecutive 404 responses have been received
        past *min_sessions*, which reliably signals the end of a Wahlperiode's
        session list while tolerating numbering gaps below *min_sessions*.
        Transient 5xx/429 errors are retried up to ``_MAX_RETRIES`` times
        before being treated as a non-terminal outcome.

        :param wahlperiode: The Wahlperiode to scan (e.g. 20).
        :param max_sessions: Hard upper bound on session numbers to probe.
        :param max_consecutive_misses: Consecutive 404 threshold past min_sessions.
        :param min_sessions: Minimum session number to probe regardless of gaps.
        :return: List of dserver XML URLs that returned HTTP 200.
        """
        if not isinstance(wahlperiode, int) or wahlperiode <= 0:
            raise ValueError("Wahlperiode must be a positive integer.")
        if not isinstance(max_sessions, int) or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive integer.")

        miss_limit = (
            max_consecutive_misses
            if max_consecutive_misses is not None
            else self.max_consecutive_misses
        )
        if not isinstance(miss_limit, int) or miss_limit <= 0:
            raise ValueError("max_consecutive_misses must be a positive integer.")

        urls: List[str] = []
        consecutive_404 = 0

        target_max = max(max_sessions, min_sessions)
        recent_times: list[float] = []

        for sn in range(1, target_max + 1):
            t0 = time.time()
            url = self._dserver_xml_url(wahlperiode, sn)
            status = self._probe_url_status(url)
            t_probe = time.time() - t0
            recent_times.append(t_probe)
            if len(recent_times) > 20:
                recent_times.pop(0)

            is_break = False
            if status == 200:
                urls.append(url)
                consecutive_404 = 0
            elif status == 404:
                logger.debug("Miss: %s", url)
                consecutive_404 += 1
                if sn >= min_sessions and consecutive_404 >= miss_limit:
                    is_break = True
            else:
                logger.debug(
                    "Transient probe outcome (%s) for %s; not counting as terminal miss.",
                    status,
                    url,
                )

            if is_break:
                eta_str = "00:00"
            else:
                avg_time = sum(recent_times) / len(recent_times) if recent_times else 0.05
                expected_end = max(min_sessions, sn + (miss_limit - consecutive_404))
                expected_end = min(expected_end, target_max)
                remaining_probes = max(0, expected_end - sn)
                eta = avg_time * remaining_probes
                if eta >= 3600:
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                else:
                    eta_str = time.strftime('%M:%S', time.gmtime(eta))

            # Throttling logging: log first, every 20th, when breaking, and last
            if sn == 1 or sn % 20 == 0 or sn == target_max or is_break:
                pct = int((sn / target_max) * 100) if target_max > 0 else 100
                logger.info(
                    f"DserverProbe: {sn}/{target_max} | {pct}% | ETA: {eta_str}"
                )

            if is_break:
                logger.info(
                    "Stopping dserver probe after %d consecutive 404s "
                    "(last sitzungsnr=%d, min_sessions=%d).",
                    miss_limit,
                    sn,
                    min_sessions,
                )
                break
            time.sleep(self.probe_delay)

        logger.info(
            "dserver probe found %d URLs for Wahlperiode %d.", len(urls), wahlperiode
        )
        return urls

    def probe_dserver_window(
        self,
        wahlperiode: int,
        max_sessions: int = 300,
        max_consecutive_misses: Optional[int] = None,
    ) -> List[str]:
        """Public wrapper for deterministic dserver probing (benchmarking/ops use)."""
        return self._fetch_via_dserver(
            wahlperiode=wahlperiode,
            max_sessions=max_sessions,
            max_consecutive_misses=max_consecutive_misses,
        )

    def _probe_url(self, url: str) -> int:
        """Probe *url* with a streaming GET and return ``200`` on success or ``0``.

        Uses ``stream=True`` so the response body is never transferred, making
        the request as lightweight as a HEAD while remaining compatible with
        servers that reject HEAD (405).  The connection timeout is
        ``_PROBE_TIMEOUT`` seconds – sufficient to receive the HTTP status line
        without waiting for a body that will never be read.

        Only HTTP 200 is treated as "found".  Any other non-5xx response (e.g.
        404) returns ``0`` immediately without retry, since these are definitive
        client-side misses.  Transient 5xx responses are retried up to
        ``_MAX_RETRIES`` times before returning ``0``.  Network-level exceptions
        are also retried; ``0`` is returned if all attempts fail.  The wait
        between retries uses :attr:`probe_delay` (not the heavier
        :attr:`request_delay` that governs downloads).

        :param url: URL to probe.
        :return: ``200`` if the server confirmed the resource exists; ``0`` for
            any other outcome (missing resource, server error, or network
            exception).
        """
        status = self._probe_url_status(url)
        return 200 if status == 200 else 0

    def _probe_url_status(self, url: str) -> int:
        """Probe URL and return raw status code (0 for exhausted network errors)."""
        # Security: URL scheme and domain validation (SSRF mitigation)
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            logger.warning("Rejected URL with credentials during probe")
            return 0
        if parsed.scheme != "https":
            logger.warning("Rejected insecure or invalid URL scheme during probe: %s", _sanitize_url(url))
            return 0
        clean_hostname = (parsed.hostname or "").lower().rstrip(".")
        if not clean_hostname or clean_hostname not in self.allowed_domains:
            logger.warning("Rejected URL with untrusted domain during probe: %s", _sanitize_url(url))
            return 0

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._secure_get(url, timeout=_PROBE_TIMEOUT, stream=True)
                resp.close()
                if resp.status_code == 200:
                    return 200
                if resp.status_code == 404:
                    return 404
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = None
                    if retry_after:
                        try:
                            sleep_time = float(retry_after)
                        except (ValueError, TypeError):
                            sleep_time = None
                    if sleep_time is None or sleep_time <= 0:
                        sleep_time = max(1.0, self.probe_delay * (2 ** (attempt + 2)))
                    logger.warning(
                        "dserver rate limit encountered (HTTP 429) for %s; backing off for %.2f seconds (attempt %d/%d)",
                        _sanitize_url(url),
                        sleep_time,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(sleep_time)
                elif resp.status_code >= 500:
                    logger.warning(
                        "dserver returned %d for %s (attempt %d/%d)",
                        resp.status_code,
                        _sanitize_url(url),
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    if attempt < _MAX_RETRIES - 1:
                        backoff = self.probe_delay * (2 ** attempt)
                        time.sleep(backoff)
                else:
                    return resp.status_code
            except Exception as exc:
                logger.warning(
                    "dserver GET failed for %s: %s (attempt %d/%d)",
                    _sanitize_url(url),
                    exc,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                if attempt < _MAX_RETRIES - 1:
                    backoff = self.probe_delay * (2 ** attempt)
                    time.sleep(backoff)
        return 0

    @staticmethod
    def _dserver_xml_url(wahlperiode: int, sitzungsnr: int) -> str:
        """Return the deterministic dserver URL for a single session.

        Pattern: ``/btp/{wp}/{wp}{sn:03d}.xml``

        Examples::

            _dserver_xml_url(20, 1)   → "https://dserver.bundestag.de/btp/20/20001.xml"
            _dserver_xml_url(20, 214) → "https://dserver.bundestag.de/btp/20/20214.xml"
            _dserver_xml_url(19, 12)  → "https://dserver.bundestag.de/btp/19/19012.xml"
        """
        return f"{_DSERVER_BASE}/{wahlperiode}/{wahlperiode}{sitzungsnr:03d}.xml"

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update(_DEFAULT_HEADERS)
        return session


def _is_dbtplenarprotokoll(content: bytes) -> bool:
    """Return ``True`` if *content* is a Bundestag plenary-protocol XML.

    Performs a fast root-element check using lxml without loading the entire
    document into memory.  Any parse error or unexpected root tag causes the
    function to return ``False`` so that callers can safely discard the file.

    The parser is configured with ``resolve_entities=False`` and
    ``no_network=True`` to guard against XML entity-expansion attacks (XML
    bombs) and server-side request forgery via external entity references.

    :param content: Raw bytes of the downloaded XML file.
    :return: ``True`` iff the root element tag equals ``"dbtplenarprotokoll"``.
    """
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        root = etree.fromstring(content, parser=parser)
        return root.tag == _EXPECTED_ROOT_TAG
    except etree.XMLSyntaxError:
        return False
