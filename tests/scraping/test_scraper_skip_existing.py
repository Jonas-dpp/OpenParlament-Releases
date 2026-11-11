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
"""
from __future__ import annotations

import time
from pathlib import Path

from src.scraper import BundestagScraper


_VALID_XML = b'<?xml version="1.0" encoding="UTF-8"?><dbtplenarprotokoll></dbtplenarprotokoll>'
_INVALID_XML = b'<?xml version="1.0" encoding="UTF-8"?><not_the_right_root></not_the_right_root>'


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200, url: str = "", history: list = None, headers: dict = None) -> None:
        self.content = content
        self.status_code = status_code
        self.url = url
        self.history = history or []
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self) -> None:
        pass

    def iter_content(self, chunk_size: int = 8192):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


class _FakeSession:
    def __init__(self, response: _FakeResponse | dict[str, _FakeResponse]) -> None:
        self._response = response
        self.get_calls: list[tuple[str, float]] = []

    def get(self, url: str, timeout: float = 30, **kwargs):
        self.get_calls.append((url, timeout))
        if isinstance(self._response, dict):
            resp = self._response.get(url, _FakeResponse(b"", status_code=404))
        else:
            resp = self._response
        if not getattr(resp, "url", None):
            resp.url = url
        return resp


def test_skip_existing_only_if_valid(tmp_path: Path) -> None:
    url = "https://example.test/20001.xml"
    dest = tmp_path / "20001.xml"

    # Case 1: existing file valid => skip without HTTP call
    dest.write_bytes(_VALID_XML)
    session = _FakeSession(_FakeResponse(_VALID_XML))
    scraper = BundestagScraper(
        download_dir=tmp_path,
        skip_existing=True,
        session=session,
        allowed_domains={"example.test"},
    )

    out = scraper.download_one(url)
    assert out == dest
    assert session.get_calls == []
    assert dest.read_bytes() == _VALID_XML

    # Case 2: existing file invalid => re-download (HTTP call) and overwrite
    dest.write_bytes(_INVALID_XML)
    session2 = _FakeSession(_FakeResponse(_VALID_XML))
    # request_delay=0 to avoid actual sleeping in tests
    scraper2 = BundestagScraper(
        download_dir=tmp_path,
        skip_existing=True,
        session=session2,
        request_delay=0,
        allowed_domains={"example.test"},
    )

    out2 = scraper2.download_one(url)
    assert out2 == dest
    assert len(session2.get_calls) == 1
    assert dest.read_bytes() == _VALID_XML


def test_no_sleep_when_skipping(tmp_path: Path) -> None:
    """download_one must not sleep when the file is already cached."""
    url = "https://example.test/20001.xml"
    dest = tmp_path / "20001.xml"
    dest.write_bytes(_VALID_XML)

    session = _FakeSession(_FakeResponse(_VALID_XML))
    # Use a non-zero delay so we'd notice if it fired
    scraper = BundestagScraper(
        download_dir=tmp_path,
        skip_existing=True,
        session=session,
        request_delay=5.0,
        allowed_domains={"example.test"},
    )

    start = time.monotonic()
    out = scraper.download_one(url)
    elapsed = time.monotonic() - start

    assert out == dest
    assert session.get_calls == []       # no HTTP request
    assert elapsed < 1.0                 # no sleep


def test_sleep_only_on_actual_download(tmp_path: Path) -> None:
    """download_all must not sleep for already-cached files."""
    urls = [
        "https://example.test/20001.xml",
        "https://example.test/20002.xml",
    ]
    for name in ("20001.xml", "20002.xml"):
        (tmp_path / name).write_bytes(_VALID_XML)

    session = _FakeSession(_FakeResponse(_VALID_XML))
    scraper = BundestagScraper(
        download_dir=tmp_path,
        skip_existing=True,
        session=session,
        request_delay=5.0,
        allowed_domains={"example.test"},
    )

    start = time.monotonic()
    paths = scraper.download_all(urls)
    elapsed = time.monotonic() - start

    assert len(paths) == 2
    assert session.get_calls == []  # all cached, no HTTP
    assert elapsed < 1.0            # no sleep at all


def test_security_url_validation(tmp_path: Path) -> None:
    """Validate that only https scheme and allowed domains are allowed."""
    session = _FakeSession(_FakeResponse(_VALID_XML))
    scraper = BundestagScraper(
        download_dir=tmp_path,
        allowed_domains={"example.test"},
        session=session,
    )

    # Insecure schemes
    assert scraper.download_one("http://example.test/20001.xml") is None
    assert scraper.download_one("gopher://example.test/20001.xml") is None
    assert scraper.download_one("file:///etc/passwd") is None

    # Untrusted domains
    assert scraper.download_one("https://malicious.test/20001.xml") is None
    assert scraper.download_one("https://attacker.com/20001.xml") is None

    # Rejects URLs with embedded credentials (SSRF & Phishing prevention)
    assert scraper.download_one("https://user:pass@example.test/20001.xml") is None
    assert scraper.download_one("https://user@example.test/20001.xml") is None
    assert scraper._probe_url_status("https://user:pass@example.test/20001.xml") == 0

    # Allows URLs with an explicit port on an allowed domain
    assert scraper.download_one("https://example.test:8443/20001.xml") is not None
    assert len(session.get_calls) == 1
    assert session.get_calls[0][0] == "https://example.test:8443/20001.xml"


def test_security_path_traversal(tmp_path: Path) -> None:
    """Validate that path traversal elements in URL filename are rejected."""
    session = _FakeSession(_FakeResponse(_VALID_XML))
    scraper = BundestagScraper(
        download_dir=tmp_path,
        allowed_domains={"example.test"},
        session=session,
    )

    # Test cases that yield relative paths or traversal patterns
    assert scraper.download_one("https://example.test/..%2f..%2f20001.xml") is None
    assert scraper.download_one("https://example.test/subdir/..%5c20001.xml") is None
    assert scraper.download_one("https://example.test/../../20001.xml") == tmp_path / "20001.xml"
    # Additional percent-encoded traversal bypass cases
    assert scraper.download_one("https://example.test/%2e%2e%2f20001.xml") is None
    assert scraper.download_one("https://example.test/%2e%2e%2f%2e%2e%2f20001.xml") is None
    assert scraper.download_one("https://example.test/%2e%2e%5c20001.xml") is None
    assert scraper.download_one("https://example.test/%2e%2e%5c%2e%2e%5c20001.xml") is None


def test_security_redirect_validation(tmp_path: Path) -> None:
    """Validate secure manual redirect handling and domain enforcement."""
    from unittest.mock import patch

    # 1. Blocked: trusted domain -> redirect -> untrusted domain
    response_map_bad_ext = {
        "https://example.test/source.xml": _FakeResponse(
            b"",
            status_code=302,
            headers={"Location": "https://attacker.com/malicious.xml"}
        ),
        "https://attacker.com/malicious.xml": _FakeResponse(
            _VALID_XML,
            status_code=200
        )
    }
    session = _FakeSession(response_map_bad_ext)
    scraper = BundestagScraper(
        download_dir=tmp_path,
        allowed_domains={"example.test"},
        session=session,
    )
    # The file download should fail (returns None) because it attempts to redirect to untrusted attacker.com
    assert scraper.download_one("https://example.test/source.xml") is None
    # The URL probe path should also fail (returns 0)
    assert scraper._probe_url_status("https://example.test/source.xml") == 0

    # 2. Blocked: trusted domain -> redirect -> localhost/private IP
    response_map_bad_int = {
        "https://example.test/source.xml": _FakeResponse(
            b"",
            status_code=302,
            headers={"Location": "https://127.0.0.1/admin"}
        ),
        "https://127.0.0.1/admin": _FakeResponse(
            _VALID_XML,
            status_code=200
        )
    }
    session2 = _FakeSession(response_map_bad_int)
    scraper2 = BundestagScraper(
        download_dir=tmp_path,
        allowed_domains={"example.test"},
        session=session2,
    )
    assert scraper2.download_one("https://example.test/source.xml") is None
    assert scraper2._probe_url_status("https://example.test/source.xml") == 0

    # 3. Allowed: trusted GitHub release URL -> trusted GitHub asset URL
    response_map_good = {
        "https://github.com/source.db": _FakeResponse(
            b"database-bytes",
            status_code=302,
            headers={"Location": "https://objects.githubusercontent.com/asset.db"}
        ),
        "https://objects.githubusercontent.com/asset.db": _FakeResponse(
            b"database-bytes",
            status_code=200
        )
    }
    from src.app import _secure_db_get
    allowed_db_domains = {
        "github.com",
        "objects.githubusercontent.com",
        "github-releases.githubusercontent.com",
        "release-assets.githubusercontent.com"
    }

    # Mock requests.get inside src.app with _FakeSession's get
    session3 = _FakeSession(response_map_good)
    with patch("requests.get", side_effect=session3.get):
        resp = _secure_db_get("https://github.com/source.db", allowed_db_domains, timeout=30)
        assert resp.status_code == 200
        assert resp.content == b"database-bytes"

        # Test rejection of embedded credentials in _secure_db_get
        import pytest
        with pytest.raises(RuntimeError, match="URL with credentials rejected"):
            _secure_db_get("https://user:pass@github.com/source.db", allowed_db_domains, timeout=30)

        # Test rejection of embedded credentials with insecure scheme in _secure_db_get
        with pytest.raises(RuntimeError, match="URL with credentials rejected"):
            _secure_db_get("http://user:pass@github.com/source.db", allowed_db_domains, timeout=30)


def test_security_wahlperiode_validation(tmp_path: Path) -> None:
    """Validate that wahlperiode must be a positive integer in BundestagScraper."""
    import pytest
    session = _FakeSession(_FakeResponse(_VALID_XML))
    scraper = BundestagScraper(
        download_dir=tmp_path,
        allowed_domains={"example.test"},
        session=session,
    )

    # 1. fetch_protocol_urls validation
    with pytest.raises(ValueError, match="Wahlperiode must be a positive integer."):
        scraper.fetch_protocol_urls("invalid")
    with pytest.raises(ValueError, match="Wahlperiode must be a positive integer."):
        scraper.fetch_protocol_urls(-1)
    with pytest.raises(ValueError, match="Wahlperiode must be a positive integer."):
        scraper.fetch_protocol_urls(0)

    # 2. _fetch_via_dserver validation
    with pytest.raises(ValueError, match="Wahlperiode must be a positive integer."):
        scraper._fetch_via_dserver("invalid")
    with pytest.raises(ValueError, match="Wahlperiode must be a positive integer."):
        scraper._fetch_via_dserver(0)

    # 3. _fetch_page validation
    with pytest.raises(ValueError, match="Wahlperiode must be a positive integer."):
        scraper._fetch_page(0, 10, "invalid")
    with pytest.raises(ValueError, match="Wahlperiode must be a positive integer."):
        scraper._fetch_page(0, 10, -5)

    # 4. fetch_all_wahlperioden validation
    with pytest.raises(ValueError, match="Wahlperiode must be a positive integer."):
        scraper.fetch_all_wahlperioden([20, "invalid"])
    with pytest.raises(ValueError, match="Wahlperiode must be a positive integer."):
        scraper.fetch_all_wahlperioden([20, 0])


def test_security_parameter_validations(tmp_path: Path) -> None:
    """Validate that parameters like max_pages, max_workers, offset, limit, delays, and max_sessions are secure."""
    import pytest
    session = _FakeSession(_FakeResponse(_VALID_XML))

    # 1. Constructor delays validation
    with pytest.raises(ValueError, match="request_delay must be a non-negative number."):
        BundestagScraper(download_dir=tmp_path, request_delay=-1.0, session=session)
    with pytest.raises(ValueError, match="request_delay must be a non-negative number."):
        BundestagScraper(download_dir=tmp_path, request_delay="invalid", session=session)
    with pytest.raises(ValueError, match="probe_delay must be a non-negative number."):
        BundestagScraper(download_dir=tmp_path, probe_delay=-0.5, session=session)
    with pytest.raises(ValueError, match="probe_delay must be a non-negative number."):
        BundestagScraper(download_dir=tmp_path, probe_delay="invalid", session=session)

    scraper = BundestagScraper(
        download_dir=tmp_path,
        allowed_domains={"example.test"},
        session=session,
    )

    # 2. max_pages validation
    with pytest.raises(ValueError, match="max_pages must be a positive integer."):
        scraper.fetch_protocol_urls(20, max_pages=0)
    with pytest.raises(ValueError, match="max_pages must be a positive integer."):
        scraper.fetch_protocol_urls(20, max_pages=-5)
    with pytest.raises(ValueError, match="max_pages must be a positive integer."):
        scraper.fetch_protocol_urls(20, max_pages="invalid")

    # 3. fetch_all_wahlperioden max_pages and max_workers validation
    with pytest.raises(ValueError, match="max_pages must be a positive integer."):
        scraper.fetch_all_wahlperioden([20], max_pages=0)
    with pytest.raises(ValueError, match="max_workers must be a positive integer."):
        scraper.fetch_all_wahlperioden([20], max_workers=0)
    with pytest.raises(ValueError, match="max_workers must be a positive integer."):
        scraper.fetch_all_wahlperioden([20], max_workers=-1)
    with pytest.raises(ValueError, match="max_workers must be a positive integer."):
        scraper.fetch_all_wahlperioden([20], max_workers="invalid")

    # 4. _fetch_page offset and limit validation
    with pytest.raises(ValueError, match="offset must be a non-negative integer."):
        scraper._fetch_page(offset=-1, limit=10, wahlperiode=20)
    with pytest.raises(ValueError, match="offset must be a non-negative integer."):
        scraper._fetch_page(offset="invalid", limit=10, wahlperiode=20)
    with pytest.raises(ValueError, match="limit must be a positive integer."):
        scraper._fetch_page(offset=0, limit=0, wahlperiode=20)
    with pytest.raises(ValueError, match="limit must be a positive integer."):
        scraper._fetch_page(offset=0, limit=-10, wahlperiode=20)
    with pytest.raises(ValueError, match="limit must be a positive integer."):
        scraper._fetch_page(offset=0, limit="invalid", wahlperiode=20)

    # 5. _fetch_via_dserver max_sessions validation
    with pytest.raises(ValueError, match="max_sessions must be a positive integer."):
        scraper._fetch_via_dserver(wahlperiode=20, max_sessions=0)
    with pytest.raises(ValueError, match="max_sessions must be a positive integer."):
        scraper._fetch_via_dserver(wahlperiode=20, max_sessions=-100)
    with pytest.raises(ValueError, match="max_sessions must be a positive integer."):
        scraper._fetch_via_dserver(wahlperiode=20, max_sessions="invalid")


def test_security_db_header_validation(tmp_path: Path, monkeypatch) -> None:
    """Validate that get_db_path() correctly validates the SQLite format 3 magic bytes of the downloaded file."""
    import os
    import pytest
    from unittest.mock import patch
    from src.app import get_db_path

    # Redirect data_path to tmp_path using os.path.join mock
    orig_join = os.path.join
    def mock_join(path, *paths):
        if path == "data" and "openparlament.db" in paths:
            return str(tmp_path / "openparlament.db")
        return orig_join(path, *paths)

    # Mock responses for the DB download
    response_map_bad = {
        "https://github.com/Jonas-dpp/OpenParlament-online/releases/download/v2.5.2/openparlament.db": _FakeResponse(
            b"This is a malicious payload or HTML error page",
            status_code=200
        )
    }

    session = _FakeSession(response_map_bad)
    with patch("os.path.join", side_effect=mock_join), patch("requests.get", side_effect=session.get):
        # We must clear the cache of get_db_path (st.cache_resource) so it executes
        try:
            get_db_path.clear()
        except AttributeError:
            pass
        with pytest.raises(RuntimeError, match="Database file has invalid SQLite format 3 header"):
            get_db_path()

    # Verify that the temp/part files were deleted/cleaned up
    assert not (tmp_path / "openparlament.db.part").exists()
    assert not (tmp_path / "openparlament.db").exists()

    # Valid response (proper SQLite magic bytes)
    valid_db_payload = b"SQLite format 3\x00" + b"A" * 1000
    response_map_good = {
        "https://github.com/Jonas-dpp/OpenParlament-online/releases/download/v2.5.2/openparlament.db": _FakeResponse(
            valid_db_payload,
            status_code=200
        )
    }

    session2 = _FakeSession(response_map_good)
    with patch("os.path.join", side_effect=mock_join), patch("requests.get", side_effect=session2.get):
        # We must clear the cache of get_db_path (st.cache_resource) so it executes again
        try:
            get_db_path.clear()
        except AttributeError:
            pass
        returned_path = get_db_path()
        assert Path(returned_path).exists()
        assert Path(returned_path).read_bytes() == valid_db_payload


def test_security_domain_normalization(tmp_path: Path) -> None:
    """Validate that allowed_domains and input/redirect URLs with trailing dots or mixed cases are normalized and matched."""
    from src.app import _secure_db_get
    from unittest.mock import patch

    # 1. Test BundestagScraper with mixed-case and trailing-dot configurations
    session = _FakeSession(_FakeResponse(_VALID_XML))
    scraper = BundestagScraper(
        download_dir=tmp_path,
        allowed_domains={"Example.Test.", "sub.domain.com"},
        session=session,
    )

    # Re-normalize check: both "Example.Test." and "example.test" are accepted
    assert scraper.download_one("https://example.test/20001.xml") is not None
    assert scraper.download_one("https://example.test./20001.xml") is not None
    assert scraper.download_one("https://EXAMPLE.TEST/20001.xml") is not None
    assert scraper.download_one("https://EXAMPLE.TEST./20001.xml") is not None

    # 2. Test _secure_db_get with mixed-case and trailing-dot configurations
    response_map = {
        "https://github.com/source.db": _FakeResponse(
            b"data",
            status_code=200
        ),
        "https://GITHUB.COM./source.db": _FakeResponse(
            b"data",
            status_code=200
        )
    }
    session2 = _FakeSession(response_map)
    allowed_db_domains = {"GitHub.com.", "Objects.GitHubUserContent.com"}
    with patch("requests.get", side_effect=session2.get):
        resp1 = _secure_db_get("https://github.com/source.db", allowed_db_domains, timeout=30)
        assert resp1.status_code == 200

        resp2 = _secure_db_get("https://GITHUB.COM./source.db", allowed_db_domains, timeout=30)
        assert resp2.status_code == 200


def test_security_type_safety_and_redirects(tmp_path: Path) -> None:
    """Validate that secure getters strictly enforce that URLs and Location headers are strings."""
    import pytest
    from src.app import _secure_db_get
    from src.scraper import BundestagScraper
    from unittest.mock import patch

    # 1. Test _secure_db_get URL type check
    with pytest.raises(RuntimeError, match="URL must be a string"):
        _secure_db_get(123, {"github.com"}, timeout=30)

    # 2. Test _secure_db_get non-string Location redirect header
    response_map_bad_header = {
        "https://github.com/source.db": _FakeResponse(
            b"",
            status_code=302,
            headers={"Location": 9999}  # non-string header
        )
    }
    session = _FakeSession(response_map_bad_header)
    with patch("requests.get", side_effect=session.get):
        with pytest.raises(RuntimeError, match="Location header must be a string"):
            _secure_db_get("https://github.com/source.db", {"github.com"}, timeout=30)

    # 3. Test BundestagScraper _secure_get URL type check
    scraper = BundestagScraper(download_dir=tmp_path, allowed_domains={"example.test"})
    with pytest.raises(ValueError, match="URL must be a string"):
        scraper._secure_get(None, timeout=30)

    # 4. Test BundestagScraper _secure_get non-string Location redirect header
    response_map_bad_header_scr = {
        "https://example.test/source.xml": _FakeResponse(
            b"",
            status_code=302,
            headers={"Location": ["http://another.com"]}  # list instead of string
        )
    }
    session_scr = _FakeSession(response_map_bad_header_scr)
    scraper_scr = BundestagScraper(
        download_dir=tmp_path,
        allowed_domains={"example.test"},
        session=session_scr
    )
    with pytest.raises(ValueError, match="Location header must be a string"):
        scraper_scr._secure_get("https://example.test/source.xml", timeout=30)


def test_dserver_gap_tolerance_and_min_sessions(tmp_path: Path) -> None:
    """Verify that _fetch_via_dserver tolerates gaps and respects min_sessions."""
    from unittest.mock import MagicMock

    hits = {1, 2, 3, 14}

    def fake_get(url: str, **kwargs):
        sn = int(url.split("/")[-1].replace(".xml", "")[-3:])
        m = MagicMock()
        m.status_code = 200 if sn in hits else 404
        return m

    session = MagicMock()
    session.get.side_effect = fake_get

    # 1. With max_consecutive_misses=15, 10 consecutive misses (sn 4..13) do NOT stop probing!
    scraper = BundestagScraper(
        download_dir=tmp_path,
        request_delay=0,
        probe_delay=0,
        max_consecutive_misses=15,
        session=session,
    )
    urls = scraper._fetch_via_dserver(wahlperiode=20, max_sessions=50)

    assert len(urls) == 4
    assert BundestagScraper._dserver_xml_url(20, 14) in urls

    # 2. With min_sessions=20, probing must continue up to sn=20 even if max_consecutive_misses is lower
    session2 = MagicMock()
    session2.get.side_effect = fake_get
    scraper2 = BundestagScraper(
        download_dir=tmp_path,
        request_delay=0,
        probe_delay=0,
        max_consecutive_misses=5,
        session=session2,
    )
    urls2 = scraper2._fetch_via_dserver(wahlperiode=20, max_sessions=50, min_sessions=20)
    assert len(urls2) == 4
    assert BundestagScraper._dserver_xml_url(20, 14) in urls2


def test_rate_limiting_http_429_backoff_and_retry_after(tmp_path: Path) -> None:
    """Verify HTTP 429 rate limiting handles Retry-After header and exponential backoff."""
    from unittest.mock import MagicMock, patch

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0.1"}

    resp_200 = MagicMock()
    resp_200.status_code = 200

    session = MagicMock()
    session.get.side_effect = [resp_429, resp_200]

    scraper = BundestagScraper(
        download_dir=tmp_path,
        probe_delay=0.01,
        allowed_domains={"dserver.bundestag.de"},
        session=session,
    )

    sleep_calls = []
    with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        status = scraper._probe_url_status("https://dserver.bundestag.de/btp/20/20001.xml")

    assert status == 200
    assert 0.1 in sleep_calls


def test_eta_calculation_accuracy_on_early_break(tmp_path: Path, caplog) -> None:
    """Verify that ETA is logged as 00:00 when early break occurs."""
    import logging
    from unittest.mock import MagicMock

    def fake_get(url: str, **kwargs):
        sn = int(url.split("/")[-1].replace(".xml", "")[-3:])
        m = MagicMock()
        m.status_code = 200 if sn == 1 else 404
        return m

    session = MagicMock()
    session.get.side_effect = fake_get

    scraper = BundestagScraper(
        download_dir=tmp_path,
        request_delay=0,
        probe_delay=0,
        max_consecutive_misses=5,
        session=session,
    )

    with caplog.at_level(logging.INFO, logger="src.scraper"):
        scraper._fetch_via_dserver(wahlperiode=20, max_sessions=100)

    break_logs = [r.message for r in caplog.records if "DserverProbe:" in r.message and "100%" not in r.message]
    assert len(break_logs) > 0
    last_log = break_logs[-1]
    assert "ETA: 00:00" in last_log
