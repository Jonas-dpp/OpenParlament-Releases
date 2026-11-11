from datetime import date
import random
from src.parser import (
    get_or_create_redner,
    _parse_date,
    BundestagXMLParser,
)
from src.database import get_session, init_db


def test_get_or_create_redner():
    init_db()
    # Test line 47-68
    with get_session() as db_session:
        rand_id = f"test_{random.randint(10000000, 99999999)}"
        # 1. Create a new redner
        redner, created = get_or_create_redner(db_session, redner_id=rand_id, vorname="Olaf", nachname="Scholz", fraktion="SPD")
        assert created is True
        assert redner.bundestag_id == rand_id
        assert redner.vorname == "Olaf"
        db_session.flush()
        # 2. Get existing redner
        redner2, created2 = get_or_create_redner(db_session, redner_id=rand_id)
        assert created2 is False
        assert redner2 is redner


def test_parse_date_edge_cases():
    # Test line 123-125
    assert _parse_date("2026-03-31") == date(2026, 3, 31)
    assert _parse_date("31.03.2026") == date(2026, 3, 31)
    assert _parse_date("invalid-date") is None


def test_local_name():
    # Test line 425-427
    assert BundestagXMLParser._local_name(123) == ""
    assert BundestagXMLParser._local_name("{http://example.com}tagname") == "tagname"
    assert BundestagXMLParser._local_name("plain_tag") == "plain_tag"


def test_materialize_redner_existing():
    init_db()
    # Test line 464, 472-473, 488
    with get_session() as db_session:
        parser = BundestagXMLParser()
        existing = {}
        rand_id = f"test_{random.randint(10000000, 99999999)}"
        # 1. First materialized (saves both keys and adds to session)
        data = {"bundestag_id": rand_id, "vorname": "Angela", "nachname": "Merkel"}
        r1 = parser._materialize_redner(data, existing, session=db_session)
        assert r1.bundestag_id == rand_id
        assert r1 in db_session
        db_session.flush()
        # 2. Hit line 464 (cache hit)
        r2 = parser._materialize_redner(data, existing, session=db_session)
        assert r2 is r1
        # 3. DB query hit inside materialize with new cache map (line 472-473)
        existing_empty = {}
        r3 = parser._materialize_redner(data, existing_empty, session=db_session)
        assert r3.bundestag_id == r1.bundestag_id
        assert r3 is r1


def test_parse_kommentar_and_classifier():
    parser = BundestagXMLParser()
    # empty comment (line 502)
    assert parser._parse_kommentar("()") is None
    assert parser._parse_kommentar("  ") is None
    # detect faction via regex fallback (line 519-520)
    z1 = parser._parse_kommentar("(Heiterkeit bei der CUSTOMPARTY)")
    assert z1 is not None
    # category classifications (line 531-537)
    assert parser._classify_kommentar("ruft dazwischen") == "Zwischenruf"
    assert parser._classify_kommentar("Ein Zuruf") == "Zwischenruf"
    assert parser._classify_kommentar("sehr richtig!") == "Zustimmung"
    assert parser._classify_kommentar("bravo") == "Zustimmung"
    assert parser._classify_kommentar("something else entirely") == "Sonstiges"


def test_parse_stream_seek_exception():
    # Test line 181-182: stream without seek/tell support
    class NonSeekableStream:
        def __init__(self):
            self.data = b"""<?xml version="1.0" encoding="utf-8"?>
            <dbtplenarprotokoll>
                <wahlperiode>20</wahlperiode>
                <sitzungsnr>123</sitzungsnr>
                <datum date="31.03.2026">31.03.2026</datum>
            </dbtplenarprotokoll>"""
            self.read_done = False
        def read(self, *args, **kwargs):
            if self.read_done:
                return b""
            self.read_done = True
            return self.data
        def seek(self, *args, **kwargs):
            raise OSError("No seek support")
        def tell(self):
            return 0
    parser = BundestagXMLParser()
    res = parser._parse_stream(NonSeekableStream())
    assert res is not None
    assert res.sitzung.wahlperiode == 20


def test_parse_sitzungsnr_warnings(caplog):
    # Test line 394 and 396: non-standard sitzungsnr and unparseable sitzungsnr
    xml_non_standard = b"""<?xml version="1.0" encoding="utf-8"?>
    <dbtplenarprotokoll>
        <wahlperiode>20</wahlperiode>
        <sitzungsnr>23b</sitzungsnr>
        <datum date="31.03.2026">31.03.2026</datum>
    </dbtplenarprotokoll>
    """
    parser = BundestagXMLParser()
    res = parser.parse_bytes(xml_non_standard)
    assert res.sitzung.sitzungsnr == 23
    assert any("Non-standard sitzungsnr" in r.message for r in caplog.records)
    xml_unparseable = b"""<?xml version="1.0" encoding="utf-8"?>
    <dbtplenarprotokoll>
        <wahlperiode>20</wahlperiode>
        <sitzungsnr>abc</sitzungsnr>
        <datum date="31.03.2026">31.03.2026</datum>
    </dbtplenarprotokoll>
    """
    caplog.clear()
    res2 = parser.parse_bytes(xml_unparseable)
    assert res2.sitzung.sitzungsnr == 0
    assert any("Could not parse sitzungsnr" in r.message for r in caplog.records)


def test_parse_stream_fallback_wahlperiode_and_unknown_redner():
    # Test line 251-252: fallback_wahlperiode and line 293 unknown redner
    xml = b"""<?xml version="1.0" encoding="utf-8"?>
    <dbtplenarprotokoll>
        <wahlperiode>20</wahlperiode>
        <sitzungsnr>123</sitzungsnr>
        <datum date="31.03.2026">31.03.2026</datum>
        <sitzungstitel>Titel</sitzungstitel>
        <rede>
            <p>Some speech text without specified speaker.</p>
        </rede>
    </dbtplenarprotokoll>
    """
    parser = BundestagXMLParser()
    res = parser.parse_bytes(xml)
    assert res.sitzung.wahlperiode == 20
    assert len(res.reden) == 1
    assert res.reden[0].redner.nachname == "Redner"


def test_parse_stream_top_attribute_variations_and_anlagen_text():
    xml = b"""<?xml version="1.0" encoding="utf-8"?>
    <dbtplenarprotokoll>
        <wahlperiode>20</wahlperiode>
        <sitzungsnr>123</sitzungsnr>
        <sitzungsverlauf>
            <tagesordnungspunkt top-nr="TOP 10">
                <rede id="ID201230100">
                    <p>Rede in TOP 10</p>
                </rede>
            </tagesordnungspunkt>
            <tagesordnungspunkt top-label="Zusatzpunkt 5">
                <rede id="ID201230200">
                    <p>Rede in Zusatzpunkt 5</p>
                </rede>
            </tagesordnungspunkt>
        </sitzungsverlauf>
        <anlagen>
            <anlage>
                <anlagen-text>
                    <p klasse="Anlage_3">(Tagesordnungspunkt 16)</p>
                    <rede id="ID201230300">
                        <p>Zu Protokoll gegebene Rede</p>
                    </rede>
                </anlagen-text>
            </anlage>
        </anlagen>
    </dbtplenarprotokoll>
    """
    parser = BundestagXMLParser()
    res = parser.parse_bytes(xml)
    assert len(res.reden) == 3
    assert res.reden[0].tagesordnungspunkt == "TOP 10"
    assert res.reden[1].tagesordnungspunkt == "Zusatzpunkt 5"
    assert res.reden[2].tagesordnungspunkt == "Tagesordnungspunkt 16"
