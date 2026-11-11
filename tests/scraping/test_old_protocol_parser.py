"""
Unit and integration tests for the generic old Bundestag protocol parser.
"""

import tempfile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from scripts.old_protocol_parser import (
    clean_hyphens,
    parse_name,
    is_valid_speaker,
    parse_speaker_header,
    build_redner_xml,
    parse_old_toc,
    extract_comments_from_text,
    parse_old_protocol,
    load_mdb_database
)


def test_clean_hyphens():
    """Test that clean_hyphens removes end-of-line hyphens and normalizes space."""
    input_text = "Grüß Gott, liebe Kolleginnen und Kollegen! Die Sit-\nzung ist eröffnet."
    expected = "Grüß Gott, liebe Kolleginnen und Kollegen! Die Sitzung ist eröffnet."
    assert clean_hyphens(input_text) == expected
    assert clean_hyphens("Hello   \n   World-  \n  wide") == "Hello Worldwide"


def test_parse_name():
    """Test parsing names into structured parts."""
    res1 = parse_name("Dr. h. c. Susanne Kastner")
    assert res1["vorname"] == "Susanne"
    assert res1["nachname"] == "Kastner"
    assert res1["titel"] == "Dr. h. c."
    assert res1["von_oder_zu"] is None

    res2 = parse_name("Eckart von Klaeden")
    assert res2["vorname"] == "Eckart"
    assert res2["von_oder_zu"] == "von"
    assert res2["nachname"] == "Klaeden"
    assert res2["titel"] is None

    res3 = parse_name("Hans Georg Wagner")
    assert res3["vorname"] == "Hans"
    assert res3["nachname"] == "Georg Wagner"


def test_is_valid_speaker():
    """Test classification of lines as speaker declarations."""
    assert is_valid_speaker("Vizepräsidentin Dr. h. c. Susanne Kastner:")
    assert is_valid_speaker("Eckart von Klaeden (CDU/CSU):")
    assert is_valid_speaker("Hans Georg Wagner, Parl. Staatssekretär:")
    assert not is_valid_speaker("Ich rufe Tagesordnungspunkt 2 auf:")
    assert not is_valid_speaker("(Peter Dreßen [SPD]: Das war klar und deutlich!)")


def test_parse_speaker_header_generic():
    """Test parsing speaker headers dynamically with regex."""
    res = parse_speaker_header("Eckart von Klaeden (CDU/CSU):")
    assert res["vorname"] == "Eckart"
    assert res["von_oder_zu"] == "von"
    assert res["nachname"] == "Klaeden"
    assert res["fraktion"] == "CDU/CSU"
    assert res["role_lang"] is None

    res2 = parse_speaker_header("Hans Georg Wagner, Parl. Staatssekretär beim Bundesminister der Verteidigung:")
    assert res2["vorname"] == "Hans"
    assert res2["nachname"] == "Georg Wagner"
    assert "Staatssekretär" in res2["role_lang"]
    assert res2["role_kurz"] == "BMVg"


def test_mdb_database_lookup():
    """Test loading MdB database and resolving speakers by name."""
    mock_data = [
        {
            "id": "11003518",
            "vorname": "Gitta",
            "nachname": "Connemann",
            "fraktion": "CDU/CSU"
        }
    ]

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
        json.dump(mock_data, tf)
        tf.flush()
        temp_path = tf.name

    try:
        db = load_mdb_database(temp_path)
        assert db is not None
        assert len(db) == 1

        # Test look up
        res = parse_speaker_header("Gitta Connemann:", mdb_db=db)
        assert res["id"] == "11003518"
        assert res["fraktion"] == "CDU/CSU"

        # Test unmatched fallback
        res2 = parse_speaker_header("Unbekannter Redner:", mdb_db=db)
        assert res2["id"].startswith("1100")
        assert res2["id"] != "11003518"
    finally:
        Path(temp_path).unlink()


def test_extract_comments_from_text():
    """Test extraction of inline and multi-line comments."""
    text = "Herr Kollege, Ihre erste Frage beantworte ich mit Nein. (Peter Dreßen [SPD]: Das war klar und deutlich!)"
    parts = extract_comments_from_text(text)
    assert len(parts) == 2
    assert parts[0] == ("text", "Herr Kollege, Ihre erste Frage beantworte ich mit Nein.")
    assert parts[1] == ("comment", "(Peter Dreßen [SPD]: Das war klar und deutlich!)")


def test_parse_old_toc_generic():
    """Test Table of Contents parsing."""
    toc_text = (
        "Tagesordnungspunkt 2:\n"
        "Fragestunde\n"
        "(Drucksachen 15/523, 15/529)  . . . . . . . . . 2273 A\n"
        "ZusFr Gitta Connemann CDU/CSU . . . . . . . 2309 A"
    )
    blocks = parse_old_toc(toc_text)
    assert len(blocks) == 1
    assert blocks[0]["title"] == "Tagesordnungspunkt 2:"

    entries = blocks[0]["entries"]
    assert len(entries) == 2
    assert entries[0]["text"] == "Fragestunde (Drucksachen 15/523, 15/529)"
    assert entries[0]["page"] == "2273"
    assert entries[0]["area"] == "A"

    assert entries[1]["text"] == "ZusFr Gitta Connemann CDU/CSU"
    assert entries[1]["page"] == "2309"
    assert entries[1]["area"] == "A"


def test_parse_old_protocol_generic_integration():
    """Integration test parsing the entire old format input xml to target XML using the generic parser."""
    input_file = Path("data/input_wp15_s30.xml")
    if not input_file.exists():
        pytest.skip("Please run the PoC generation step first.")

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    output_xml = parse_old_protocol(content)

    root = ET.fromstring(output_xml)

    # Check attributes
    assert root.tag == "dbtplenarprotokoll"
    assert root.attrib["wahlperiode"] == "15"
    assert root.attrib["sitzung-nr"] == "30"
    assert root.attrib["sitzung-datum"] == "12.03.2003"
    assert root.attrib["sitzung-start-uhrzeit"] == "13:00"
    assert root.attrib["start-seitennr"] == "2273"

    # Check kopfdaten
    vorspann = root.find("vorspann")
    assert vorspann is not None
    kopfdaten = vorspann.find("kopfdaten")
    assert kopfdaten is not None
    assert kopfdaten.find("sitzungstitel").text == "30. Sitzung"

    # Check sitzungsverlauf
    verlauf = root.find("sitzungsverlauf")
    assert verlauf is not None

    # Check sitzungsbeginn
    beginn = verlauf.find("sitzungsbeginn")
    assert beginn is not None
    assert "Susanne Kastner" in "".join(beginn.find("p[@klasse='N']").itertext())

    # Check tagesordnungspunkt
    top = verlauf.find("tagesordnungspunkt")
    assert top is not None
    assert top.attrib["top-id"] == "Tagesordnungspunkt 1"

    # Check speeches
    reden = top.findall("rede")
    assert len(reden) >= 2

    # First speech
    wagner_rede = reden[0]
    wagner_p_redner = wagner_rede.find("p[@klasse='redner']")
    assert wagner_p_redner is not None
    assert "Hans Georg Wagner" in "".join(wagner_p_redner.itertext())

    # Comments inside first speech
    assert wagner_rede.find("kommentar") is not None
    assert "Peter Dreßen" in wagner_rede.find("kommentar").text


def test_xml_escaping_raw_strings():
    """Test XML escaping of inputs with special characters like <, >, &."""
    # Test speaker builder escaping
    speaker = {
        "id": "1100&1",
        "vorname": "Gitta <",
        "von_oder_zu": None,
        "nachname": "Connemann >",
        "fraktion": "CDU/CSU & Others"
    }
    xml_str = build_redner_xml(speaker)
    assert 'id="1100&amp;1"' in xml_str
    assert '<vorname>Gitta &lt;</vorname>' in xml_str
    assert '<nachname>Connemann &gt;</nachname>' in xml_str
    assert '<fraktion>CDU/CSU &amp; Others</fraktion>' in xml_str


def test_load_mdb_database_security_path_traversal():
    """Test that load_mdb_database rejects unsafe paths to prevent path traversal."""
    assert load_mdb_database("/etc/passwd") is None
    assert load_mdb_database("../../outside_file.json") is None


def test_parse_old_protocol_xxe_prevention():
    """Test that parse_old_protocol prevents XML External Entity (XXE) expansion attacks."""
    xxe_xml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE DOKUMENT [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<DOKUMENT>
    <WAHLPERIODE>15</WAHLPERIODE>
    <NR>15/30</NR>
    <DATUM>12.03.2003</DATUM>
    <TEXT>Inhalt: TOP 1
    Sitzung
    Beginn: 13:00 Uhr
    &xxe;
    Eckart von Klaeden (CDU/CSU):
    Das ist eine Rede.</TEXT>
</DOKUMENT>"""
    parsed_output = parse_old_protocol(xxe_xml)
    assert "root:x:0:0" not in parsed_output
    assert "/bin/bash" not in parsed_output
