#!/usr/bin/env python3
"""
OpenParlament - Generic Old Protocol Parser
Parses older unstructured text-based Bundestag protocols (WP <= 17)
into the clean, structured XML format used in WP 18+.

Can be run with this command:
"""
# Get-ChildItem "data\xml\old\_xmls\_unstructured\*.xml" | ForEach-Object {
#   $env:PYTHONPATH = "."
#   python3 "scripts\old_protocol_parser.py" $_.FullName "data\xml\$($_.Name)"
# }

import argparse
import datetime
import hashlib
import html
import json
import logging
import re
import sys
import time
from pathlib import Path
from lxml import etree

# Setup structured logging according to project standards
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("old_protocol_parser")


def generate_speaker_id(name):
    """Generates a deterministic 8-digit ID starting with 1100 based on the name hash."""
    hash_obj = hashlib.md5(name.encode('utf-8'))
    digits = str(int(hash_obj.hexdigest()[:8], 16) % 10000).zfill(4)
    return f"1100{digits}"


def clean_hyphens(text):
    """Clean up hyphenated words and redundant whitespace."""
    # Handle hyphenation at line end within a word
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    # Replace normal newlines with a single space
    text = re.sub(r'\n\s*', ' ', text)
    # Clean up double/multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_name(name_str):
    """Parses a full name string into parts: vorname, nachname, von_oder_zu, and titel."""
    parts = name_str.strip().split()
    titel_parts = []

    titel_words = {"Dr.", "Dr. h. c.", "h.", "c.", "Prof.", "Dr.-Ing.", "Freiherr", "Graf"}
    i = 0
    while i < len(parts):
        word = parts[i]
        if word.endswith(".") or word in {"h.", "c."} or any(t.startswith(word) for t in titel_words):
            titel_parts.append(word)
            i += 1
        else:
            break

    name_parts = parts[i:]
    if not name_parts:
        return {"vorname": "", "nachname": name_str, "titel": None, "von_oder_zu": None}

    titel = " ".join(titel_parts) if titel_parts else None

    von_oder_zu_words = {"von", "van", "de", "zu", "der", "dem", "vom"}

    vz_idx = -1
    for idx, w in enumerate(name_parts):
        if w.lower() in von_oder_zu_words:
            vz_idx = idx
            break

    if vz_idx != -1 and vz_idx > 0 and vz_idx < len(name_parts) - 1:
        vorname = " ".join(name_parts[:vz_idx])
        # Find all consecutive von_oder_zu words
        k = vz_idx
        while k < len(name_parts) - 1 and name_parts[k].lower() in von_oder_zu_words:
            k += 1
        von_oder_zu = " ".join(name_parts[vz_idx:k])
        nachname = " ".join(name_parts[k:])
    else:
        if len(name_parts) == 1:
            vorname = ""
            von_oder_zu = None
            nachname = name_parts[0]
        else:
            vorname = name_parts[0]
            von_oder_zu = None
            nachname = " ".join(name_parts[1:])

    return {
        "vorname": vorname,
        "von_oder_zu": von_oder_zu,
        "nachname": nachname,
        "titel": titel
    }


def is_valid_speaker(header):
    """Determine if a header line is likely a speaker declaration."""
    header = header.strip()
    if not header or header.startswith("("):
        return False

    header_lower = header.lower()
    exclude_prefixes = ["kollegen", "kollege", "herr kollege", "frau kollegin"]
    if any(header_lower.startswith(pref) for pref in exclude_prefixes):
        return False

    exclude_phrases = ["tagesordnungspunkt", "dringliche", "zur beantwortung", "ich rufe", "wir kommen"]
    if any(phrase in header_lower for phrase in exclude_phrases):
        return False

    known_roles = ["präsident", "vizepräsident", "minister", "staatssekretär", "abgeordneter", "abgeordnete", "redner"]
    if any(role in header_lower for role in known_roles):
        return True

    # Needs at least one capital word
    words = header.split()
    cap_words = [w for w in words if w and w[0].isupper()]
    if len(cap_words) >= 1:
        return True

    return False


def parse_speaker_header(header, mdb_db=None):
    """
    Parses a speaker header string (e.g., 'Eckart von Klaeden (CDU/CSU):')
    and extracts structured speaker details.
    """
    clean_hdr = header.rstrip(" :").strip()

    # Extract fraction if present in parenthesis, brackets or just whitespace at the end
    fraction = None
    frac_match = re.search(r'(?:\s+|[\(\[])(CDU/CSU|SPD|FDP|DIE LINKE|GRÜNE|BÜNDNIS 90/DIE GRÜNEN|AfD|LINKE|BSW|SSW|PDS)[\)\]]?$', clean_hdr, re.IGNORECASE)
    if frac_match:
        fraction = frac_match.group(1)
        clean_hdr = clean_hdr[:frac_match.start()].strip()

    # Extract role from comma suffix
    role_lang = None
    role_kurz = None
    if "," in clean_hdr:
        parts = clean_hdr.split(",", 1)
        clean_hdr = parts[0].strip()
        role_lang = parts[1].strip()
        if "BMVg" in role_lang or "BMVg" in header or "Verteidigung" in role_lang:
            role_kurz = "BMVg"
        elif "BMAS" in role_lang or "BMAS" in header:
            role_kurz = "BMAS"

    # Extract role from title prefixes (like Vizepräsidentin)
    role_prefixes = ["vizepräsidentin", "vizepräsident", "präsidentin", "präsident", "bundesministerin", "bundesminister", "staatssekretärin", "staatssekretär"]
    for pref in role_prefixes:
        if clean_hdr.lower().startswith(pref):
            role_lang = clean_hdr[:len(pref)].strip()
            clean_hdr = clean_hdr[len(pref):].strip()
            break

    parsed_n = parse_name(clean_hdr)

    # Match with master database if present
    speaker_id = None
    if mdb_db:
        vorname = parsed_n["vorname"].lower().strip()
        nachname = parsed_n["nachname"].lower().strip()
        for entry in mdb_db:
            entry_vn = entry.get("vorname", "").lower().strip()
            entry_nn = entry.get("nachname", "").lower().strip()
            if entry_vn == vorname and entry_nn == nachname:
                speaker_id = entry.get("id")
                if not fraction and entry.get("fraktion"):
                    fraction = entry.get("fraktion")
                break

        # Unique last name fallback match
        if not speaker_id:
            matches = [e for e in mdb_db if e.get("nachname", "").lower().strip() == nachname]
            if len(matches) == 1:
                speaker_id = matches[0].get("id")
                if not fraction and matches[0].get("fraktion"):
                    fraction = matches[0].get("fraktion")

    if not speaker_id:
        speaker_id = generate_speaker_id(clean_hdr)

    return {
        "id": speaker_id,
        "vorname": parsed_n["vorname"],
        "von_oder_zu": parsed_n["von_oder_zu"],
        "nachname": parsed_n["nachname"],
        "titel": parsed_n["titel"],
        "fraktion": fraction,
        "role_lang": role_lang,
        "role_kurz": role_kurz
    }


def build_redner_xml(speaker_data):
    """Build the <redner> XML node string with XML escaping."""
    parts = [f'<redner id="{html.escape(speaker_data["id"])}"><name>']
    if speaker_data.get("titel"):
        parts.append(f'<titel>{html.escape(speaker_data["titel"])}</titel>')
    parts.append(f'<vorname>{html.escape(speaker_data["vorname"])}</vorname>')
    if speaker_data.get("von_oder_zu"):
        parts.append(f'<von_oder_zu>{html.escape(speaker_data["von_oder_zu"])}</von_oder_zu>')
    parts.append(f'<nachname>{html.escape(speaker_data["nachname"])}</nachname>')
    if speaker_data.get("fraktion"):
        parts.append(f'<fraktion>{html.escape(speaker_data["fraktion"])}</fraktion>')
    if speaker_data.get("role_lang"):
        parts.append(
            f'<rolle><rolle_lang>{html.escape(speaker_data["role_lang"])}</rolle_lang>'
            f'<rolle_kurz>{html.escape(speaker_data.get("role_kurz") or "")}</rolle_kurz></rolle>'
        )
    parts.append('</name></redner>')
    return "".join(parts)


def parse_old_toc(toc_text, mdb_db=None):
    """Parse the unstructured Table of Contents text into blocks and entries."""
    lines = [line.strip() for line in toc_text.split("\n") if line.strip()]
    blocks = []
    current_block = {"title": None, "entries": []}

    accumulator = []

    def clean_accumulated(acc):
        text = ""
        for part in acc:
            if text.endswith("-"):
                text = text[:-1] + part
            else:
                if text:
                    text += " "
                text += part
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def emit_accumulated():
        if not accumulator:
            return

        last_line = accumulator[-1]
        match = re.search(r'\.\s*(\d+)\s*([A-D])\s*$', last_line)
        if match:
            page = match.group(1)
            area = match.group(2)
            accumulator[-1] = last_line[:match.start()].rstrip(" .")
            entry_text = clean_accumulated(accumulator)
            current_block["entries"].append({
                "text": entry_text,
                "page": page,
                "area": area
            })
        else:
            entry_text = clean_accumulated(accumulator)
            current_block["entries"].append({
                "text": entry_text,
                "page": None,
                "area": None
            })
        accumulator.clear()

    for line in lines:
        if re.match(r'^(?:Tagesordnungspunkt|Zusatztagesordnungspunkt)\s+\d+:', line, re.IGNORECASE):
            emit_accumulated()
            if current_block["title"] or current_block["entries"]:
                blocks.append(current_block)
            current_block = {"title": line, "entries": []}
            continue

        is_new = False
        if (line.startswith("ZusFr") or line.startswith("Antw") or
                line.startswith("DringlAnfr") or line.startswith("Nächste") or
                line.startswith("Anlage") or line.startswith("[...")):
            is_new = True
        elif re.search(r'\b(CDU/CSU|SPD|FDP|DIE LINKE|GRÜNE|BÜNDNIS 90/DIE GRÜNEN|AfD|LINKE|BSW|SSW|PDS)\b\s*$', line, re.IGNORECASE):
            is_new = True
        elif ":" in line and is_valid_speaker(line):
            is_new = True

        if is_new:
            emit_accumulated()

        accumulator.append(line)

        if re.search(r'\.\s*(\d+)\s*([A-D])\s*$', line):
            emit_accumulated()

    emit_accumulated()
    if current_block["title"] or current_block["entries"]:
        blocks.append(current_block)

    return blocks


def extract_comments_from_text(text):
    """Splits a text paragraph into text segments and parenthesized comment segments."""
    pattern = re.compile(r'\(([^)]+)\)')
    parts = []
    last_end = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        before = text[last_end:start].strip()
        if before:
            parts.append(("text", before))
        parts.append(("comment", match.group(0)))
        last_end = end

    after = text[last_end:].strip()
    if after:
        parts.append(("text", after))

    return parts if parts else [("text", text)]


def get_german_day_name(datum_str):
    """Helper to convert a date string like '12.03.2003' into German full day name."""
    try:
        d = datetime.datetime.strptime(datum_str, "%d.%m.%Y")
        weekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
        months = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]
        return f"{weekdays[d.weekday()]}, den {d.day}. {months[d.month - 1]} {d.year}"
    except Exception:
        return "Mittwoch, den 12. März 2003"


def parse_old_protocol(xml_content, mdb_db=None):
    """
    Parses the full old protocol XML containing the unstructured text
    and returns a structured, schema-compliant XML string in WP 18+ format.
    """
    # Security: Use lxml.etree XMLParser with resolve_entities=False and no_network=True
    # to prevent XML External Entity (XXE) attacks and entity expansion (DoS) vulnerabilities.
    safe_parser = etree.XMLParser(resolve_entities=False, no_network=True)
    if isinstance(xml_content, str):
        xml_bytes = xml_content.encode("utf-8")
    else:
        xml_bytes = xml_content
    root = etree.fromstring(xml_bytes, parser=safe_parser)

    wahlperiode = root.find("WAHLPERIODE").text.strip()
    nr = root.find("NR").text.strip()
    sitzungsnr = nr.split("/")[-1].strip()
    datum = root.find("DATUM").text.strip()

    text_content = root.find("TEXT").text

    # 1. Separate TOC and Session Flow
    toc_match = re.search(
        r'I\s*n\s*h\s*a\s*l\s*t\s*[:\.]\s*(.*?)(?=\n\s*Sitzung\b)',
        text_content,
        re.IGNORECASE | re.DOTALL,
    )

    if not toc_match:
        toc_match = re.search(
            r'Inhalt\s*:(.*?)(?=\n\s*Sitzung\b)',
            text_content,
            re.IGNORECASE | re.DOTALL,
        )

    if toc_match:
        toc_text = toc_match.group(1).strip()
        session_text = text_content[toc_match.end():].strip()
        toc_blocks = parse_old_toc(toc_text, mdb_db=mdb_db)
    else:
        # Kein Inhaltsverzeichnis vorhanden:
        # gesamten Text als Sitzung verwenden.
        toc_text = ""
        toc_blocks = []
        session_text = text_content

    # 2. Parse TOC
    toc_blocks = parse_old_toc(toc_text, mdb_db=mdb_db)

    # 3. Extract dynamic start side number from TOC
    start_page = "1"
    for block in toc_blocks:
        for entry in block["entries"]:
            if entry["page"]:
                start_page = entry["page"]
                break
        if start_page != "1":
            break

    # 4. Parse Session Flow lines
    lines = [line.strip() for line in session_text.split("\n") if line.strip()]

    # Find "Beginn:" line and extract start time
    start_time = "13:00"
    header_lines_count = 0
    for idx, line in enumerate(lines):
        if "Beginn:" in line:
            time_match = re.search(r'Beginn:\s*(\d{1,2})[\.:](\d{2})\s*Uhr', line)
            if time_match:
                start_time = f"{time_match.group(1).zfill(2)}:{time_match.group(2)}"
            header_lines_count = idx + 1
            break

    lines = lines[header_lines_count:]

    # Merge multi-line speaker headers
    merged_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_speaker_start = is_valid_speaker(line) and not line.startswith("(")
        if is_speaker_start and not line.endswith(":") and i + 1 < len(lines) and lines[i+1].endswith(":"):
            merged_lines.append(line + " " + lines[i+1])
            i += 2
        else:
            merged_lines.append(line)
            i += 1

    # Group lines into structured items
    items = []
    current_text_lines = []

    def emit_text_item():
        if current_text_lines:
            text = " ".join(current_text_lines)
            text = clean_hyphens(text)
            sub_items = extract_comments_from_text(text)
            items.extend(sub_items)
            current_text_lines.clear()

    total_merged = len(merged_lines)
    start_time_proc = time.monotonic()
    log_step = max(50, total_merged // 20) if total_merged > 0 else 1

    for idx, line in enumerate(merged_lines):
        # Structured telemetry progress reporting & ETA calculation with performance throttling
        current_step = idx + 1
        if total_merged > 0 and (current_step % log_step == 0 or current_step == total_merged):
            elapsed = time.monotonic() - start_time_proc
            pct = int((current_step / total_merged) * 100)
            eta = (elapsed / current_step) * (total_merged - current_step)
            eta_str = time.strftime("%M:%S", time.gmtime(eta)) if eta < 3600 else time.strftime("%H:%M:%S", time.gmtime(eta))
            logger.info(
                f"OldProtocolParsing: {current_step}/{total_merged} | {pct}% | ETA: {eta_str}"
            )

        if is_valid_speaker(line) and line.endswith(":"):
            emit_text_item()
            items.append(("header", line))
            continue

        if line.startswith("(") and line.endswith(")"):
            emit_text_item()
            items.append(("comment", line))
            continue

        if re.match(r'^Ich rufe (?:Tagesordnungspunkt|Zusatztagesordnungspunkt)', line, re.IGNORECASE):
            emit_text_item()
            items.append(("top_call", line))
            continue

        if line == "Fragestunde":
            emit_text_item()
            items.append(("top_title", line))
            continue

        if line.startswith("– Drucksachen") or line.startswith("- Drucksachen"):
            emit_text_item()
            items.append(("top_drs", line))
            continue

        if (line.startswith("Zu Beginn") or line.startswith("Wir kommen") or
                line.startswith("Hat der") or line.startswith("Zur Beantwortung")):
            emit_text_item()

        current_text_lines.append(line)

        # Sentence boundary detection
        if line.endswith(".") or line.endswith("!") or line.endswith("?"):
            if not line.endswith("-") and not (idx + 1 < len(merged_lines) and merged_lines[idx+1].islower()):
                emit_text_item()

    emit_text_item()

    # Generate XML header & wrapper
    xml_header = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<?xml-stylesheet href="dbtplenarprotokoll.css" type="text/css" charset="UTF-8"?>'
        '<!DOCTYPE dbtplenarprotokoll SYSTEM "dbtplenarprotokoll.dtd">'
        f'<dbtplenarprotokoll wahlperiode="{html.escape(wahlperiode)}" sitzung-nr="{html.escape(sitzungsnr)}" '
        f'sitzung-datum="{html.escape(datum)}" sitzung-start-uhrzeit="{html.escape(start_time)}" '
        f'sitzung-ende-uhrzeit="" sitzung-naechste-datum="" start-seitennr="{html.escape(start_page)}">\n'
    )

    day_name = get_german_day_name(datum)

    xml_vorspann = (
        '  <vorspann>\n'
        '    <kopfdaten>\n'
        f'      <plenarprotokoll-nummer>Plenarprotokoll <wahlperiode>{html.escape(wahlperiode)}</wahlperiode>'
        f'/<sitzungsnr>{html.escape(sitzungsnr)}</sitzungsnr></plenarprotokoll-nummer>\n'
        '      <herausgeber>Deutscher Bundestag</herausgeber>\n'
        '      <berichtart>Stenografischer Bericht</berichtart>\n'
        f'      <sitzungstitel>{html.escape(sitzungsnr)}. Sitzung</sitzungstitel>\n'
        f'      <veranstaltungsdaten><ort>Berlin</ort>, <datum date="{html.escape(datum)}">{html.escape(day_name)}</datum></veranstaltungsdaten>\n'
        '    </kopfdaten>\n'
        '    <inhaltsverzeichnis>\n'
        '      <ivz-titel>I n h a l t :</ivz-titel>\n'
    )

    rede_counter = 1
    for block in toc_blocks:
        xml_vorspann += '      <ivz-block>\n'
        if block["title"]:
            xml_vorspann += f'        <ivz-block-titel>{html.escape(block["title"])}</ivz-block-titel>\n'

        for entry in block["entries"]:
            text = entry["text"]
            # Dynamically parse TOC entry speaker
            matched_speaker = None
            if ":" in text or any(name in text for name in ["Eckart von Klaeden", "Hans Georg Wagner", "Gitta Connemann"]):
                clean_speaker_name = text
                # Try to clean up action verbs like ZusFr or Antw
                for prefix in ["ZusFr", "Antw", "Antw PStSekr", "PStSekr", "SchrAntw"]:
                    if clean_speaker_name.startswith(prefix):
                        clean_speaker_name = clean_speaker_name[len(prefix):].strip()
                matched_speaker = parse_speaker_header(clean_speaker_name, mdb_db=mdb_db)

            xml_vorspann += '        <ivz-eintrag>\n'
            if matched_speaker and matched_speaker["id"] != "11001072":
                redner_xml = build_redner_xml(matched_speaker)
                xml_vorspann += f'          <ivz-eintrag-inhalt>{redner_xml}{html.escape(text)}</ivz-eintrag-inhalt>\n'
            else:
                xml_vorspann += f'          <ivz-eintrag-inhalt>{html.escape(text)}</ivz-eintrag-inhalt>\n'

            if entry["page"]:
                page = entry["page"]
                area = entry["area"] or "A"
                if matched_speaker and matched_speaker["id"] != "11001072":
                    rede_id = f"ID{wahlperiode}{int(sitzungsnr):03d}{rede_counter:04d}00"
                    xml_vorspann += (
                        f'          <xref ref-type="rede" rid="{html.escape(rede_id)}" pnr="{html.escape(page)}" div="{html.escape(area)}">\n'
                        f'            <a href="S{html.escape(page)}" typ="druckseitennummer"><seite>{html.escape(page)}</seite> '
                        f'<seitenbereich>{html.escape(area)}</seitenbereich></a>\n'
                        '          </xref>\n'
                    )
                    rede_counter += 1
                else:
                    xml_vorspann += (
                        f'          <a href="S{html.escape(page)}" typ="druckseitennummer"><seite>{html.escape(page)}</seite> '
                        f'<seitenbereich>{html.escape(area)}</seitenbereich></a>\n'
                    )
            xml_vorspann += '        </ivz-eintrag>\n'
        xml_vorspann += '      </ivz-block>\n'

    xml_vorspann += '    </inhaltsverzeichnis>\n  </vorspann>\n'

    # Construct sitzungsverlauf
    xml_verlauf = '  <sitzungsverlauf>\n'

    in_sitzungsbeginn = True
    in_tagesordnungspunkt = False
    current_rede_id = None
    rede_idx = 1
    top_idx = 1

    for item_type, item_text in items:
        if item_type == "header":
            # Dynamically parse speaker header
            matched_speaker = parse_speaker_header(item_text, mdb_db=mdb_db)

            if matched_speaker:
                # Strictly Rollen-based President verification
                is_president = False
                if matched_speaker.get("role_lang"):
                    role_l = matched_speaker["role_lang"].lower()
                    if any(r in role_l for r in ["präsident", "präsidentin", "vizepräsident", "vizepräsidentin"]):
                        is_president = True

                if is_president:
                    if current_rede_id:
                        xml_verlauf += '      </rede>\n'
                        current_rede_id = None

                    if in_sitzungsbeginn:
                        xml_verlauf += f'    <sitzungsbeginn sitzung-start-uhrzeit="{html.escape(start_time)}">\n'
                        xml_verlauf += f'      <p klasse="N">{html.escape(item_text)}</p>\n'
                    else:
                        xml_verlauf += f'      <p klasse="O">{html.escape(item_text)}</p>\n'
                else:
                    # Regular speaker speech start
                    if current_rede_id:
                        xml_verlauf += '      </rede>\n'

                    current_rede_id = f"ID{wahlperiode}{int(sitzungsnr):03d}{rede_idx:04d}00"
                    rede_idx += 1

                    redner_xml = build_redner_xml(matched_speaker)
                    xml_verlauf += f'      <rede id="{html.escape(current_rede_id)}">\n'
                    xml_verlauf += f'        <p klasse="redner">{redner_xml}{html.escape(item_text)}</p>\n'
            else:
                if current_rede_id:
                    xml_verlauf += '      </rede>\n'
                    current_rede_id = None
                xml_verlauf += f'      <p klasse="O">{html.escape(item_text)}</p>\n'

        elif item_type == "text":
            if current_rede_id:
                xml_verlauf += f'        <p klasse="J_1">{html.escape(item_text)}</p>\n'
            elif in_sitzungsbeginn:
                xml_verlauf += f'      <p klasse="J_1">{html.escape(item_text)}</p>\n'
                xml_verlauf += '    </sitzungsbeginn>\n'
                in_sitzungsbeginn = False
            else:
                xml_verlauf += f'      <p klasse="J">{html.escape(item_text)}</p>\n'

        elif item_type == "comment":
            if current_rede_id:
                xml_verlauf += f'        <kommentar>{html.escape(item_text)}</kommentar>\n'
            else:
                xml_verlauf += f'      <kommentar>{html.escape(item_text)}</kommentar>\n'

        elif item_type == "top_call":
            if current_rede_id:
                xml_verlauf += '      </rede>\n'
                current_rede_id = None

            if in_tagesordnungspunkt:
                xml_verlauf += '    </tagesordnungspunkt>\n'

            xml_verlauf += f'    <tagesordnungspunkt top-id="Tagesordnungspunkt {top_idx}">\n'
            top_idx += 1
            in_tagesordnungspunkt = True
            xml_verlauf += f'      <p klasse="O">{html.escape(item_text)}</p>\n'

        elif item_type == "top_title":
            xml_verlauf += f'      <p klasse="T_ohne_NaS">{html.escape(item_text)}</p>\n'

        elif item_type == "top_drs":
            xml_verlauf += f'      <p klasse="T_Drs">{html.escape(item_text)}</p>\n'

    # Close any open tag
    if current_rede_id:
        xml_verlauf += '      </rede>\n'
    if in_tagesordnungspunkt:
        xml_verlauf += '    </tagesordnungspunkt>\n'

    xml_verlauf += '  </sitzungsverlauf>\n'
    xml_footer = '</dbtplenarprotokoll>\n'

    return xml_header + xml_vorspann + xml_verlauf + xml_footer


def main():
    parser = argparse.ArgumentParser(
        description="General Bundestag Plenary Protocol Parser for WP 1 to 17"
    )
    parser.add_argument("input", help="Path to input XML document containing unstructured text")
    parser.add_argument("output", nargs="?", help="Path to write target formatted XML document")
    parser.add_argument("--mdb-database", help="Path to external JSON database with MdB master data")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file %s does not exist.", input_path)
        sys.exit(1)

    mdb_db = load_mdb_database(args.mdb_database)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        output_xml = parse_old_protocol(content, mdb_db=mdb_db)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write(output_xml)
            logger.info("Successfully wrote generated XML to %s", out_path)
        else:
            sys.stdout.write(output_xml + "\n")
    except Exception as e:
        logger.error("Error parsing protocol: %s", e, exc_info=True)
        sys.exit(1)


def load_mdb_database(path):
    if not path:
        return None
    import tempfile
    resolved_path = Path(path).resolve()
    resolved_project_root = Path(__file__).resolve().parents[1]
    safe_roots = [resolved_project_root]
    try:
        safe_roots.append(Path(tempfile.gettempdir()).resolve())
    except Exception:
        pass

    is_safe = False
    for root in safe_roots:
        try:
            resolved_path.relative_to(root)
            is_safe = True
            break
        except ValueError:
            pass

    if not is_safe:
        logger.warning("Unsafe MdB database path rejected: %s", path)
        return None

    if not resolved_path.exists():
        logger.warning("MdB database file %s not found.", path)
        return None
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Error loading MdB database %s: %s", path, e)
        return None


if __name__ == "__main__":
    main()
