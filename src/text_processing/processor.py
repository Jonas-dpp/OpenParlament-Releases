import re
import unicodedata
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Module-level precompiled regex patterns to avoid compilation overhead in hot paths
_COMMENT_PATTERN = re.compile(r"\(([^)]+)\)", re.UNICODE)

_PRESIDENT_PATTERN = re.compile(
    r"^(?:Präsident(?:in)?|Vizepräsident(?:in)?|Alterspräsident(?:in)?)\b.*?:",
    re.IGNORECASE | re.UNICODE
)

_STAGE_DIRECTION_PATTERN = re.compile(
    r"^(?:Sitzung unterbrochen|Sitzung geschlossen|Nächster Redner|Das Wort hat)\b.*",
    re.IGNORECASE | re.UNICODE
)

_MULTIPLE_SPACES_PATTERN = re.compile(r"\s+", re.UNICODE)
_SPACES_TABS_PATTERN = re.compile(r"[ \t]+", re.UNICODE)
_CR_LF_PATTERN = re.compile(r"\r\n")
_REDUNDANT_NEWLINES_PATTERN = re.compile(r"\n\s*\n+")

@dataclass
class ProcessingMetrics:
    char_count_raw: int = 0
    char_count_clean: int = 0
    char_count_metadata: int = 0
    removed_chars: int = 0
    removed_percent: float = 0.0
    count_applause: int = 0
    count_heckling: int = 0
    count_laughter: int = 0
    count_president: int = 0
    processing_time_ms: int = 0

@dataclass
class ProcessedText:
    raw_text: str
    clean_text: str
    normalized_text: str
    metadata_text: str
    metrics: ProcessingMetrics

class TextProcessor:
    """
    Zentrale Klasse für die mehrschichtige Textverarbeitung von Bundestagsreden.

    Verantwortlichkeiten:
    - Aufteilung von Rohtext in Clean-Text und Metadaten-Text (Schnittstellen, Applaus, etc.)
    - Unicode-Normalisierung und Text-Bereinigung für ML
    - Berechnung detaillierter Qualitätsmetriken
    """

    def __init__(self, version: int = 1):
        self.version = version

        # Keep attributes for backward compatibility
        self.comment_pattern = _COMMENT_PATTERN
        self.president_pattern = _PRESIDENT_PATTERN
        self.stage_direction_pattern = _STAGE_DIRECTION_PATTERN

    def split_raw_text(self, raw_text: str) -> Tuple[str, str, Dict[str, int]]:
        """
        Trennt Metadaten, Zurufe, Applaus und Regieanweisungen vom Sprachanteil.
        Behält klassische Redeelemente wie "Herr Präsident!", "Meine Damen und Herren!" etc. bei.

        Returns:
            Tuple of: (clean_text, metadata_text, counts_dict)
        """
        if raw_text and len(raw_text) > 500000:
            raise ValueError("Input text exceeds the maximum safe limit of 500,000 characters.")

        if not raw_text:
            return "", "", {"applause": 0, "heckling": 0, "laughter": 0, "president": 0}

        clean_lines: List[str] = []
        metadata_parts: List[str] = []

        counts = {
            "applause": 0,
            "heckling": 0,
            "laughter": 0,
            "president": 0
        }

        # We split the text by lines to parse and maintain paragraph boundaries
        lines = raw_text.split("\n")

        for line in lines:
            line_str = line.strip()
            if not line_str:
                clean_lines.append("")
                continue

            # Check if line is a president/stage direction that should be separated
            # Optimization: Guard regex match calls with lightweight, fast-path string checks.
            # Avoid executing the expensive _PRESIDENT_PATTERN regex match unless ":" is in the line.
            if ":" in line_str and _PRESIDENT_PATTERN.match(line_str):
                counts["president"] += 1
                metadata_parts.append(line_str)
                continue

            if _STAGE_DIRECTION_PATTERN.match(line_str):
                metadata_parts.append(line_str)
                continue

            # Parse comments/interjections within parentheses
            line_clean = line_str
            # Avoid executing finditer unless parenthesis actually exists in the line.
            matches = list(_COMMENT_PATTERN.finditer(line_str)) if "(" in line_str else []

            if matches:
                # Extract comment strings
                for m in matches:
                    comment_content = m.group(1).strip()
                    metadata_parts.append(f"({comment_content})")

                    # Classify comment type
                    comment_lower = comment_content.lower()
                    has_category = False
                    if "beifall" in comment_lower or "bravo" in comment_lower or "zustimmung" in comment_lower:
                        counts["applause"] += 1
                        has_category = True
                    if "zuruf" in comment_lower or "ruft" in comment_lower or "widerspruch" in comment_lower or "gegenruf" in comment_lower:
                        counts["heckling"] += 1
                        has_category = True
                    if "lachen" in comment_lower or "heiterkeit" in comment_lower:
                        counts["laughter"] += 1
                        has_category = True

                    if not has_category:
                        # Fallback interjection
                        counts["heckling"] += 1

                # Replace comments with whitespace
                line_clean = _COMMENT_PATTERN.sub("", line_str).strip()

            # Clean up double spaces if any comments were inside or if line contains multiple spaces/tabs
            if matches or "  " in line_clean or "\t" in line_clean or "\xa0" in line_clean:
                line_clean = _MULTIPLE_SPACES_PATTERN.sub(" ", line_clean).strip()

            if line_clean:
                clean_lines.append(line_clean)

        # Re-assemble texts
        clean_text = "\n".join(clean_lines).strip()
        # Clean multiple newlines in clean_text
        # Optimization: Guard the redundant newline regex substitution with a fast-path string check.
        if "\n\n" in clean_text:
            clean_text = _REDUNDANT_NEWLINES_PATTERN.sub("\n\n", clean_text)

        metadata_text = "\n".join(metadata_parts).strip()

        return clean_text, metadata_text, counts

    def normalize_text(self, clean_text: str) -> str:
        """
        Normalisiert Text für Machine Learning / NLP.
        - Unicode-Normalisierung (NFKC)
        - Bereinigen überflüssiger Leerzeichen
        - Zeilenumbrüche vereinheitlichen
        """
        if not clean_text:
            return ""

        # Unicode Normalization (e.g. handle umlauts or special hyphen representations)
        normalized = unicodedata.normalize("NFKC", clean_text)

        # Replace multiple spacing / tabs
        # Optimization: Guard the multiple spaces/tabs regex substitution with a fast-path string check.
        if "  " in normalized or "\t" in normalized:
            normalized = _SPACES_TABS_PATTERN.sub(" ", normalized)

        # Clean redundant newlines
        # Optimization: Only substitute carriage returns and redundant newlines if they are present.
        if "\r\n" in normalized:
            normalized = _CR_LF_PATTERN.sub("\n", normalized)
        if "\n\n" in normalized:
            normalized = _REDUNDANT_NEWLINES_PATTERN.sub("\n\n", normalized)

        return normalized.strip()

    def quality_report(self, raw_text: str, clean_text: str, metadata_text: str, counts: Dict[str, int], duration_ms: int) -> ProcessingMetrics:
        """
        Berechnet Qualitätsmetriken für den Verarbeitungsdurchlauf.
        """
        char_count_raw = len(raw_text) if raw_text else 0
        char_count_clean = len(clean_text) if clean_text else 0
        char_count_metadata = len(metadata_text) if metadata_text else 0

        removed_chars = max(0, char_count_raw - char_count_clean)
        removed_percent = (removed_chars / char_count_raw * 100.0) if char_count_raw > 0 else 0.0

        return ProcessingMetrics(
            char_count_raw=char_count_raw,
            char_count_clean=char_count_clean,
            char_count_metadata=char_count_metadata,
            removed_chars=removed_chars,
            removed_percent=removed_percent,
            count_applause=counts.get("applause", 0),
            count_heckling=counts.get("heckling", 0),
            count_laughter=counts.get("laughter", 0),
            count_president=counts.get("president", 0),
            processing_time_ms=duration_ms
        )

    def process(self, raw_text: str) -> ProcessedText:
        """
        Orchestriert den Workflow und gibt das Ergebnisobjekt zurück.
        """
        if raw_text and len(raw_text) > 500000:
            raise ValueError("Input text exceeds the maximum safe limit of 500,000 characters.")

        start_time = time.perf_counter()

        clean_text, metadata_text, counts = self.split_raw_text(raw_text)
        normalized_text = self.normalize_text(clean_text)

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        metrics = self.quality_report(raw_text, clean_text, metadata_text, counts, duration_ms)

        return ProcessedText(
            raw_text=raw_text or "",
            clean_text=clean_text,
            normalized_text=normalized_text,
            metadata_text=metadata_text,
            metrics=metrics
        )
