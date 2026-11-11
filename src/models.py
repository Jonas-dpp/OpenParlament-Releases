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
SQLAlchemy ORM models for OpenParlament.

Domain objects
──────────────
Sitzung      – A plenary session (Plenarsitzung) of the Bundestag.
Redner       – A speaker (Abgeordneter / Minister).
Rede         – A single speech held during a Sitzung.
Zwischenruf  – A verbal or non-verbal interjection during a Rede.

Relationships
─────────────
Sitzung  1 ── n  Rede
Redner   1 ── n  Rede
Rede     1 ── n  Zwischenruf
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ─────────────────────────────────────────────────────────────────────────────
# Sitzung
# ─────────────────────────────────────────────────────────────────────────────

class Sitzung(Base):
    """Represents a single plenary session of the Bundestag."""

    __tablename__ = "sitzungen"

    sitzungs_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wahlperiode: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    sitzungsnr: Mapped[int] = mapped_column(Integer, nullable=False)
    datum: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
    # German weekday name derived directly from datum during XML parsing: "Montag" … "Sonntag".
    wochentag: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    titel: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    gesamtwortzahl: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    reden: Mapped[List["Rede"]] = relationship(
        "Rede", back_populates="sitzung", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("wahlperiode", "sitzungsnr", name="uq_sitzung_wp_nr"),
    )

    def __repr__(self) -> str:
        return (
            f"<Sitzung wahlperiode={self.wahlperiode} "
            f"nr={self.sitzungsnr} datum={self.datum} wochentag={self.wochentag}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Redner
# ─────────────────────────────────────────────────────────────────────────────

class Redner(Base):
    """Represents a speaker (MP, minister, etc.)."""

    __tablename__ = "redner"

    redner_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bundestag_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )
    vorname: Mapped[str] = mapped_column(String(128), nullable=False)
    nachname: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    titel: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    fraktion: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    partei: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    reden: Mapped[List["Rede"]] = relationship(
        "Rede", back_populates="redner", cascade="all, delete-orphan"
    )

    @property
    def vollname(self) -> str:
        """Return the full display name including optional title."""
        parts = []
        if self.titel:
            parts.append(self.titel)
        parts.append(self.vorname)
        parts.append(self.nachname)
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"<Redner {self.vollname} ({self.fraktion})>"


# ─────────────────────────────────────────────────────────────────────────────
# Rede
# ─────────────────────────────────────────────────────────────────────────────

class Rede(Base):
    """Represents a single speech during a plenary session."""

    __tablename__ = "reden"

    rede_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bundestag_rede_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )
    sitzung_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sitzungen.sitzungs_id"), nullable=False, index=True
    )
    redner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("redner.redner_id"), nullable=False, index=True
    )
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tagesordnungspunkt: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    wortanzahl: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # --- Mehrschichtige Text- und Versionierungsfelder ---
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    clean_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    normalized_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processing_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # --- Qualitätsmetriken ---
    char_count_raw: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_count_clean: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_count_metadata: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    removed_chars: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    removed_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    count_applause: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    count_heckling: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    count_laughter: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    count_president: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # --- NLP Felder (directly initialized in ORM schema) ---
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    ton_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Dedicated tone probabilities (directly declared in ORM schema).
    tone_aggression: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tone_sarkasmus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tone_humor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tone_neutral: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Legacy JSON payload, kept temporarily for compatibility during rollout.
    tone_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    adressaten: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    sitzung: Mapped["Sitzung"] = relationship("Sitzung", back_populates="reden")
    redner: Mapped["Redner"] = relationship("Redner", back_populates="reden")
    zwischenrufe: Mapped[List["Zwischenruf"]] = relationship(
        "Zwischenruf", back_populates="rede", cascade="all, delete-orphan"
    )
    stylometrie: Mapped[Optional["StylometrieFeatures"]] = relationship(
        "StylometrieFeatures", back_populates="rede", cascade="all, delete-orphan", uselist=False
    )

    def __repr__(self) -> str:
        return (
            f"<Rede id={self.rede_id} redner_id={self.redner_id} "
            f"wörter={self.wortanzahl}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Zwischenruf
# ─────────────────────────────────────────────────────────────────────────────

class Zwischenruf(Base):
    """Represents a verbal or non-verbal interjection during a speech."""

    __tablename__ = "zwischenrufe"

    ruf_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rede_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("reden.rede_id"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Faction that produced the interjection (extracted from the raw text).
    fraktion: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    # Sentiment score assigned by the NLP engine: -1.0 (very negative) to +1.0 (positive).
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    # Human-readable category: "Beifall", "Lachen", "Widerspruch", "Zwischenruf", "Unruhe", "Zustimmung", "Sonstiges".
    kategorie: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # Rhetorical tone label assigned by ToneClassifier: "Aggression", "Sarkasmus", "Humor", "Neutral".
    ton_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # Dedicated tone probabilities (kept nullable for backward-compatible migration).
    tone_aggression: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tone_sarkasmus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tone_humor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tone_neutral: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Detailed probabilities for each tone.
    tone_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Comma-separated list of detected addressees (factions / persons) from AddresseeDetector.
    adressaten: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    rede: Mapped["Rede"] = relationship("Rede", back_populates="zwischenrufe")

    def __repr__(self) -> str:
        return (
            f"<Zwischenruf id={self.ruf_id} fraktion={self.fraktion} "
            f"score={self.sentiment_score}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# StylometrieFeatures
# ─────────────────────────────────────────────────────────────────────────────

class StylometrieFeatures(Base):
    """Represents the stylometric features of a speech."""

    __tablename__ = "stylometrie_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rede_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("reden.rede_id"), nullable=False, unique=True, index=True
    )
    
    avg_sentence_length: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    passive_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    type_token_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fill_word_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    punctuation_freq: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    rede: Mapped["Rede"] = relationship("Rede", back_populates="stylometrie")

    def __repr__(self) -> str:
        return f"<StylometrieFeatures rede_id={self.rede_id}>"


# ─────────────────────────────────────────────────────────────────────────────
# FramingResult (Precomputed)
# ─────────────────────────────────────────────────────────────────────────────

class FramingResult(Base):
    """Precomputed framing terms per faction, segment, variant, and wahlperiode.

    These scores are generated offline by scripts/precompute_framing.py
    using TF-IDF to identify characteristic vocabulary for each faction, communication segment, and variant.
    """

    __tablename__ = "framing_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wahlperiode: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    fraktion: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    segment: Mapped[str] = mapped_column(String(32), nullable=False, default="all", server_default="all", index=True)
    variant: Mapped[str] = mapped_column(String(32), nullable=False, default="global", server_default="global", index=True)
    term: Mapped[str] = mapped_column(String(256), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("wahlperiode", "fraktion", "segment", "variant", "term", name="uq_framing_wp_frak_seg_var_term"),
    )

    def __repr__(self) -> str:
        return f"<FramingResult wp={self.wahlperiode} fraktion={self.fraktion} seg={self.segment} var={self.variant} term={self.term} score={self.score:.4f}>"


# ─────────────────────────────────────────────────────────────────────────────
# LexicalResult (Precomputed)
# ─────────────────────────────────────────────────────────────────────────────

class LexicalResult(Base):
    """Precomputed lexical metrics per faction and wahlperiode.

    Stores measures of lexical diversity (e.g., TTR) and density (content word ratio).
    Generated offline by scripts/precompute_framing.py.
    """

    __tablename__ = "lexical_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wahlperiode: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    fraktion: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Lexical Diversity: Type-Token Ratio (unique words / total words)
    ttr: Mapped[float] = mapped_column(Float, nullable=False)
    # Mean Segmental TTR (more stable across document lengths)
    msttr: Mapped[float] = mapped_column(Float, nullable=True)
    # Lexical Density: (Nouns + Verbs + Adj + Adv) / Total words
    lexical_density: Mapped[float] = mapped_column(Float, nullable=False)

    # Metadata for weighting / diagnostics
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_types: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("wahlperiode", "fraktion", name="uq_lexical_wp_frak"),
    )

    def __repr__(self) -> str:
        return f"<LexicalResult wp={self.wahlperiode} fraktion={self.fraktion} ttr={self.ttr:.4f} density={self.lexical_density:.4f}>"


# ─────────────────────────────────────────────────────────────────────────────
# SpeechComplexityResult (New/Linguistic)
# ─────────────────────────────────────────────────────────────────────────────

class SpeechComplexityResult(Base):
    """Linguistische Komplexität und Reaktionsmetriken einer Rede."""

    __tablename__ = "speech_complexity_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rede_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("reden.rede_id"), nullable=False, index=True
    )
    wahlperiode: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    redner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("redner.redner_id"), nullable=False, index=True
    )
    wstf_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flesch_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_sentence_length: Mapped[float] = mapped_column(Float, nullable=False)
    subordinate_clause_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    avg_word_length: Mapped[float] = mapped_column(Float, nullable=False)
    long_word_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    lexical_diversity: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, default=func.now())

    rede: Mapped["Rede"] = relationship("Rede")
    redner: Mapped["Redner"] = relationship("Redner")

    def __repr__(self) -> str:
        return f"<SpeechComplexityResult rede_id={self.rede_id} wstf={self.wstf_score:.2f} flesch={self.flesch_score:.2f}>"


# ─────────────────────────────────────────────────────────────────────────────
# ReactionGapResult (New/Action-Reaction-Gap)
# ─────────────────────────────────────────────────────────────────────────────

class ReactionGapResult(Base):
    """Unterschied im Sentiment zwischen Rede und nächstem Zwischenruf (Eskalation)."""

    __tablename__ = "reaction_gap_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rede_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("reden.rede_id"), nullable=False, index=True
    )
    zwischenruf_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("zwischenrufe.ruf_id"), nullable=False, index=True
    )
    topic_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    speaker_sentiment: Mapped[float] = mapped_column(Float, nullable=False)
    reaction_sentiment: Mapped[float] = mapped_column(Float, nullable=False)
    gap: Mapped[float] = mapped_column(Float, nullable=False)
    time_to_first_interjection: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, default=func.now())

    rede: Mapped["Rede"] = relationship("Rede")
    zwischenruf: Mapped["Zwischenruf"] = relationship("Zwischenruf")

    def __repr__(self) -> str:
        return f"<ReactionGapResult rede_id={self.rede_id} gap={self.gap:.2f}>"
