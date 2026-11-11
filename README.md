# OpenParlament: Der Bundestag-Scanner

**Demokratie-Mining — die digitale Röntgenaufnahme des Bundestags.**

[![Tests](https://github.com/Jonas-dpp/OpenParlament/actions/workflows/ci.yml/badge.svg)](https://github.com/Jonas-dpp/OpenParlament/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-3.2.0-brightgreen.svg)](pyproject.toml)
[![Schülerprojekt](https://img.shields.io/badge/🎓-Schülerprojekt-blueviolet.svg)](README.md)

Willkommen bei **OpenParlament** – einem **Schülerprojekt** zur NLP-gestützten Analyse der Plenarprotokolle des Deutschen Bundestags.
Das Projekt macht verborgene politische Dynamiken über ein interaktives Streamlit-Dashboard sichtbar und wird
offen auf GitHub entwickelt.

> **Version 3.2.0** — Neu: *Wissenschaftliche Hypothesen-Prüfung (H1–H3)*, *Framing-Analyse*, *Lexik-Analyse*,
> *Sprachkomplexität (Wiener Sachtextformel & Flesch FRE)*, *Eskalationsanalyse (Action-Reaction-Gap)*,
> *Polarisierungsindex*, *Stylometrie (Blind-Test)* sowie automatisierte *DB-Integritätsprüfung* und *Plot-Exporter*.

---

## 🧭 Dashboard-Seiten (25 Seiten in 6 Kategorien)

Das Dashboard gliedert sich in sechs logische Themenbereiche mit insgesamt 25 interaktiven Analysen und Werkzeugen:

### 1. 🔬 Start & Hypothesen
| Seite | Frage / Beschreibung | Version |
|---|---|---|
| 1.1 🏠 **Startseite** | Landing Page mit Live-Statistiken, Quick-Start-Guide & SPDX-2.3 SBOM | v1.0.0 / v2.5.0 |
| 1.2 🔬 **Hypothesen-Prüfung** | Live quantitative Tests für H1 (Opposition), H2 (Polarisierung) & H3 (Asymmetrische Angriffe) | **v3.2.0** |

### 2. 🏛️ Themen & Struktur
| Seite | Frage / Beschreibung | Version |
|---|---|---|
| 2.1 🏛️ **Tagesordnungspunkte** | Welche Agenda-Items provozieren die meisten negativen Reaktionen? | v2.2.0 |
| 2.2 📈 **Themen-Trend** | Wann dominieren welche Schlagworte? (inkl. Multi-WP & Reizwort-Index) | v1.0.0 / v2.3.0 |
| 2.3 🎯 **Framing-Analyse** | Welche TF-IDF Keywords kennzeichnen die Sprache der Fraktionen? | **v3.2.0** |
| 2.4 🕸️ **Interaktions-Netzwerk** | Welche Fraktionen unterbrechen sich wie oft? (inkl. NetworkX/Gephi-Export & H3-Asymmetrie) | v1.0.0 / **v3.2.0** |

### 3. 💬 Sprache & Stil
| Seite | Frage / Beschreibung | Version |
|---|---|---|
| 3.1 🎭 **Ton-Analyse** | Wie ist der rhetorische Ton: Aggression, Sarkasmus, Humor oder Neutral? | v1.2.0 |
| 3.2 📚 **Lexik-Analyse** | Wie reichhaltig ist der Wortschatz der Fraktionen? (TTR, MSTTR, lexikalische Dichte) | **v3.2.0** |
| 3.3 📏 **Sprachkomplexität** | Vergleichende Lesbarkeit (Wiener Sachtextformel & Flesch FRE) & Methodische Triangulation | **v3.2.0** |
| 3.4 🧬 **Stylometrie (Blind-Test)** | Erkennt ein ML-Modell die Fraktion anonymisierter Texte anhand syntaktischer Stilmerkmale? | **v3.1.0** |

### 4. ⚡ Stimmung & Konflikt
| Seite | Frage / Beschreibung | Version |
|---|---|---|
| 4.1 🌡️ **Debattenklima-Index** | Wie heiß war das Parlament pro Sitzung? (Composite Temperatur-Index) | v2.4.0 |
| 4.2 🔥 **Aggressions-Radar** | Wer kassiert / verteilt die meisten negativen Zwischenrufe? | v1.0.0 |
| 4.3 ⚖️ **Polarisierungsindex** | Wie stark sind feindselige Außeninteraktionen zwischen den politischen Lagern? | **v3.2.0** |
| 4.4 ⚡ **Eskalationsanalyse** | Misst das Sentiment-Delta (Action-Reaction-Gap) und Reaktionsgeschwindigkeiten | **v3.2.0** |
| 4.5 🎯 **Adressaten-Analyse** | An wen richtet sich ein Zwischenruf oder eine Rede? | v1.2.0 |
| 4.6 👏 **Reaktions-Analyse** | Wer produziert Beifall, wer Widerspruch? Civility-Index pro Fraktionspaar. | v2.2.0 |

### 5. 👥 Akteure & Parteien
| Seite | Frage / Beschreibung | Version |
|---|---|---|
| 5.1 🎤 **Redner-Profil** | Was ist das rhetorische DNA-Profil jedes Abgeordneten? (Radar-Charts) | v2.4.0 |
| 5.2 👥 **Redner-Vergleich** | Vergleiche zwei Abgeordnete direkt: Ton, Aggression, Redeaktivität & Kosinus-Ähnlichkeit | **v3.0.0** |
| 5.3 📡 **Fraktions-Dynamik** | Sunburst-Hierarchie & Zeitreihen: Wie entwickelt sich der Ton der Fraktionen? | **v3.0.0** |
| 5.4 ⏱️ **Redezeit-Gerechtigkeit** | Wird Redezeit proportional zur Fraktionsgröße verteilt? (Fairness-Index) | v2.2.0 |
| 5.5 ⚖️ **Wahlperioden-Vergleich** | Wie unterscheiden sich verschiedene Legislaturperioden in Ton & Aktivität? | v2.0.0 |

### 6. 🛠️ Daten, Technik & Validierung
| Seite | Frage / Beschreibung | Version |
|---|---|---|
| 6.1 📊 **Scraping-Monitor** | Wie ist der Datenbestand und NLP-Abdeckungsgrad? | v1.2.0 |
| 6.2 🗄️ **DB-Übersicht** | Datenbankschema, Zeilenzähler, Sankey-Datenfluss & ER-Diagramm | v2.5.0 |
| 6.3 ✅ **Modell-Validierung** | Validierung des Sentiment-Modells gegen einen manuell annotierten Gold Standard (N=100) | **v3.2.0** |
| 6.4 🏗️ **Architektur & UML** | System-Design, Klassendiagramme und Workflows (Mermaid) | **v3.1.0** |

Das System arbeitet **100 % lokal** (SQLite + FTS5), ohne API-Kosten und nutzt quelloffene KI-Modelle (HuggingFace, spaCy).

---

## ⚠️ Disclaimer

OpenParlament ist ein **Schülerprojekt** zu Lehr- und Forschungszwecken. Die
Analyseergebnisse basieren auf automatisierten NLP-Verfahren und stellen **keine politische
Stellungnahme** dar. Die Nutzung erfolgt auf eigene Verantwortung.

## 📜 Datenquelle & Lizenz

Alle Plenarprotokolle stammen aus dem offenen Datenportal des Deutschen Bundestags:

> **Bundestag Open Data** — <https://www.bundestag.de/services/opendata>  
> **Bereitsteller:** Deutscher Bundestag
> **Lizenz der Rohdaten:** [Datenlizenz Deutschland – Namensnennung – Version 2.0](https://www.govdata.de/dl-de/by-2-0) (`dl-de/by-2-0`)
> **DCAT-AP.de Metadaten-Konformität:** Lizenz-URI [`http://www.dcat-ap.de/def/licenses/dl-by-de/2.0`](http://www.dcat-ap.de/def/licenses/dl-by-de/2.0) (Lizenzcode `dl-by-de/2.0`)
> **Hinweis zur Datenbearbeitung (§ 3 dl-de/by-2-0):** Die Rohdaten (XML-Plenarprotokolle) wurden für die OpenParlament-Analyse-Pipeline automatisiert extrahiert, bereinigt, strukturiert und verarbeitet.

Der Quellcode dieses Projekts steht unter der **GNU Affero General Public License v3 (AGPL-3.0-or-later)**.
Eine Kopie der Lizenz findet sich in der Datei [`LICENSE`](LICENSE) im Repository-Wurzelverzeichnis.

---

## 📁 Projektstruktur

```
OpenParlament/
├── alembic/            # Database schema migrations & version scripts
│   ├── env.py          # Alembic environment migration configuration
│   └── versions/       # Schema migration scripts (Alembic delta revisions)
├── config/             # Party metadata, faction sizes & benchmark acceptance criteria
├── data/               # SQLite database, precomputed Parquet caches & raw XML protocols
├── docs/               # Lastenheft, Pflichtenheft & Project Architecture documentation
├── scripts/
│   ├── db_init.py              # Datenbank initialisieren & FTS5-Indizes anlegen (CLI: openparlament-init)
│   ├── db_inspect.py           # DB-Inspektor für Tabellenzeilen, FTS5-Status & Schema
│   ├── db_patch.py             # Schema-Migrationen & Nachträgliche Daten-Updates
│   ├── check_db_integrity.py   # Integritäts- & Vollständigkeitsprüfung (Sequence-Gaps, Precomputations)
│   ├── compute_standard_deviations.py # Statistische Varianz- & Extremwert-Integritätsprüfung (Bessel ddof=1)
│   ├── extract_sarcasm_case_studies.py # Extraktion qualitativer Sarkasmus-Fallstudien & Sentiment-Extrema
│   ├── import_xmls.py          # Batch-Import vorhandener XML-Protokolle & Datums-Backfill
│   ├── run_scraper.py          # Protokolle herunterladen, importieren & optional NLP-Scoring (CLI: openparlament-scrape)
│   ├── run_nlp_cli.py          # Standalone NLP-Scoring CLI (CUDA, FP16, Batching) (CLI: openparlament-nlp)
│   ├── run_dashboard.py        # Streamlit-Dashboard Startskript (CLI: openparlament-dashboard)
│   ├── nlp_session.py          # NLPSession-Kontext (Device-Management, Lifecycle)
│   ├── precompute_h3.py         # Precomputation für H3 Asymmetrische Angriffe (Parquet)
│   ├── precompute_framing.py    # Precomputation für Framing-Analyse (TF-IDF)
│   ├── precompute_complexity.py # Precomputation für Sprachkomplexität & Reaction-Gaps
│   ├── train_setfit_sentiment.py # SetFit Few-Shot Fine-Tuning für Sentiment
│   ├── train_stylometry.py     # ML-Trainingsskript für Stylometrie (RandomForest)
│   ├── migrate_text_processing.py # Textverarbeitungs-Migration & Bereinigung
│   ├── benchmark_baseline.py   # Baseline-Benchmarks für NLP- und Analytics-Engines
│   ├── benchmark_spacy_cache.py # Performance-Benchmark für spaCy Caching Layer
│   ├── export_dashboard_plots.py # Automatisierter Dashboard Plot-Scanner & PNG/JSON-Exporter
│   └── old_protocol_parser.py  # Parser für historische Protokolle (WP <= 17)
├── src/
│   ├── models.py       # SQLAlchemy ORM (Sitzung, Redner, Rede, Zwischenruf, SpeechComplexityResult, etc.)
│   ├── database.py     # DB-Engine, Session-Management & SQLite FTS5 Virtual Tables
│   ├── parser.py       # Bundestag-XML-Parser (OOP & Regex-Extractor)
│   ├── scraper.py      # Open-Data-Scraper & Download-Validator
│   ├── nlp.py          # NLP-Engines: SentimentEngine, ToneClassifier, AddresseeDetector, StylometryEngine
│   ├── analytics.py    # 20+ Analytics-Klassen (A–U, Hypothesen H1-H3, Komplexität, Reaction-Gap, ERGM, etc.)
│   ├── text_processing/# Textverarbeitungs-Pipelines und Normalisierung
│   └── app.py          # Streamlit-Dashboard (st.navigation mit 25 Seiten in 6 Kategorien)
├── tests/              # 530+ Unit- & Integrationstests in 5 partitionierten Domain-Ordnern
│   ├── app/            # Dashboard UI & Page Component Tests
│   ├── cli/            # CLI Tools & Script Tests
│   ├── precomputing/   # Precomputation Pipeline Tests
│   ├── processing/     # Parser, NLP, Stylometrie & Analytics Tests
│   └── scraping/       # Scraper & Ingestion Tests
├── alembic.ini         # Alembic database migration configuration
├── pyproject.toml      # Projekt-Konfiguration, Dependencies & CLI-Entrypoints
├── requirements.txt    # Python-Paketabhängigkeiten
├── requirements-render.txt # Streamlit-Rendering Minimal-Abhängigkeiten
├── setup.sh / setup.ps1 # Automatisierte Setup-Skripte für Linux/macOS & Windows
└── LICENSE             # GNU Affero General Public License v3.0
```

---

## 💻 Quick Start

### 1. Repository klonen & Umgebung einrichten

```bash
git clone https://github.com/Jonas-dpp/OpenParlament.git
cd OpenParlament
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm
pip install -e .           # Installiert OpenParlament als Package inkl. CLI-Console-Commands
```

> **Hinweis (Automatisches Setup):**
> Alternativ kannst du auch das automatisierte Setup-Skript ausführen:
> - Linux / macOS: `./setup.sh`
> - Windows (PowerShell): `.\setup.ps1`

### 2. Registrierte Console Commands (pyproject.toml)

Nach der Installation stehen dir globale Konsolenbefehle direkt im Terminal zur Verfügung:

| Befehl | Entsprechungs-Skript | Zweck |
|---|---|---|
| `openparlament-init` | `scripts/db_init.py` | Datenbank initialisieren & FTS5-Indizes anlegen |
| `openparlament-scrape` | `scripts/run_scraper.py` | Protokolle scrapen, importieren & verarbeiten |
| `openparlament-nlp` | `scripts/run_nlp_cli.py` | Standalone NLP-Scoring ausführen |
| `openparlament-dashboard` | `scripts/run_dashboard.py` | Streamlit Dashboard starten |

### 3. Datenbank initialisieren & Integrität prüfen

```bash
# Datenbank und FTS5-Suchindizes initialisieren (oder via: openparlament-init)
python scripts/db_init.py

# Integritätsprüfung ausführen (Prüft Sequenzlücken, ORM-Spalten & Precomputation-Status)
python scripts/check_db_integrity.py
```

### 4. Protokolle herunterladen & importieren

```bash
# 20. Wahlperiode, max. 5 Seiten der Listing-Page (ca. 50 Protokolle)
python scripts/run_scraper.py --wahlperiode 20 --max-pages 5
# Oder via Console Command: openparlament-scrape --wahlperiode 20 --max-pages 5

# Inkl. automatischer NLP-Analyse (Sentiment, Ton, Adressaten)
python scripts/run_scraper.py --wahlperiode 20 --nlp

# GPU-Beschleunigung + Half-Precision (CUDA):
python scripts/run_scraper.py --wahlperiode 20 --nlp --nlp-cuda --nlp-fp16

# Standalone NLP nachträglich auf bestehenden DB-Daten ausführen:
python scripts/run_nlp_cli.py --target all --batch-size 64 --only-unscored
# Oder via Console Command: openparlament-nlp --target all --batch-size 64 --only-unscored
```

### 5. Precomputation & ML-Modelle trainieren

```bash
# Precomputations für lückenlose Dashboard-Performance vorab berechnen:
python scripts/precompute_complexity.py --all
python scripts/precompute_framing.py --all
python scripts/precompute_h3.py --all

# SetFit Few-Shot Sentiment-Domain-Adaption & Evaluierung ausführen:
python scripts/train_setfit_sentiment.py --eval-baseline

# Stylometrie ML-Klassifikator (RandomForest) trainieren:
python scripts/train_stylometry.py --text_layer clean

# Automatisierte Dashboard Plot-Erfassung & PNG/JSON Export:
python scripts/export_dashboard_plots.py --max-configs 500
```

### 6. Dashboard starten

```bash
streamlit run src/app.py
# Oder direkt via Console Command: openparlament-dashboard
```

Die App öffnet sich automatisch unter **http://localhost:8501**.
Die Navigation erfolgt über die **Sidebar** mit 6 gruppierten Kategorien (`st.navigation`).

---

## 🧪 Tests & Qualitätssicherung

Das Projekt verfügt über eine umfassende automatische Test-Suite mit 530+ Unit- und Integrationstests, die in 5 domain-spezifische Test-Ordner unterteilt sind:

```bash
# Gesamte Test-Suite (530+ Tests) ausführen:
python -m pytest tests/ -v

# Gezielte Tests einzelner Domain-Ordner ausführen:
python -m pytest tests/app/ -v          # UI & Dashboard Page Component Tests
python -m pytest tests/cli/ -v          # CLI Tools, Integrity & Security Tests
python -m pytest tests/precomputing/ -v # Precomputation Pipeline Caching Tests
python -m pytest tests/processing/ -v   # Parser, NLP, Stylometrie & Analytics Tests
python -m pytest tests/scraping/ -v     # Scraper & Ingestion Tests
```

---

## 🔄 Clean Rebuild (Datenbank neu aufbauen)

```bash
# 1. Abhängigkeiten aktualisieren
pip install --no-cache-dir -r requirements.txt

# 2. Datenbank komplett droppen und mit allen Spalten & FTS-Indizes neu initialisieren
python scripts/db_init.py --reset

# 3. Frische XML-Dateien scrapen
python scripts/run_scraper.py --wahlperiode 20

# 4. XML-Protokolle in die frische DB importieren
python scripts/import_xmls.py

# 5. NLP-Pipeline für ungescorete Reden ausführen
python scripts/run_nlp_cli.py --batch-size 700 --target all --fp16 --cuda --device-index 0 --only-unscored

# 6. Stylometrie-Modelle trainieren
python scripts/train_stylometry.py --text-layer clean

# 7. Precomputation-Analysen & statistische Audits neu berechnen
python scripts/precompute_complexity.py --all
python scripts/precompute_framing.py --all
python scripts/precompute_h3.py --all
python scripts/compute_standard_deviations.py
python scripts/extract_sarcasm_case_studies.py

# 8. Datenbank-Integrität & Benchmarks verifizieren
python scripts/check_db_integrity.py --strict
python scripts/benchmark_baseline.py
```

---

## 📊 Analytics-Klassen

| Klasse | Modul | Beschreibung | Version |
|---|---|---|---|
| `AggressionsIndex` | `src/analytics.py` | Top-Empfänger und Top-Störer negativer Zwischenrufe, Reziprozitäts-Matrix | v0.8.0 / v3.2.0 |
| `ThemenKarriere` | `src/analytics.py` | Keyword-Trend, Multi-WP-Karriere, Reizwort-Korrelation | v0.8.0 / v2.3.0 |
| `InteraktionsNetzwerk` | `src/analytics.py` | Interaktionsmatrix, ERGM Netzwerkanalyse, NetworkX/Gephi-Export & Polarisierungsindex | v0.8.0 / **v3.2.0** |
| `TonAnalyse` | `src/analytics.py` | Ton-Label-Verteilung (Aggression/Sarkasmus/Humor/Neutral) & Rhetorischer Waffen-Mix | v1.2.0 / **v3.2.0** |
| `AdressatenAnalyse` | `src/analytics.py` | Adressaten-Ranking und Fraktions-Adressierungs-Matrix | v1.2.0 |
| `ScrapingMonitor` | `src/analytics.py` | DB-Füllstand, NLP-Abdeckungsgrad & Korpusstatistiken | v1.2.0 |
| `WahlperiodenVergleich` | `src/analytics.py` | Vergleich von Aggression, Ton und Aktivität über Wahlperioden | v2.0.0 |
| `TOPAnalyse` | `src/analytics.py` | Tagesordnungspunkt-Hostility-Ranking | v2.2.0 |
| `KategorieAnalyse` | `src/analytics.py` | Kategorie-Verteilung & Civility-Index | v2.2.0 |
| `RedeZeitAnalyse` | `src/analytics.py` | Speech-Time-Fairness-Index & Temporale Unterbrechungs-Dynamik | v2.2.0 / **v3.2.0** |
| `SitzungsKlima` | `src/analytics.py` | Composite Temperatur-Index pro Sitzung | v2.4.0 |
| `RednerProfil` | `src/analytics.py` | Rhetorischer Fingerabdruck aus `Rede.tone_scores` JSON | v2.4.0 |
| `RednerVergleich` | `src/analytics.py` | Direktvergleich zweier Abgeordneter & Kosinus-Ähnlichkeits-Permutationstest | **v3.0.0** |
| `FraktionsDynamik` | `src/analytics.py` | Fraktions-Ton-Zeitreihe, Aggressions-Timeline, Sunburst-Hierarchie | **v3.0.0** |
| `FramingAnalysis` | `src/analytics.py` | Vorberechnete TF-IDF Begriffe, Chi-Quadrat-Unabhängigkeitstest & Hypergeometrische Anreicherung | **v3.2.0** |
| `LexicalAnalysis` | `src/analytics.py` | Lexikalische Diversität (TTR, MSTTR) und Informationsdichte | **v3.2.0** |
| `SpeechComplexityAnalysis` | `src/analytics.py` | Wiener Sachtextformel (WSTF) vs Flesch FRE Triangulation & Kruskal-Wallis-Test | **v3.2.0** |
| `ReactionGapAnalysis` | `src/analytics.py` | Eskalationsanalyse (Action-Reaction-Gap) & Reaktionszeiten | **v3.2.0** |
| `SentimentValidation` | `src/analytics.py` | Gold Standard Evaluierung & Confusion Matrix gegen annotierte Referenzdaten | **v3.2.0** |
| `StylometryAnalyzer` | `src/analytics.py` | Syntaktische Merkmale, Fraktions-DNA & RandomForest Blind-Test Klassifikation | **v3.1.0** |
| `StylometryPredictor` | `src/analytics.py` | Inferenz-Engine zur Fraktions-Vorhersage anhand stylometrischer Merkmale | **v3.1.0** |

---

## 🏷️ Precomputation & Offline Pipelines

Um sub-sekundäre Ladezeiten im Dashboard zu garantieren, nutzen komplexe Analysen vorberechnete Tabellen und Parquet-Caches.

### 1. Precomputations berechnen:
```bash
# Framing- & Lexik-Analyse vorab berechnen (TF-IDF):
python scripts/precompute_framing.py --all

# Sprachkomplexität & Action-Reaction-Gap vorab berechnen:
python scripts/precompute_complexity.py --all

# H3 Asymmetrische Angriffs-Netzwerke vorab berechnen:
python scripts/precompute_h3.py --all
```

### 2. Statistische Audits & Sarkasmus-Fallstudien:
```bash
# Statistische Varianz- & Extremwert-Integrität prüfen (Bessel ddof=1):
python scripts/compute_standard_deviations.py

# Qualitative Sarkasmus-Fallstudien & Sentiment-Extrema extrahieren:
python scripts/extract_sarcasm_case_studies.py
```

### 3. Datenbank-Integritätsprüfung:
```bash
python scripts/check_db_integrity.py --strict --verbose
```
Das Skript prüft auf Lücken in Sitzungssequenzen, Chronologie-Inversionen, fehlende ORM-Spalten und ausstehende Precomputation-Ergebnisse.

---

## 📖 Dokumentation

- 🎯 [Lastenheft (Anforderungen)](docs/requirements/Lastenheft.md)
- 🛠️ [Pflichtenheft (Technische Umsetzung)](docs/specs/Pflichtenheft.md)
- 🗺️ [Project Tracker & Architecture](docs/project/Project.md)

---

## 🏗️ Tech Stack

| Schicht | Technologie | Zweck |
|---|---|---|
| Datenhaltung | SQLite + SQLAlchemy 2.x + FTS5 | Lokale relationale Datenbank, ORM, Volltextsuche |
| Datenbeschaffung | Requests + BeautifulSoup (lxml) | Bundestag Open Data scraping & Validierung |
| Parsing | BeautifulSoup + Regex | XML-Protokoll-Zerlegung in ORM-Objekte |
| NLP – Sentiment | HuggingFace `distilbert-base-multilingual-cased` / SetFit | Sentiment-Analyse (lokal, offline) |
| NLP – Ton-Analyse | HuggingFace `mDeBERTa-v3` (Zero-Shot) | Klassifikation: Aggression / Sarkasmus / Humor / Neutral |
| NLP – Adressaten | Regel-Engine + HuggingFace `xlm-roberta` NER | Erkennung von Fraktionen und Personen als Adressaten |
| NLP – Lesbarkeit | spaCy `de_core_news_sm` + WSTF / Flesch FRE | Berechnung der Wiener Sachtextformel & Flesch Reading Ease |
| ML – Stylometrie | Scikit-Learn `RandomForestClassifier` | Extraktion syntaktischer Merkmale und Fraktions-Klassifikation |
| Data Science | Pandas + NetworkX + Statsmodels | Aggregation, ERGM, Kruskal-Wallis, Levene, Permutationstests, Graph-Export |
| Frontend | Streamlit ≥ 1.36 + Plotly | Interaktives Web-Dashboard mit nativer Navigation (`st.navigation`) |

---

## 📜 Changelog

### v3.2.0 (2026-04-18) — *Aktuelle Version*

- **🔬 Hypothesen-Prüfung & Advanced Analytics:**
  - Live quantitative Prüfung der drei wissenschaftlichen Hypothesen H1 (Opposition), H2 (Polarisierung) & H3 (Asymmetrische Angriffe).
  - **Framing-Analyse:** TF-IDF Schlüsselbegriffe, Chi-Quadrat-Unabhängigkeitstest & hypergeometrische Schlüsselwort-Anreicherung.
  - **Lexik-Analyse:** TTR, MSTTR & lexikalische Dichte je Fraktion.
  - **Sprachkomplexität:** Methodische Triangulation von Wiener Sachtextformel (WSTF) und Flesch Reading Ease (FRE), Kruskal-Wallis-Test.
  - **Eskalationsanalyse (Action-Reaction-Gap):** Sentiment-Differenz zwischen Rednern und Reaktionen samt zeitlicher Eskalationsgeschwindigkeit.
  - **Polarisierungsindex:** Fraktionsübergreifende Angriffsquote und feindselige Außeninteraktionen.
  - **Sentiment-Modell-Validierung:** Evaluation gegen Gold Standard Datensatz (N=100) mit Confusion Matrix und F1-Scores.
- **🛠️ Tools & Qualitätssicherung:**
  - `scripts/check_db_integrity.py`: CLI-Werkzeug zur Prüfung von Sequenzlücken, Inversionen und Precomputation-Vollständigkeit.
  - `scripts/export_dashboard_plots.py`: Automatisierter Plot-Scanner und Exporter für PNG und JSON Manifeste.
  - Partitionierte Test-Suite (`tests/app`, `tests/cli`, `tests/precomputing`, `tests/processing`, `tests/scraping`) mit 530+ Tests.
  - Neugliederung der Sidebar in 6 Hauptkategorien (25 Seiten).

---

### v3.1.0

- **🧬 Stylometrie (Blind-Test):** Syntaktische Feature-Extraktion (Satzlänge, Passiv quote, TTR, Füllwortquote) und ML-Klassifikator (`RandomForestClassifier`) zur Fraktionserkennung.
- `scripts/train_stylometry.py`: Trainingsskript für den stylometrischen Klassifikator.
- Dokumentationserweiterungen und Codebase-Optimierungen.

---

### v3.0.0 (2026-04-07)

- **👥 Redner-Vergleich:** Direktvergleich zweier Abgeordneter mit überlappenden Radar-Charts, Divergenz-Balken und Kosinus-Ähnlichkeit.
- **📡 Fraktions-Dynamik:** Sunburst-Hierarchie, gestapelte Flächendiagramme & monatliche Aggressions-Zeitreihen je Fraktion.
- Startseite: Live-Metriken, Quick-Start-Guide & SPDX-2.3 Software Bill of Materials (SBOM).
