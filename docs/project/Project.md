# Agent Memory & Session Tracker

*Note for Dev/AI: Always review this file at the start of a session and update it before ending a session. Update the "Status" column in the Version Plan as milestones are reached.*

---

## Current Version: v3.2.0

### Completed Milestones
* [x] v0.0.0 — Project conceptualisation, Techstack defined (Python, SQLite, SQLAlchemy, Streamlit, HuggingFace).
* [x] v0.0.0 — Lastenheft und Pflichtenheft erstellt, Experimente (Aggressions-Index, Themen-Karriere, Netzwerk) definiert.
* [x] v0.1.0 — `requirements.txt` erstellt, Repository-Struktur (`src/`, `scripts/`, `data/`, `tests/`) angelegt.
* [x] v0.1.0 — SQLAlchemy ORM-Setup (`src/models.py` & `src/database.py`): Klassen `Sitzung`, `Redner`, `Rede`, `Zwischenruf` mit Relationships, vollständig OOP und typed (mapped_column).
* [x] v0.5.0 — `src/parser.py`: `BundestagXMLParser` parst strukturierte Bundestag-XML-Dateien (19./20. Wahlperiode) in ORM-Objekte.
* [x] v0.5.0 — `src/scraper.py`: `BundestagScraper` lädt Protokoll-XMLs von der Bundestag-Open-Data-Seite mit Paginierung und Duplikatschutz.
* [x] v0.5.0 — `scripts/run_scraper.py`: `ProtocolImporter` persistiert ParseResults in die DB mit Deduplizierung.
* [x] v0.8.0 — `src/nlp.py`: `SentimentEngine` kombiniert regelbasierte Schnellklassifikation mit HuggingFace-Inferenz (lazy loading, CPU-fähig).
* [x] v0.8.0 — `src/analytics.py`: `AggressionsIndex`, `ThemenKarriere`, `InteraktionsNetzwerk` — alle drei Kern-Experimente als Pandas-DataFrame-Ausgabe.
* [x] v1.0.0 — **Quality & Performance (Final Gate):**
  * Architektur-Konsistenz: Doppelte Imports in `nlp.py` entfernt; "Triple Crown" — `SentimentEngine`, `ToneClassifier` und `AddresseeDetector` folgen identischer interner Logik.
  * `AddresseeDetector` erhält `batch_size`-Parameter; `_neural_detect_batch()` extrahiert; `NLPSession` übergibt `batch_size` korrekt an alle drei Engines.
  * Shared-mutable-object-Bugs in Fehler-Fallbacks von `_neural_classify_batch`, `run_scraper.py` und `run_nlp_cli.py` behoben.
  * Test-Suite auf 151 Tests erweitert.
* [x] v1.0.0 — `src/app.py`: Streamlit-Dashboard mit 3 Tabs (Aggressions-Radar, Themen-Trend, Interaktions-Netzwerk), globaler Sidebar-Filter.
* [x] v1.0.0 — `tests/test_core.py`: Vollständige Unit-Tests für DB, Models, Parser, NLP und Analytics (in-memory SQLite, kein torch nötig).
* [x] v1.1.0 — **Scraper-Bugfixes & neue NLP-Engines:** `ToneClassifier` (mDeBERTa-v3) und `AddresseeDetector` (xlm-roberta NER); neue DB-Felder in `Zwischenruf`.
* [x] v1.2.0 — **Wahlperioden-Bugfix & Dashboard-Erweiterung:** Tabs D (Ton-Analyse), E (Adressaten-Analyse), F (Scraping-Monitor) in `src/app.py`.
* [x] v2.0.0 — **Multi-Wahlperioden-Analyse & Graph-Export:** Tab G (Wahlperioden-Vergleich); GraphML/GEXF-Export; CI/CD via GitHub Actions; `pyproject.toml`.
* [x] v2.1.0 — **Navigation & UX:** Startseite (Landing Page); Sidebar-Navigation; URL-Routing; `render_*()`-Funktionen; `DEFAULT_PAGE`-Konstante.
* [x] v2.2.0 — **Untapped Data Analytics:** `TOPAnalyse` (Tab H), `KategorieAnalyse` (Tab I), `RedeZeitAnalyse` (Tab M).
* [x] v2.3.0 — **ThemenKarriere-Erweiterungen:** Multi-WP-Keyword-Trend, Reizwort-Index, Peak-Sitzungen je Wahlperiode.
* [x] v2.4.0 — **Advanced Analytics:** `SitzungsKlima` (Tab L: Temperatur-Index), `RednerProfil` (Tab K: Speaker DNA).
* [x] v2.5.0 — **Native Navigation & DB-Übersicht:** `st.navigation` / `st.Page` (Streamlit ≥ 1.36); Tab P DB-Übersicht (Sankey, ERD, Schema-Inspector).
* [x] v3.0.0 — **UX-Verbesserung & Neue Analysen:** `RednerVergleich` (Experiment N) und `FraktionsDynamik` (Experiment O). 16 Dashboard-Seiten.
* [x] v3.1.0 — **Stylometrie & ML-Klassifikator:** `StylometryAnalyzer` und `StylometryEngine` (Satzlänge, Passiv quote, TTR) mit `RandomForestClassifier` für Fraktions-Blind-Test (Experiment P).
* [x] v3.2.0 — **Hypothesen-Prüfung, Advanced Analytics & Quality Gate:**
  * **Wissenschaftliche Hypothesen-Prüfung:** Live quantitative Tests für H1 (Opposition), H2 (Polarisierung) & H3 (Asymmetrische Angriffe).
  * **Advanced Analytics Modules:** `FramingAnalysis` (TF-IDF & Chi2/Hypergeometrische Anreicherung), `LexicalAnalysis` (TTR & MSTTR), `SpeechComplexityAnalysis` (WSTF & Flesch FRE Triangulation), `ReactionGapAnalysis` (Action-Reaction-Gap), `SentimentValidation` (Gold Standard N=100) & Polarisierungsindex.
  * **Sub-sekundäre Precomputation Pipelines:** `precompute_h3.py`, `precompute_framing.py`, `precompute_complexity.py`.
  * **Automatisierte Qualitätssicherung & CLI-Tools:** `check_db_integrity.py` (Lückenerkennung & Schema-Validierung), `compute_standard_deviations.py` (Statistische Varianz- & Extremwert-Auditierung), `extract_sarcasm_case_studies.py` (Sarkasmus-Fallstudien-Extraktion) und `export_dashboard_plots.py` (Plot-Scanner & PNG/JSON-Exporter).
  * **Navigation & Struktur:** 25 Dashboard-Seiten in 6 Sidebar-Kategorien (`st.navigation`). Partitionierte Test-Suite in 5 Subdirectories (`tests/app`, `tests/cli`, `tests/precomputing`, `tests/processing`, `tests/scraping`) mit 530+ bestandenen Tests.

---

## Recommended Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **Datenbeschaffung & Lizenzierung** | `Python` + `Requests` | Automatisierter Abruf der XML-Protokolle (Bundestag Open Data); konform zur *Datenlizenz Deutschland – Namensnennung – Version 2.0* (`dl-de/by-2-0`, DCAT-AP.de: `http://www.dcat-ap.de/def/licenses/dl-by-de/2.0` / `dl-by-de/2.0`). |
| **Datenhaltung & Migration** | `SQLite` + `SQLAlchemy 2.x` + `Alembic` + `FTS5` | Keine Server- oder Cloud-Kosten. Relationale Tabellen mit schema-versionierten Migrationen, Volltextsuche und ORM-Mapping. |
| **Parsing** | `Regex` + `BeautifulSoup (lxml)` | Bundestags-Protokolle sind unstrukturierter Text im XML-Gewand. |
| **NLP (Sentiment & Ton)** | `HuggingFace` + `SetFit` | Lokale KI-Inferenz (DistilBERT, mDeBERTa-v3, XLM-RoBERTa). Vermeidet API-Kosten und garantiert Offline-Betrieb. |
| **Data Science & Stats** | `Pandas` + `NetworkX` + `Statsmodels` | Aggregation, ERGM, Kruskal-Wallis-Tests, Levene-Tests, Permutationsprüfungen. |
| **Frontend (GUI)** | `Streamlit ≥ 1.36` + `Plotly` | Native Navigation (`st.navigation`), interaktive Plotly-Charts und Mermaid-Diagramme. |

---

## Complete Version Plan (v0.0.0 → v3.2.0)

| Version | Focus Area | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **v0.0.0** | Setup & Planung | Repo scaffolding, OOA/OOD Modelle definiert, Lasten/Pflichtenheft. | ✅ Done |
| **v0.1.0** | Proof of Concept | `requirements.txt`, SQLAlchemy Models (`models.py`, `database.py`), `scripts/db_init.py`. | ✅ Done |
| **v0.5.0** | Data Pipeline | `parser.py` (BundestagXMLParser), `scraper.py` (BundestagScraper), `run_scraper.py` (ProtocolImporter). | ✅ Done |
| **v0.8.0** | MVP GUI & Analytics | `nlp.py` (SentimentEngine), `analytics.py` (AggressionsIndex, ThemenKarriere, InteraktionsNetzwerk). | ✅ Done |
| **v1.0.0** | Release | `app.py` (Streamlit-Dashboard mit 3 Tabs), Unit-Tests, vollständige Dokumentation. | ✅ Done |
| **v1.1.0** | Scraper-Robustheit & Erweiterte NLP | Scraper-Bugfixes; `ToneClassifier` (Aggression/Sarkasmus/Humor); `AddresseeDetector`; neue DB-Felder. | ✅ Done |
| **v1.2.0** | Dashboard-Erweiterung | Wahlperioden-Bugfix; Tabs D (Ton-Analyse), E (Adressaten-Analyse), F (Scraping-Monitor). | ✅ Done |
| **v2.0.0** | Vollständige Wahlperioden-Analyse | Multi-Wahlperioden-Vergleich (Tab G); NetworkX/Gephi-Export; CI/CD via GitHub Actions. | ✅ Done |
| **v2.1.0** | Navigation & UX | Startseite (Landing Page); Sidebar-Navigation; URL-Query-Parameter-Routing. | ✅ Done |
| **v2.2.0** | Untapped Data Analytics | `TOPAnalyse` (Tab H), `KategorieAnalyse` (Tab I), `RedeZeitAnalyse` (Tab M). | ✅ Done |
| **v2.3.0** | Longitudinal & Sentiment Upgrades | `ThemenKarriere.multi_wp_keyword_trend`, `keyword_aggression_correlation`, `most_polarizing_keywords`. | ✅ Done |
| **v2.4.0** | Advanced Analytics | `SitzungsKlima` (Tab L: Temperatur-Index), `RednerProfil` (Tab K: Speaker DNA). | ✅ Done |
| **v2.5.0** | Native Navigation & Enhanced Home | `st.navigation` / `st.Page` (Streamlit ≥ 1.36) mit 4 gruppierten Sidebar-Kategorien; DB-Übersicht. | ✅ Done |
| **v3.0.0** | UX-Verbesserung & Neue Analysen | `RednerVergleich` und `FraktionsDynamik`. 16 Seiten. Startseite: v3.0.0-Badge, SPDX-2.3 SBOM. | ✅ Done |
| **v3.1.0** | Stylometrie & ML-Klassifikator | `StylometryAnalyzer` und `StylometryEngine`, RandomForest Klassifikation für Fraktions-Blind-Test. | ✅ Done |
| **v3.2.0** | Hypothesen-Prüfung & Full Analytics Suite | Hypothesen-Prüfung (H1–H3), Framing, Lexik, Sprachkomplexität (WSTF & FRE), Action-Reaction-Gap, Polarisierung, Modell-Validierung, Precomputations, `check_db_integrity.py`, Plot-Exporter, 25 Seiten in 6 Kategorien. | ✅ Done |

---

## Architecture Summary

```text
┌──────────────────────────────────────────────────────────────┐
│  DATA PIPELINE (Backend / Offline)                           │
│                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │ Bundestag API   │───>│ BundestagScraper                │  │
│  │ (Open Data)     │    │ src/scraper.py                  │  │
│  └─────────────────┘    └─────────────────────────────────┘  │
│                                  │                           │
│                                  v                           │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │ SentimentEngine │<───│ BundestagXMLParser              │  │
│  │ ToneClassifier  │    │ src/parser.py                   │  │
│  │ AddresseeDetect │    └─────────────────────────────────┘  │
│  │ StylometryEngine│                                         │
│  │ src/nlp.py      │                                         │
│           │                      │                           │
│           └──────────────────────v                           │
│                         [ SQLite Database + FTS5 ]           │
│                         via SQLAlchemy ORM & Alembic         │
│                         (src/models.py + database.py + alembic)
│                                  │                           │
│                                  v                           │
│                         [ Precomputation Scripts ]           │
│                         precompute_h3 / framing / complexity │
└──────────────────────────────────────────────────────────────┘
                                   │
                                   v
┌───────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER (Frontend & Analytics)                    │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ AggressionsIndex │ ThemenKarriere │ InteraktionsNetzwerk │ │
│  │ TonAnalyse │ AdressatenAnalyse │ ScrapingMonitor         │ │
│  │ WahlperiodenVergleich │ TOPAnalyse │ KategorieAnalyse    │ │
│  │ RedeZeitAnalyse │ SitzungsKlima │ RednerProfil           │ │
│  │ RednerVergleich │ FraktionsDynamik │ FramingAnalysis     │ │
│  │ LexicalAnalysis │ SpeechComplexityAnalysis               │ │
│  │ ReactionGapAnalysis │ SentimentValidation │ Stylometry   │ │
│  │ StylometryPredictor │ src/analytics.py                   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                  │                            │
│                                  v                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Streamlit Dashboard  src/app.py  (v3.2.0)                │ │
│  │ Native st.navigation / st.Page (Streamlit ≥ 1.36)        │ │
│  │ 25 Seiten in 6 Kategorien:                               │ │
│  │  1. Start & Hypothesen: 1.1 🏠 Startseite • 1.2 🔬 Hypothesen│ │
│  │  2. Themen & Struktur: 2.1 🏛️ TOPs • 2.2 📈 Trend • 2.3 🎯 Framing│ │
│  │     • 2.4 🕸️ Netzwerk                                    │ │
│  │  3. Sprache & Stil: 3.1 🎭 Ton • 3.2 📚 Lexik • 3.3 📏 Komplexität│ │
│  │     • 3.4 🧬 Stylometrie                                 │ │
│  │  4. Stimmung & Konflikt: 4.1 🌡️ Klima • 4.2 🔥 Aggression │ │
│  │     • 4.3 ⚖️ Polarisierung • 4.4 ⚡ Eskalation • 4.5 🎯 Adressaten│ │
│  │     • 4.6 👏 Reaktions-Analyse                           │ │
│  │  5. Akteure & Parteien: 5.1 🎤 Profil • 5.2 👥 Vergleich   │ │
│  │     • 5.3 📡 Dynamik • 5.4 ⏱️ Redezeit • 5.5 ⚖️ Wahlperioden│ │
│  │  6. Daten & Technik: 6.1 📊 Monitor • 6.2 🗄️ DB • 6.3 ✅ Validierung│ │
│  │     • 6.4 🏗️ Architektur & UML                          │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1 — Sub-sekundäre Precomputation Caching Architecture
Complex NLP analytics (TF-IDF framing matrices, Wiener Sachtextformel readability, Action-Reaction-Gap escalation velocities, and H3 asymmetric attack networks) require processing hundreds of thousands of speech tokens. To ensure instant dashboard rendering without UI lag, precomputation CLI scripts (`precompute_framing.py`, `precompute_complexity.py`, `precompute_h3.py`) persist clean aggregates into optimized SQLite tables and compressed Parquet files (`data/h3_precomputed.parquet`).

### 2 — Methodological Triangulation (WSTF vs Flesch FRE)
To guarantee statistical rigor when evaluating speech readability, the system implements dual readability indices: the Wiener Sachtextformel (WSTF, tailored for German academic texts) and Flesch Reading Ease (FRE, Amstad German adaptation). Correlation analysis ($|r| > 0.80$) and non-parametric Kruskal-Wallis tests validate consistency across political factions.

### 3 — Disjoint Gold Standard Validation & Domain Adaptation
To prevent data leakage during SentimentEngine evaluation, the Gold Standard test set ($N=100$) and SetFit Few-Shot fine-tuning training set ($N=120$) are declared strictly disjoint. Automated evaluation in `SentimentValidation` computes confusion matrices, precision, recall, and macro F1-scores directly inside the dashboard.

### 4 — Automated Integrity Checks & SHA-256 Plot Auditing
`scripts/check_db_integrity.py` performs post-parsing and post-precomputation validation to identify non-linear parliamentary session sequence gaps, date inversions, missing ORM columns, or pending precomputations. `scripts/compute_standard_deviations.py` computes sample standard deviations (Bessel ddof=1) and audits extreme sentiment values, while `scripts/extract_sarcasm_case_studies.py` mines sentiment extremes into qualitative sarcasm case studies. `scripts/export_dashboard_plots.py` utilizes SHA-256 chart hashing to discover and export deduplicated Plotly visualizations across all 25 pages.

### 5 — Console Entry Points & Automated Setup Workflows
The application registers global console entry points in `pyproject.toml` (`openparlament-init`, `openparlament-scrape`, `openparlament-nlp`, and `openparlament-dashboard`), wrapping Python scripts into seamless terminal commands. Cross-platform setup scripts (`setup.sh` and `setup.ps1`) automate environment setup, database initialization, date patching, XML importing, text migration, and integrity verification.

### 6 — Schema Migrations via Alembic
Database schema alterations (such as adding text processing, framing, and lexical analysis tables) are version-controlled via Alembic (`alembic/env.py` and `alembic/versions/`). This ensures safe incremental schema migrations without data loss across environment deployments.
