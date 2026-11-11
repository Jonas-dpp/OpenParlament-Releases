# 🛠️ Pflichtenheft: Demokratie-Mining

## 1. Einleitung & Systemzweck
Das System "Demokratie-Mining" dient der systematischen, quantitativen Analyse der parlamentarischen Kommunikation im Deutschen Bundestag (Fokus auf eine oder mehrere volle Wahlperioden).
Die Software soll nicht nur abbilden, was gesagt wird, sondern durch Data Mining **implizites Verhalten** sichtbar machen. Das System liefert die technische Infrastruktur, um folgende übergeordnete Fragen datenbasiert zu beantworten:
* Wer triggert wen?
* Welche Themen werden "totgeschwiegen" oder dominieren?
* Wie aggressiv ist die Debattenkultur objektiv messbar?
* Welche Tagesordnungspunkte provozieren strukturell die meisten negativen Reaktionen?
* Wird Redezeit fair proportional zur Fraktionsgröße verteilt?
* Wie entwickeln sich parlamentarische Beziehungen über eine gesamte Wahlperiode?
* Treffen die wissenschaftlichen Hypothesen zu Oppositions-Sentiment (H1), Polarisierung bei Reizthemen (H2) und asymmetrischen Störnetzwerken (H3) empirisch zu?

## 2. Systemarchitektur & Tech-Stack
Das System nutzt ausschließlich kostenlose Open-Source-Tools und setzt sich wie folgt zusammen:

| Schicht / Phase | Tool / Technologie | Technischer Zweck |
| :--- | :--- | :--- |
| **Datenbeschaffung & Lizenzierung** | `Python` + `Requests` | Automatisierter Abruf der XML/JSON-Protokolle (Bundestag Open Data); konform zur *Datenlizenz Deutschland – Namensnennung – Version 2.0* (`dl-de/by-2-0`, DCAT-AP.de: `http://www.dcat-ap.de/def/licenses/dl-by-de/2.0` / `dl-by-de/2.0`). |
| **Datenhaltung & Migration** | `SQLite` + `SQLAlchemy 2.x` + `Alembic` + `FTS5` | Lokale relationale Datenbank mit schema-versionierten Migrationen, Volltextsuche und ORM-Mapping für Rohdaten und Analyseergebnisse. |
| **Parsing-Engine** | `Regex` + `BeautifulSoup (lxml)` | Trennung von Sprecher, Fraktion, Redetext und in Klammern gesetzten Zwischenrufen. |
| **Analyse & Logik** | `Pandas` + `Scikit-Learn` + `Statsmodels` | Statistische Auswertung, ML-Klassifikation, ERGM-Netzwerkmodelle, Kruskal-Wallis- und Permutationstests. |
| **NLP (KI)** | `HuggingFace` + `spaCy` + `SetFit` | Lokale Stimmungsanalyse (DistilBERT), Ton-Klassifikation (mDeBERTa-v3), Adressaten-NER (XLM-RoBERTa), Wiener Sachtextformel & Flesch FRE. |
| **Frontend (GUI)** | `Streamlit ≥ 1.36` + `Plotly` | Interaktives Web-Dashboard mit nativer 6-Kategorien Sidebar-Navigation (`st.navigation`). |
| **Graph & Export** | `NetworkX` / `Gephi` / `GraphML` | Erstellung, Visualisierung und Export von Interaktionsgraphen. |
| **Precomputation & Audit Tooling** | `Python` CLI Scripts + `Parquet` | Vorberechnete Tabellen für sub-sekundäre Dashboard-Ladezeiten (`precompute_h3`, `precompute_framing`, `precompute_complexity`), DB-Integritätsprüfung (`check_db_integrity.py`), statistisches Audit (`compute_standard_deviations.py`) und Sarkasmus-Extraktion (`extract_sarcasm_case_studies.py`). |

## 3. Datenmodell (OOD & Datenbank-Schema)
Das System extrahiert und speichert folgende 9 zentralen ORM-Entitäten:
* **`Sitzung`** (`sitzungen`): Speichert Sitzungsmetadaten (`sitzungs_id`, `wahlperiode`, `sitzungsnr`, `datum`, `wochentag`, `titel`, `gesamtwortzahl`).
* **`Redner`** (`redner`): Speichert Personendaten (`redner_id`, `bundestag_id`, `vorname`, `nachname`, `titel`, `fraktion`, `partei`).
* **`Rede`** (`reden`): Speichert Redebeiträge samt Textschichten und NLP-Metriken (`rede_id`, `sitzung_id` FK, `redner_id` FK, `text`, `raw_text`, `clean_text`, `normalized_text`, `wortanzahl`, `tagesordnungspunkt`, `sentiment_score`, `ton_label`, `tone_aggression`, `tone_sarkasmus`, `tone_humor`, `tone_neutral`, `tone_scores` JSON, `adressaten`).
* **`Zwischenruf`** (`zwischenrufe`): Speichert verbale oder non-verbale Einwürfe während einer Rede (`ruf_id`, `rede_id` FK, `text`, `fraktion`, `sentiment_score`, `kategorie`, `ton_label`, `tone_aggression`, `tone_sarkasmus`, `tone_humor`, `tone_neutral`, `tone_scores` JSON, `adressaten`).
* **`StylometrieFeatures`** (`stylometrie_features`): Speichert syntaktische Stilmerkmale einer Rede (`id`, `rede_id` FK, `avg_sentence_length`, `passive_ratio`, `type_token_ratio`, `fill_word_ratio`, `punctuation_freq` JSON).
* **`SpeechComplexityResult`** (`speech_complexity_results`): Speichert Lesbarkeits- und Sprachkomplexitätsindizes (`id`, `rede_id` FK, `wahlperiode`, `redner_id` FK, `wstf_score`, `flesch_score`, `avg_sentence_length`, `subordinate_clause_ratio`, `avg_word_length`, `long_word_ratio`, `lexical_diversity`).
* **`ReactionGapResult`** (`reaction_gap_results`): Speichert Eskalationsmetriken (`id`, `rede_id` FK, `zwischenruf_id` FK, `topic_id`, `speaker_sentiment`, `reaction_sentiment`, `gap`, `time_to_first_interjection`).
* **`LexicalResult`** (`lexical_results`): Speichert fraktionsweite lexikalische Diversitäts- und Dichtemetriken (`id`, `wahlperiode`, `fraktion`, `ttr`, `msttr`, `lexical_density`, `total_tokens`, `total_types`).
* **`FramingResult`** (`framing_results`): Speichert vorberechnete TF-IDF Schlüsselbegriffe je Fraktion (`id`, `wahlperiode`, `fraktion`, `term`, `score`).

## 4. Experimente & Datenanalysen

### A. Der "Aggressions-Index" (Zwischenruf-Mining) – Tab A
**Analytics-Klasse:** `AggressionsIndex`. Top-Empfänger und Top-Störer negativer Zwischenrufe sowie Reziprozitäts-Matrix (Austeilen vs. Einstecken).

### B. Die "Themen-Karriere" (Trend-Mining) – Tab B
**Analytics-Klasse:** `ThemenKarriere`. Keyword-Trend-Analyse, Multi-Wahlperioden-Zeitreihen, Peak-Sitzungen und Reizwort-Aggressions-Korrelation.

### C. Das "Netzwerk der Feindschaften" (Graph-Mining) – Tab C
**Analytics-Klasse:** `InteraktionsNetzwerk`. Adjazenzmatrix-Heatmap, ERGM-Netzwerkanalyse, GraphML/GEXF-Export, interaktiver Timeline-Slider und Polarisierungsindex.

### D. Ton-Analyse – Tab D
**Analytics-Klasse:** `TonAnalyse`. Verteilung der Ton-Labels (Aggression/Sarkasmus/Humor/Neutral) nach Fraktion und Zeitverlauf sowie Rhetorischer Waffen-Mix (gestapeltes 100%-Balkendiagramm).

### E. Adressaten-Erkennung – Tab E
**Analytics-Klasse:** `AdressatenAnalyse`. Top-Adressaten und Fraktions-Adressierungs-Matrix.

### F. Scraping-Monitor – Tab F
**Analytics-Klasse:** `ScrapingMonitor`. DB-Füllstand, NLP-Abdeckungsgrad und Korpusstatistiken.

### G. Wahlperioden-Vergleich – Tab G
**Analytics-Klasse:** `WahlperiodenVergleich`. Vergleichende Analyse von Aggression, Ton und Redeaktivität über mehrere Legislaturperioden hinweg.

### H. Tagesordnungspunkt-Analyse – Tab H
**Analytics-Klasse:** `TOPAnalyse`. Tagesordnungspunkt-Hostility-Ranking.

### I. Reaktions-Analyse (Kategorie-Index) – Tab I
**Analytics-Klasse:** `KategorieAnalyse`. Beifall vs. Widerspruch Ratio und Civility-Index.

### M. Redezeit-Gerechtigkeit – Tab M
**Analytics-Klasse:** `RedeZeitAnalyse`. Speech-Time-Fairness-Index (proportional zur Fraktionsgröße) und temporale Unterbrechungs-Dynamik.

### L. Debattenklima-Index – Tab L
**Analytics-Klasse:** `SitzungsKlima`. Composite Temperatur-Index pro Sitzung ($[0,1]$) mit rollierendem Durchschnitt und Hot-Session-Ranking.

### K. Redner-Profil (Speaker DNA) – Tab K
**Analytics-Klasse:** `RednerProfil`. Rhetorischer Fingerabdruck jedes Abgeordneten als Radar-Chart basierend auf `Rede.tone_scores`.

### N. Redner-Vergleich – Tab N (v3.0.0)
**Analytics-Klasse:** `RednerVergleich`. Direktvergleich zweier Abgeordneter mit überlagerten Radar-Charts, Ton-Delta-Balken, Gruppenbalken und Kosinus-Ähnlichkeits-Permutationstest.

### O. Fraktions-Dynamik – Tab O (v3.0.0)
**Analytics-Klasse:** `FraktionsDynamik`. Sunburst-Hierarchie (Fraktion → Ton), gestapelte Ton-Flächendiagramme und monatliche Aggressions-Zeitreihen.

### P. Stylometrie & Blind-Test – Tab P (v3.1.0)
**Analytics-Klasse:** `StylometryAnalyzer`, `StylometryPredictor` und `StylometryEngine`. Extraktion syntaktischer Stilmerkmale (Satzlänge, Passiv quote, TTR, Füllwortquote) und ML-Klassifikation (`RandomForestClassifier`) für den Fraktions-Blind-Test.

### Q. Lexik-Analyse (v3.2.0)
**Analytics-Klasse:** `LexicalAnalysis`. Vergleichende Analyse von Type-Token-Ratio (TTR), Mean Segmental TTR (MSTTR) und lexikalischer Dichte.

### R. Framing-Analyse (v3.2.0)
**Analytics-Klasse:** `FramingAnalysis`. Vorberechnete TF-IDF Schlüsselbegriffe, Chi-Quadrat-Unabhängigkeitstest und hypergeometrische Anreicherungstests.

### S. Sprachkomplexität (v3.2.0)
**Analytics-Klasse:** `SpeechComplexityAnalysis`. Methodische Triangulation von Wiener Sachtextformel (WSTF / Schulstufe) und Flesch Reading Ease (FRE / Amstad) mit non-parametrischem Kruskal-Wallis-Test.

### T. Eskalationsanalyse / Action-Reaction-Gap (v3.2.0)
**Analytics-Klasse:** `ReactionGapAnalysis`. Sentiment-Differenz zwischen Rednern und ersten Zwischenrufen ($Gap$) sowie Eskalationsgeschwindigkeit.

### U. Modell-Validierung (v3.2.0)
**Analytics-Klasse:** `SentimentValidation`. Validierung des automatisieren Sentiment-Modells gegen einen manuell annotierten Gold-Standard-Testdatensatz ($N=100$) mit Confusion Matrix und F1-Scores.

---

## 5. GUI & Frontend-Anforderungen
Die Streamlit-App macht alle Analysen über eine moderne, interaktive Browser-Oberfläche ohne Programmierkenntnisse bedienbar.

Die Navigation erfolgt über die **Sidebar** mit 6 gruppierten Hauptkategorien (`st.navigation`, Streamlit ≥ 1.36) für insgesamt 25 nummerierte Seiten:

| Gruppe | Enthaltene Seiten (mit Seitennummern) |
| :--- | :--- |
| **1. Start & Hypothesen** | 1.1 🏠 Startseite, 1.2 🔬 Hypothesen-Prüfung |
| **2. Themen & Struktur** | 2.1 🏛️ Tagesordnungspunkte, 2.2 📈 Themen-Trend, 2.3 🎯 Framing-Analyse, 2.4 🕸️ Interaktions-Netzwerk |
| **3. Sprache & Stil** | 3.1 🎭 Ton-Analyse, 3.2 📚 Lexik-Analyse, 3.3 📏 Sprachkomplexität, 3.4 🧬 Stylometrie (Blind-Test) |
| **4. Stimmung & Konflikt** | 4.1 🌡️ Debattenklima-Index, 4.2 🔥 Aggressions-Radar, 4.3 ⚖️ Polarisierungsindex, 4.4 ⚡ Eskalationsanalyse, 4.5 🎯 Adressaten-Analyse, 4.6 👏 Reaktions-Analyse |
| **5. Akteure & Parteien** | 5.1 🎤 Redner-Profil, 5.2 👥 Redner-Vergleich, 5.3 📡 Fraktions-Dynamik, 5.4 ⏱️ Redezeit-Gerechtigkeit, 5.5 ⚖️ Wahlperioden-Vergleich |
| **6. Daten, Technik & Validierung** | 6.1 📊 Scraping-Monitor, 6.2 🗄️ DB-Übersicht, 6.3 ✅ Modell-Validierung, 6.4 🏗️ Architektur & UML |

