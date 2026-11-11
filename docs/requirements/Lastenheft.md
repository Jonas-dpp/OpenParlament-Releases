# 🎯 Lastenheft: Demokratie-Mining

**Projekt:** Demokratie-Mining – Die digitale Röntgenaufnahme des Bundestags  
**Dokumentenart:** Lastenheft (Anforderungsspezifikation)  
**Version:** v3.2.0
**Datum:** 18.04.2026

---

## 1. Einführung und Zielsetzung
Das Projekt "Demokratie-Mining" zielt darauf ab, die parlamentarische Kommunikation im Deutschen Bundestag quantitativ und qualitativ messbar zu machen. Das System soll eine Brücke zwischen riesigen, unstrukturierten Textmengen (Sitzungsprotokollen) und dem Endnutzer (Bürger, Journalisten, Forscher) schlagen. 
Hauptziel ist es, verborgene politische Dynamiken – wie die Aggressivität der Debattenkultur, thematische Konjunkturen, fraktionsübergreifende Netzwerke, rhetorische Stilmittel und wissenschaftliche Hypothesen zur Parlamentskommunikation – durch Data Mining sichtbar zu machen und über eine intuitive grafische Benutzeroberfläche (GUI) bereitzustellen.

## 2. Ist-Zustand (Ausgangssituation)
Der Deutsche Bundestag veröffentlicht alle Sitzungsprotokolle als Open Data ([Bundestag Open Data](https://www.bundestag.de/services/opendata)) im XML- oder Text-Format, lizenziert unter der [Datenlizenz Deutschland – Namensnennung – Version 2.0](https://www.govdata.de/dl-de/by-2-0) (`dl-de/by-2-0`) gemäß DCAT-AP.de Metadaten-Spezifikation (Lizenz-URI `http://www.dcat-ap.de/def/licenses/dl-by-de/2.0` / Lizenzcode `dl-by-de/2.0`).
* Diese Dokumente umfassen pro Wahlperiode zehntausende Seiten. 
* Eine systematische Auswertung, wer wen wie oft unterbricht oder wie sich die Stimmung zu bestimmten Themen entwickelt, ist händisch nicht leistbar.
* Bisherige Analysen stützen sich oft auf "Gefühle" oder anekdotische Evidenz, statt auf harte Daten und statistische Signifikanztests.

## 3. Soll-Zustand (Vision)
Es soll eine Softwareanwendung entwickelt werden, die den gesamten Prozess von der Datenbeschaffung bis zur Visualisierung automatisiert. Die Anwendung arbeitet rein lokal (offline-fähig nach dem Download) und verursacht **keine laufenden Kosten** für API-Aufrufe oder Server-Infrastruktur. Der Endnutzer bedient die Software über ein modernes, interaktives Web-Dashboard im Browser mit 25 strukturierten Analyse-Seiten in 6 Themenkategorien (Seiten 1.1 bis 6.4).

## 4. Funktionale Anforderungen (Muss-Kriterien)

Das System muss zwingend folgende Funktionen bereitstellen:

* **FA-10: Automatisierte Datenbeschaffung (Scraping) & Lizenz-Konformität**
  * Das System muss Plenarprotokolle mehrerer Wahlperioden automatisiert von offiziellen Schnittstellen (Bundestag Open Data) herunterladen und auf Vollständigkeit prüfen.
  * Das System muss die Lizenzvorgaben der *Datenlizenz Deutschland – Namensnennung – Version 2.0* (`dl-de/by-2-0` / DCAT-AP.de: `dl-by-de/2.0`) einhalten und den verfassungskonformen Quellenvermerk (Bereitsteller: Deutscher Bundestag; Datensatz-URI; Hinweis auf automatisierte Datenaufbereitung gem. § 3) in Dokumentation und Frontend anzeigen.
* **FA-20: Intelligentes Text-Parsing & SQLite FTS5**
  * Rohe XML-Dokumente müssen in strukturierte Entitäten zerlegt werden: Sitzung, Redner, Rede, Zwischenruf, StylometrieFeatures, SpeechComplexityResult, ReactionGapResult, LexicalResult und FramingResult.
  * Volltextsuchindizes (`reden_fts`, `zwischenrufe_fts`) müssen angelegt werden, um sekundenschnelle Begriffssuchen zu ermöglichen.
* **FA-30: Natural Language Processing (NLP & SetFit)**
  * Das System muss Zwischenrufe und Reden automatisch analysieren und Sentiments (von -1 negativ bis +1 positiv), Ton-Labels (Aggression, Sarkasmus, Humor, Neutral) sowie Adressaten zuordnen.
* **FA-40: Interaktives Dashboard (GUI & 25 Nummerierte Seiten)**
  * Das Frontend muss eine native 6-Kategorien Sidebar-Navigation (`st.navigation`) für 25 explizit nummerierte Analyseseiten bieten (Seiten 1.1–1.2, 2.1–2.4, 3.1–3.4, 4.1–4.6, 5.1–5.5, 6.1–6.4).
* **FA-50: Aggressions- & Reziprozitäts-Radar**
  * Generierung von Ranglisten für Empfänger und Störer sowie Berechnung einer Reziprozitäts-Matrix (Austeilen vs. Einstecken).
* **FA-60: Themen-Karriere & Multi-WP-Trends**
  * Freitextsuche für Schlagworte mit Zeitreihen (pro 1 000 Wörter normiert) über einzelne und mehrere Wahlperioden hinweg.
* **FA-70: Interaktions-Netzwerk & ERGM**
  * Adjazenzmatrix-Heatmap, Exponential Random Graph Models (ERGM), GraphML/GEXF-Export und Timeline-Slider für Netzwerk-Evolution.
* **FA-80: Stylometrie & Blind-Test**
  * Syntaktische Feature-Extraktion (Satzlänge, Passiv quote, TTR, Füllwortquote) und RandomForest-Klassifikation zur automatisierten Fraktionszuordnung anonymisierter Texte.
* **FA-90: Framing-Analyse (TF-IDF)**
  * TF-IDF Schlüsselbegriff-Extraktion je Fraktion, Chi-Quadrat-Unabhängigkeitstests und hypergeometrische Anreicherungstests.
* **FA-100: Lexik-Analyse**
  * Berechnung von Type-Token-Ratio (TTR), Mean Segmental TTR (MSTTR) und lexikalischer Dichte je Fraktion.
* **FA-110: Sprachkomplexität & Methodische Triangulation**
  * Berechnung und Gegenüberstellung von Wiener Sachtextformel (WSTF) und Flesch Reading Ease (FRE) inklusive Kruskal-Wallis-Tests.
* **FA-120: Eskalationsanalyse (Action-Reaction-Gap)**
  * Messung der Sentiment-Differenz zwischen Rednern und Reaktionen samt Eskalationsgeschwindigkeit.
* **FA-130: Wissenschaftliche Hypothesen-Prüfung**
  * Live-Prüfung der drei Kernthesen H1 (Oppositions-Kritik), H2 (Reizthemen-Polarisierung) und H3 (Asymmetrische Störnetzwerke).
* **FA-140: Redner-Vergleich & Fraktions-Dynamik**
  * Direktvergleich zweier Abgeordneter mit überlagernden Radar-Charts, Ton-Delta-Balken und Kosinus-Ähnlichkeit sowie Sunburst-Hierarchien.
* **FA-150: Sentiment-Modell-Validierung**
  * Evaluierung der Modellvorhersagen gegen einen Gold-Standard-Referenzdatensatz ($N=100$) mit Confusion Matrix, Präzision, Sensitivität und F1-Score.
* **FA-160: Qualitäts-, Integritäts- & Auditwerkzeuge**
  * Automatisierte DB-Integritätsprüfung (`check_db_integrity.py`) auf Sequenzlücken, Inversionen und Precomputation-Status, statistische Varianz- & Extremwert-Auditierung mit Bessel-Korrektur (`compute_standard_deviations.py`), Extraktion qualitativer Sarkasmus-Fallstudien (`extract_sarcasm_case_studies.py`) sowie Plot-Exporter (`export_dashboard_plots.py`).
* **FA-170: Console Commands & Automatisierte Setup-Skripte**
  * Bereitstellung registrierter Konsolenbefehle (`openparlament-init`, `openparlament-scrape`, `openparlament-nlp`, `openparlament-dashboard`) via `pyproject.toml` sowie Plattform-übergreifender Setup-Skripte (`setup.sh`, `setup.ps1`).
* **FA-180: Schema-Migrationen (Alembic Integration)**
  * Das System muss relationale Schema-Migrationen und Tabellen-Erweiterungen versioniert über Alembic (`alembic/`) verwalten und nachvollziehbar ausführen können.

## 5. Nicht-funktionale Anforderungen

* **NFA-10: Kostenfreiheit (Zero-Cost-Policy)**
  * Das Projekt darf ausschließlich kostenlose Open-Source-Software und -Modelle nutzen. Cloud-APIs (wie OpenAI) sind strikt ausgeschlossen.
* **NFA-20: Performance und Usability**
  * Dank vorgeschalteter Offline-Precomputations (`precompute_h3`, `precompute_framing`, `precompute_complexity`) muss das Laden aller Dashboard-Seiten unter 2 Sekunden erfolgen.
* **NFA-30: Datenhaltung & Zuverlässigkeit**
  * Alle Daten müssen in einer lokalen relationalen SQLite-Datenbank mit FTS5-Unterstützung gespeichert werden.
* **NFA-40: Wartbarkeit & Testabdeckung**
  * Der Code folgt streng objektorientierten Prinzipien (OOP). Eine automatische Test-Suite mit 530+ Unit- und Integrationstests stellt die Codequalität sicher.

## 6. Lieferumfang
1. Python-Backend (Scraper, Parser, NLP-Engines, Analytics-Module).
2. Initialisiertes SQLite-Datenbankschema mit FTS5-Volltextindizes.
3. Interactive Streamlit-Frontend (`src/app.py`) mit 25 Analyseseiten.
4. Precomputation-Pipelines, CLI-Integrity-Checker, statistische Audit-Tools & Sarkasmus-Fallstudien-Extraktor.
5. Vollständige Test-Suite (530+ Tests in 5 partitionierten Ordnern).
6. Vollständige Dokumentation (README, Lastenheft, Pflichtenheft, Project Tracker & Architecture).
