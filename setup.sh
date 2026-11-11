#!/bin/bash
echo "Starte Setup für OpenParlament..."

# 1. Abhängigkeiten installieren & als Paket registrieren (inkl. CLI-Console-Commands)
pip install -r requirements.txt
pip install -e .

# 2. spaCy Sprachmodell herunterladen
python -m spacy download de_core_news_sm

# 3. Datenbank initialisieren/patchen
python scripts/db_init.py
python scripts/db_patch.py

# 4. Falls Daten existieren, diese XML-Dateien in die Datenbank importieren
python scripts/import_xmls.py
python scripts/migrate_text_processing.py

# 5. Datenbank inspizieren und Integrität verifizieren
python scripts/db_inspect.py
python scripts/check_db_integrity.py

echo "Setup abgeschlossen! Du kannst das Dashboard nun mit 'streamlit run src/app.py' starten."