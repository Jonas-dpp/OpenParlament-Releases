Write-Host "Starte Setup für OpenParlament..."

pip install -r requirements.txt
pip install -e .

python -m spacy download de_core_news_sm

python scripts/db_init.py
python scripts/db_patch.py

python scripts/import_xmls.py
python scripts/migrate_text_processing.py

python scripts/db_inspect.py
python scripts/check_db_integrity.py

Write-Host "Setup abgeschlossen!"
Write-Host "Du kannst das Dashboard nun mit 'streamlit run src/app.py' starten." 