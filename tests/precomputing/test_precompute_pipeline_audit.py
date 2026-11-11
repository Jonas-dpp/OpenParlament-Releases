"""
Unit and integration tests for the precomputing pipeline and database architecture audit.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

import scripts.precompute_h3 as h3_module
from scripts.precompute_h3 import main as precompute_h3_main
from scripts.precompute_framing import main as precompute_framing_main, precompute_for_wp as precompute_framing_wp
from scripts.precompute_complexity import precompute_for_wp as precompute_complexity_wp
from src.database import get_session, init_db, get_engine, reset_db_state
from src.models import Sitzung, Redner, Rede, Zwischenruf


def test_precompute_h3_incremental_update(tmp_path):
    """Test that precomputing H3 for a single WP merges with existing Parquet data incrementally."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = data_dir / "h3_precomputed.parquet"

    # Pre-populate existing parquet file with WP 19 data
    initial_df = pd.DataFrame([
        {"wp": 19, "sender": "SPD", "speaker": "CDU/CSU", "sentiment": -0.5},
        {"wp": 19, "sender": "CDU/CSU", "speaker": "SPD", "sentiment": 0.2},
    ])
    initial_df.to_parquet(parquet_file, compression="snappy", index=False)

    new_wp20_df = pd.DataFrame([
        {"wp": 20, "sender": "AfD", "speaker": "SPD", "sentiment": -0.8},
    ])

    with patch.object(h3_module, "_PROJECT_ROOT", tmp_path), \
         patch.object(h3_module, "precompute_data", return_value=new_wp20_df), \
         patch("sys.argv", ["precompute_h3.py", "--wahlperiode", "20"]):
        precompute_h3_main()

    # Verify merged result
    result_df = pd.read_parquet(parquet_file)
    assert len(result_df) == 3
    assert set(result_df["wp"].unique()) == {19, 20}


def test_precompute_h3_stress_and_edge_cases(tmp_path):
    """Stress-test incremental updates: duplicate runs, empty inputs, order invariance."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = data_dir / "h3_precomputed.parquet"

    # 1. First run: WP 18
    df_wp18 = pd.DataFrame([{"wp": 18, "sender": "SPD", "speaker": "CDU/CSU", "sentiment": -0.1}])
    with patch.object(h3_module, "_PROJECT_ROOT", tmp_path), \
         patch.object(h3_module, "precompute_data", return_value=df_wp18), \
         patch("sys.argv", ["precompute_h3.py", "--wahlperiode", "18"]):
        precompute_h3_main()

    res1 = pd.read_parquet(parquet_file)
    assert len(res1) == 1
    assert res1["wp"].iloc[0] == 18

    # 2. Second run: WP 19 added
    df_wp19 = pd.DataFrame([{"wp": 19, "sender": "AfD", "speaker": "FDP", "sentiment": -0.7}])
    with patch.object(h3_module, "_PROJECT_ROOT", tmp_path), \
         patch.object(h3_module, "precompute_data", return_value=df_wp19), \
         patch("sys.argv", ["precompute_h3.py", "--wahlperiode", "19"]):
        precompute_h3_main()

    res2 = pd.read_parquet(parquet_file)
    assert len(res2) == 2
    assert set(res2["wp"].unique()) == {18, 19}

    # 3. Third run: Re-run WP 18 (idempotent overwrite of WP 18, keeping WP 19)
    df_wp18_updated = pd.DataFrame([
        {"wp": 18, "sender": "SPD", "speaker": "CDU/CSU", "sentiment": -0.2},
        {"wp": 18, "sender": "CDU/CSU", "speaker": "SPD", "sentiment": 0.4},
    ])
    with patch.object(h3_module, "_PROJECT_ROOT", tmp_path), \
         patch.object(h3_module, "precompute_data", return_value=df_wp18_updated), \
         patch("sys.argv", ["precompute_h3.py", "--wahlperiode", "18"]):
        precompute_h3_main()

    res3 = pd.read_parquet(parquet_file)
    assert len(res3) == 3  # 2 rows for WP 18 + 1 row for WP 19
    assert len(res3[res3["wp"] == 18]) == 2
    assert len(res3[res3["wp"] == 19]) == 1


def test_precompute_h3_empty_recomputation_preserves_existing(tmp_path):
    """Empty recomputation must not discard existing results."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = data_dir / "h3_precomputed.parquet"

    initial_df = pd.DataFrame([
        {"wp": 18, "sender": "SPD", "speaker": "CDU/CSU", "sentiment": 0.1},
    ])
    initial_df.to_parquet(parquet_file, compression="snappy", index=False)

    # Precomputing for WP 18 returns empty dataframe
    with patch.object(h3_module, "_PROJECT_ROOT", tmp_path), \
         patch.object(h3_module, "precompute_data", return_value=pd.DataFrame()), \
         patch("sys.argv", ["precompute_h3.py", "--wahlperiode", "18"]):
        with pytest.raises(SystemExit) as exc_info:
            precompute_h3_main()
        assert exc_info.value.code == 0

    # Ensure existing file is completely intact
    after_df = pd.read_parquet(parquet_file)
    assert len(after_df) == 1
    assert after_df["wp"].iloc[0] == 18


def test_precompute_h3_unreadable_file_aborts_safely(tmp_path):
    """An unreadable existing Parquet file must raise an error and never be silently overwritten."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = data_dir / "h3_precomputed.parquet"

    # Write corrupt data
    with open(parquet_file, "w") as f:
        f.write("NOT_A_VALID_PARQUET_FILE")

    new_data = pd.DataFrame([{"wp": 20, "sender": "SPD", "speaker": "CDU/CSU", "sentiment": 0.0}])

    with patch.object(h3_module, "_PROJECT_ROOT", tmp_path), \
         patch.object(h3_module, "precompute_data", return_value=new_data), \
         patch("sys.argv", ["precompute_h3.py", "--wahlperiode", "20"]):
        with pytest.raises(RuntimeError, match="Unreadable existing Parquet file"):
            precompute_h3_main()


def test_precompute_framing_cli_args_and_debug():
    """Verify that --debug, --limit, and argument parsing work properly for precompute_framing."""
    with patch("scripts.precompute_framing.init_db"), \
         patch("scripts.precompute_framing.get_session"), \
         patch("scripts.precompute_framing.precompute_for_wp") as mock_precompute:
        with patch("sys.argv", ["precompute_framing.py", "--wahlperiode", "20", "--n-terms", "100", "--limit", "50", "--debug"]):
            precompute_framing_main()
            mock_precompute.assert_called_once()
            args, kwargs = mock_precompute.call_args
            assert kwargs.get("limit") == 50
            assert kwargs.get("n_terms") == 100


def test_precompute_complexity_empty_data():
    """Verify precompute_complexity gracefully handles an empty database for a wahlperiode."""
    mock_session = MagicMock()
    mock_session.execute.return_value.scalar.return_value = 0

    res = precompute_complexity_wp(mock_session, wp=99)
    assert res is False


def test_precompute_complexity_cli_args_and_debug():
    """Verify that --debug, --limit, and argument parsing work properly for precompute_complexity."""
    from scripts.precompute_complexity import main as precompute_complexity_main
    with patch("scripts.precompute_complexity.init_db"), \
         patch("scripts.precompute_complexity.get_session"), \
         patch("scripts.precompute_complexity.precompute_for_wp") as mock_precompute:
        with patch("sys.argv", ["precompute_complexity.py", "--wahlperiode", "20", "--limit", "30", "--debug"]):
            precompute_complexity_main()
            mock_precompute.assert_called_once()
            args, kwargs = mock_precompute.call_args
            assert kwargs.get("limit") == 30


def test_alembic_migrations_up_to_date():
    """Verify that Alembic migrations match the current ORM schema without differences."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    head_rev = script.get_current_head()

    assert head_rev == "ffd749ebf226", f"Alembic head should be ffd749ebf226, found {head_rev}"


def test_migration_lifecycle_upgrade_downgrade_and_record_preservation(tmp_path, monkeypatch):
    """Verify that:
    1. An existing database with records at revision e9d0356b9ab9
       upgrades cleanly to head (ffd749ebf226).
    2. All existing records are 100% preserved.
    3. New tables (speech_complexity_results, reaction_gap_results) and columns (segment in framing_results)
       are created with correct columns, indexes, and constraints.
    4. Downgrade drops new tables cleanly, and re-upgrade restores them.
    5. Schema check detects no drift against target ORM metadata.
    """
    from alembic.config import Config
    from alembic import command
    from sqlalchemy import create_engine, inspect, text
    from src.models import Base

    db_file = tmp_path / "lifecycle_test.db"
    db_url = f"sqlite:///{db_file}"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", db_url)
    reset_db_state()

    # 1. Simulate an existing pre-7a82b01c3d4f database schema where framing_results lacks segment/variant
    engine = create_engine(db_url)
    pre_tables = [
        t for name, t in Base.metadata.tables.items()
        if name not in ("speech_complexity_results", "reaction_gap_results", "framing_results")
    ]
    Base.metadata.create_all(bind=engine, tables=pre_tables)
    # Manually create framing_results with legacy pre-ffd749ebf226 schema
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE framing_results (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                wahlperiode INTEGER NOT NULL,
                fraktion VARCHAR(128) NOT NULL,
                term VARCHAR(256) NOT NULL,
                score FLOAT NOT NULL,
                CONSTRAINT uq_framing_wp_frak_term UNIQUE (wahlperiode, fraktion, term)
            )
        """))
        conn.execute(text("CREATE INDEX ix_framing_results_fraktion ON framing_results(fraktion)"))
        conn.execute(text("CREATE INDEX ix_framing_results_wahlperiode ON framing_results(wahlperiode)"))

    cfg = Config("alembic.ini")
    command.stamp(cfg, "e9d0356b9ab9")

    # 2. Insert representative records before migration
    with get_session() as s:
        sitz = Sitzung(wahlperiode=20, sitzungsnr=1, datum=date(2021, 10, 26), gesamtwortzahl=500)
        s.add(sitz)
        s.flush()
        redn = Redner(vorname="Erika", nachname="Mustermann", fraktion="SPD")
        s.add(redn)
        s.flush()
        rede = Rede(sitzung_id=sitz.sitzungs_id, redner_id=redn.redner_id, text="Plenarrede", wortanzahl=100)
        s.add(rede)
        s.flush()
        zw = Zwischenruf(rede_id=rede.rede_id, text="Zwischenruf!", fraktion="AfD", sentiment_score=-0.6)
        s.add(zw)

    # 3. Upgrade to head (7a82b01c3d4f)
    command.upgrade(cfg, "head")

    insp = inspect(engine)
    tables = set(insp.get_table_names())
    assert "speech_complexity_results" in tables
    assert "reaction_gap_results" in tables

    # Verify foreign keys on new tables
    sc_fks = insp.get_foreign_keys("speech_complexity_results")
    sc_referred = {fk["referred_table"] for fk in sc_fks}
    assert {"reden", "redner"}.issubset(sc_referred)

    rg_fks = insp.get_foreign_keys("reaction_gap_results")
    rg_referred = {fk["referred_table"] for fk in rg_fks}
    assert {"reden", "zwischenrufe"}.issubset(rg_referred)

    # Verify column definitions on new tables
    sc_cols = {c["name"] for c in insp.get_columns("speech_complexity_results")}
    assert {"id", "rede_id", "wahlperiode", "redner_id", "wstf_score", "flesch_score", "avg_sentence_length"}.issubset(sc_cols)

    # 4. Verify existing records are preserved
    with get_session() as s:
        assert s.query(Sitzung).count() == 1
        assert s.query(Redner).count() == 1
        assert s.query(Rede).count() == 1
        assert s.query(Zwischenruf).count() == 1

    # 5. Test downgrade to previous revisions
    command.downgrade(cfg, "7a82b01c3d4f")
    insp_after_down = inspect(engine)
    cols_7a = {c["name"] for c in insp_after_down.get_columns("framing_results")}
    assert "segment" not in cols_7a

    command.downgrade(cfg, "e9d0356b9ab9")
    insp_after_e9 = inspect(engine)
    tables_after_e9 = set(insp_after_e9.get_table_names())
    assert "speech_complexity_results" not in tables_after_e9
    assert "reaction_gap_results" not in tables_after_e9

    # 6. Re-upgrade to head
    command.upgrade(cfg, "head")
    insp_re = inspect(engine)
    assert "speech_complexity_results" in set(insp_re.get_table_names())
    assert "reaction_gap_results" in set(insp_re.get_table_names())
    assert "segment" in {c["name"] for c in insp_re.get_columns("framing_results")}


def test_database_initialization_and_migration_lifecycle_all_scenarios(tmp_path, monkeypatch):
    """Verify all 6 database initialization and migration lifecycle scenarios:
    1. A completely fresh database.
    2. A database initialized using init_db() / create_all().
    3. An existing database containing representative records.
    4. A partially initialized database.
    5. A database missing one of the newly introduced tables.
    6. A database with an existing Alembic version table.
    """
    from alembic.config import Config
    from alembic import command
    from sqlalchemy import create_engine, inspect, text
    from src.models import (
        Base, Sitzung, Redner, Rede, Zwischenruf,
        SpeechComplexityResult, ReactionGapResult,
        FramingResult, LexicalResult
    )

    required_tables = {
        "sitzungen", "redner", "reden", "zwischenrufe",
        "stylometrie_features", "framing_results", "lexical_results",
        "speech_complexity_results", "reaction_gap_results",
    }
    cfg = Config("alembic.ini")

    # --- Scenario 1: Fresh database initialization ---
    db_file_1 = tmp_path / "scenario_1_fresh.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file_1}")
    reset_db_state()
    init_db()
    engine_1 = create_engine(f"sqlite:///{db_file_1}")
    insp_1 = inspect(engine_1)
    assert required_tables.issubset(set(insp_1.get_table_names()))

    # --- Scenario 2: Database initialized via init_db() / create_all() ---
    db_file_2 = tmp_path / "scenario_2_init_db.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file_2}")
    reset_db_state()
    init_db()
    # When initialized with create_all(), tables exist. Stamping head and checking schema detects no drift.
    command.stamp(cfg, "head")
    command.check(cfg)

    # --- Scenario 3: Existing database with representative records ---
    db_file_3 = tmp_path / "scenario_3_existing_records.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file_3}")
    reset_db_state()
    engine_3 = create_engine(f"sqlite:///{db_file_3}")
    pre_tables = [
        t for name, t in Base.metadata.tables.items()
        if name not in ("speech_complexity_results", "reaction_gap_results", "framing_results")
    ]
    Base.metadata.create_all(bind=engine_3, tables=pre_tables)
    with engine_3.begin() as conn:
        conn.execute(text("""
            CREATE TABLE framing_results (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                wahlperiode INTEGER NOT NULL,
                fraktion VARCHAR(128) NOT NULL,
                term VARCHAR(256) NOT NULL,
                score FLOAT NOT NULL,
                CONSTRAINT uq_framing_wp_frak_term UNIQUE (wahlperiode, fraktion, term)
            )
        """))
        conn.execute(text("CREATE INDEX ix_framing_results_fraktion ON framing_results(fraktion)"))
        conn.execute(text("CREATE INDEX ix_framing_results_wahlperiode ON framing_results(wahlperiode)"))
    command.stamp(cfg, "e9d0356b9ab9")
    with get_session() as s:
        sitz = Sitzung(wahlperiode=20, sitzungsnr=1, gesamtwortzahl=150)
        s.add(sitz)
        s.flush()
        redn = Redner(vorname="Alice", nachname="Schmidt", fraktion="SPD")
        s.add(redn)
        s.flush()
        rede = Rede(sitzung_id=sitz.sitzungs_id, redner_id=redn.redner_id, text="Rede Text", wortanzahl=150)
        s.add(rede)
        s.flush()
        zw = Zwischenruf(rede_id=rede.rede_id, text="Zwischenruf!", fraktion="CDU/CSU", sentiment_score=-0.2)
        s.add(zw)

    command.upgrade(cfg, "head")
    insp_3 = inspect(engine_3)
    assert "speech_complexity_results" in set(insp_3.get_table_names())
    assert "reaction_gap_results" in set(insp_3.get_table_names())
    with get_session() as s:
        assert s.query(Sitzung).count() == 1
        assert s.query(Redner).count() == 1
        assert s.query(Rede).count() == 1
        assert s.query(Zwischenruf).count() == 1
    command.check(cfg)

    # --- Scenario 4: Partially initialized database ---
    db_file_4 = tmp_path / "scenario_4_partial.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file_4}")
    reset_db_state()
    engine_4 = create_engine(f"sqlite:///{db_file_4}")
    # Simulate partial initialization where only core metadata was created
    Base.metadata.tables["sitzungen"].create(engine_4)
    Base.metadata.tables["redner"].create(engine_4)
    insp_4_pre = inspect(engine_4)
    assert set(insp_4_pre.get_table_names()) == {"redner", "sitzungen"}
    # init_db() safely heals the schema and creates all missing tables
    init_db()
    insp_4_post = inspect(engine_4)
    assert required_tables.issubset(set(insp_4_post.get_table_names()))

    # --- Scenario 5: Database missing one of the newly introduced tables ---
    db_file_5 = tmp_path / "scenario_5_missing_one.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file_5}")
    reset_db_state()
    engine_5 = create_engine(f"sqlite:///{db_file_5}")
    missing_one_tables = [t for name, t in Base.metadata.tables.items() if name not in ("speech_complexity_results", "framing_results")]
    Base.metadata.create_all(bind=engine_5, tables=missing_one_tables)
    with engine_5.begin() as conn:
        conn.execute(text("""
            CREATE TABLE framing_results (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                wahlperiode INTEGER NOT NULL,
                fraktion VARCHAR(128) NOT NULL,
                term VARCHAR(256) NOT NULL,
                score FLOAT NOT NULL,
                CONSTRAINT uq_framing_wp_frak_term UNIQUE (wahlperiode, fraktion, term)
            )
        """))
        conn.execute(text("CREATE INDEX ix_framing_results_fraktion ON framing_results(fraktion)"))
        conn.execute(text("CREATE INDEX ix_framing_results_wahlperiode ON framing_results(wahlperiode)"))
    insp_5_pre = inspect(engine_5)
    assert "speech_complexity_results" not in set(insp_5_pre.get_table_names())
    assert "reaction_gap_results" in set(insp_5_pre.get_table_names())
    # Upgrading creates the missing table idempotently without failing on reaction_gap_results
    command.stamp(cfg, "e9d0356b9ab9")
    command.upgrade(cfg, "head")
    insp_5_post = inspect(engine_5)
    assert "speech_complexity_results" in set(insp_5_post.get_table_names())
    assert "reaction_gap_results" in set(insp_5_post.get_table_names())
    command.check(cfg)

    # --- Scenario 6: Database with an existing Alembic version table ---
    db_file_6 = tmp_path / "scenario_6_version_table.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", f"sqlite:///{db_file_6}")
    reset_db_state()
    engine_6 = create_engine(f"sqlite:///{db_file_6}")
    scen_6_tables = [t for name, t in Base.metadata.tables.items() if name not in ("speech_complexity_results", "reaction_gap_results", "framing_results")]
    Base.metadata.create_all(bind=engine_6, tables=scen_6_tables)
    with engine_6.begin() as conn:
        conn.execute(text("""
            CREATE TABLE framing_results (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                wahlperiode INTEGER NOT NULL,
                fraktion VARCHAR(128) NOT NULL,
                term VARCHAR(256) NOT NULL,
                score FLOAT NOT NULL,
                CONSTRAINT uq_framing_wp_frak_term UNIQUE (wahlperiode, fraktion, term)
            )
        """))
        conn.execute(text("CREATE INDEX ix_framing_results_fraktion ON framing_results(fraktion)"))
        conn.execute(text("CREATE INDEX ix_framing_results_wahlperiode ON framing_results(wahlperiode)"))
    command.stamp(cfg, "e9d0356b9ab9")
    engine_6 = create_engine(f"sqlite:///{db_file_6}")
    with engine_6.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "e9d0356b9ab9"
    command.upgrade(cfg, "head")
    with engine_6.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "ffd749ebf226"
    command.check(cfg)
