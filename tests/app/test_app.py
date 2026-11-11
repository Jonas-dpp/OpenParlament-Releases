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
Frontend tests for the OpenParlament Streamlit dashboard.

Uses ``streamlit.testing.v1.AppTest`` to exercise the app's render paths
with an empty, per-test SQLite database in a temporary directory so that
runtime errors (e.g. a missing import, a NameError, or a broken render
function) are caught early in CI.
"""

from __future__ import annotations

import pytest
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

APP_PATH = str(Path(__file__).resolve().parents[2] / "src" / "app.py")

# Every page key that exists in the dashboard's PAGES map (see src/app.py).
ALL_PAGE_KEYS = [
    "startseite",
    "aggressions-radar",
    "themen-trend",
    "interaktions-netzwerk",
    "ton-analyse",
    "adressaten-analyse",
    "scraping-monitor",
    "wahlperioden-vergleich",
    "top-analyse",
    "reaktions-analyse",
    "redezeit-gerechtigkeit",
    "debattenklima",
    "redner-profil",
]


def _reset_db_cache() -> None:
    """Reset the cached SQLAlchemy engine and session factory via the public API."""
    from src.database import reset_db_state
    reset_db_state()


def _make_app(page_key: str):
    """Return a configured ``AppTest`` instance for *page_key*."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.query_params["page"] = page_key
    return at


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    """Each test gets its own isolated SQLite database."""
    db_url = f"sqlite:///{tmp_path}/test_app.db"
    monkeypatch.setenv("OPENPARLAMENT_DB_URL", db_url)

    _reset_db_cache()

    # Also clear Streamlit's cache so helpers re-query the fresh DB.
    try:
        import streamlit as st
        st.cache_data.clear()
    except (ImportError, AttributeError):
        # If Streamlit is missing or the cache API is unavailable, skip cache clear.
        pass

    yield

    _reset_db_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAppLoadsWithoutCrash:
    """The app must not raise an exception on any page with an empty DB."""

    @pytest.mark.parametrize("page_key", ALL_PAGE_KEYS)
    def test_page_renders_without_exception(self, page_key):
        """Each dashboard page should load without raising a Python exception."""
        at = _make_app(page_key)
        at.run()
        assert not at.exception, (
            f"Page '{page_key}' raised an exception: {at.exception}"
        )


class TestStartseiteContent:
    """Basic smoke-test for the landing page."""

    def test_startseite_has_header(self):
        at = _make_app("startseite")
        at.run()
        assert not at.exception, f"Startseite raised: {at.exception}"
        # The page should contain at least one visible text element.
        all_text_parts = []
        for collection in (at.title, at.header, at.subheader, at.markdown):
            for el in collection:
                if hasattr(el, "value") and el.value:
                    all_text_parts.append(el.value)
        assert len(all_text_parts) > 0, "Startseite rendered no visible text elements"

    def test_startseite_navigation_buttons_present(self):
        at = _make_app("startseite")
        at.run()
        assert not at.exception
        # The landing page should show at least one navigation button.
        assert len(at.button) >= 1, "Expected at least one navigation button on Startseite"

    def test_onboarding_wizard_opposition_button_click(self):
        """Clicking 'wiz_path_opp' should update session state without StreamlitWidgetAlreadyInstantiatedError."""
        at = _make_app("startseite")
        at.run()
        assert not at.exception

        opp_btn = next((b for b in at.button if b.key == "wiz_path_opp"), None)
        assert opp_btn is not None, "Button 'wiz_path_opp' not found on startseite"

        opp_btn.click().run()
        assert not at.exception, f"Unexpected exception after clicking 'wiz_path_opp': {at.exception}"
        assert at.session_state["selected_wp"] == 20
        assert at.session_state["active_hypothesis"] == "Hypothese 1: Die Oppositions-Hypothese (Makro-Ebene)"


class TestImportsAreComplete:
    """Regression test: ensure Rede (and other models) are imported in app source."""

    def test_rede_import_in_app_source(self):
        """src/app.py must explicitly import Rede from src.models."""
        app_source = Path(APP_PATH).read_text(encoding="utf-8")
        assert "Rede" in app_source, (
            "src/app.py does not import 'Rede' from src.models. "
            "This would cause a NameError at runtime."
        )
        # Check the models import line contains Rede.
        import re
        matches = re.findall(r"from src\.models import[^\n]+", app_source)
        assert any("Rede" in m for m in matches), (
            "The 'from src.models import ...' line does not include Rede."
        )

    def test_no_use_container_width_in_app(self):
        """src/app.py must not use the deprecated use_container_width parameter."""
        app_source = Path(APP_PATH).read_text(encoding="utf-8")
        assert "use_container_width" not in app_source, (
            "Deprecated 'use_container_width' parameter found in src/app.py. "
            "Replace it with width='stretch' or width='content'."
        )


class TestCachedAnalyticsHelpers:
    """Ensure heavy analytics helpers are cached and parameterized."""

    def test_cached_helper_functions_exist(self):
        app_source = Path(APP_PATH).read_text(encoding="utf-8")
        for fn_name in (
            "_cached_keyword_trend",
            "_cached_multi_wp_keyword_trend",
            "_cached_keyword_peak_by_wp",
            "_cached_most_polarizing_keywords",
            "_cached_keyword_aggression_correlation",
            "_cached_klima_per_sitzung",
            "_cached_hottest_sessions",
        ):
            assert f"def {fn_name}(" in app_source, f"Missing cached helper {fn_name}"

    def test_cached_helpers_use_cache_data_decorator(self):
        import re
        app_source = Path(APP_PATH).read_text(encoding="utf-8")
        assert re.search(r"@st\.cache_data\(ttl=600\)\s*\ndef _cached_keyword_trend", app_source)
        assert re.search(r"@st\.cache_data\(ttl=600\)\s*\ndef _cached_klima_per_sitzung", app_source)


class TestTOPAnalysePage:
    """Tests for the top-analyse dashboard page, including the NLP-score fallback path."""

    def _seed_top_data_no_nlp(self) -> None:
        """Seed TOPs with un-scored Zwischenrufe (sentiment_score=None).

        The autouse ``_isolated_db`` fixture has already set ``OPENPARLAMENT_DB_URL``
        and reset the engine cache, so we just init tables and insert rows.
        At least 3 reden per TOP are inserted so the default ``min_reden=3``
        slider does not filter them out.
        """
        from src.database import init_db, get_session
        from src.models import Sitzung, Redner, Rede, Zwischenruf
        from datetime import date

        init_db()
        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=1,
                              datum=date(2022, 1, 1), gesamtwortzahl=500)
            redner = Redner(vorname="Anna", nachname="Test", fraktion="SPD")
            reden = [
                Rede(
                    sitzung=sitzung, redner=redner, text=f"Rede {i}",
                    tagesordnungspunkt="TOP 1 – Klimaschutz", wortanzahl=5,
                )
                for i in range(4)
            ]
            # Un-scored interjections on each rede (NLP pipeline not yet run)
            zwischenrufe = [
                Zwischenruf(rede=r, text="Widerspruch", fraktion="AfD",
                            sentiment_score=None, kategorie="Widerspruch")
                for r in reden
            ]
            s.add_all([sitzung, redner] + reden + zwischenrufe)

    def _make_render_fn(self):
        """Return a minimal Streamlit function that exercises render_top_analyse's critical paths."""

        def _render():
            import streamlit as st
            import plotly.express as px
            from src.analytics import TOPAnalyse
            from src.database import get_session

            top_n = st.slider("Top N Tagesordnungspunkte", 5, 30, 15, key="top_n_top")
            min_reden = st.slider("Mindestanzahl Reden", 1, 20, 3, key="min_reden_top")

            with get_session() as session:
                ta = TOPAnalyse(session)
                df = ta.aggression_by_top(n=top_n, wahlperiode=20, min_reden=min_reden)
                df_kat = ta.kategorie_by_top(n=top_n, wahlperiode=20, min_reden=min_reden)

            if df.empty:
                st.info(
                    "Keine Tagesordnungspunkt-Daten vorhanden. "
                    "Bitte zuerst Protokolle mit dem Scraper importieren."
                )
            else:
                has_nlp_scores = df["avg_aggression"].notna().any()
                if has_nlp_scores:
                    fig = px.bar(
                        df, x="avg_aggression", y="tagesordnungspunkt", orientation="h",
                        color="anteil_negativ_pct", color_continuous_scale="Reds",
                        title=f"Top {top_n} Reizthemen nach Aggressions-Score",
                    )
                else:
                    st.info(
                        "ℹ️ Keine Sentiment-Scores für Zwischenrufe vorhanden. "
                        "Bitte die NLP-Pipeline ausführen."
                    )
                    fig = px.bar(
                        df.sort_values("gesamt_zwischenrufe", ascending=False),
                        x="gesamt_zwischenrufe", y="tagesordnungspunkt", orientation="h",
                        title=f"Top {top_n} Tagesordnungspunkte nach Anzahl Zwischenrufe",
                    )
                fig.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig)

            if not df_kat.empty:
                st.subheader("Kategorie-Verteilung")
                st.plotly_chart(px.bar(
                    df_kat, x="tagesordnungspunkt", y="anzahl", color="kategorie",
                    barmode="stack", title="Reaktions-Kategorien nach TOP",
                ))

        return _render

    def test_top_analyse_fallback_chart_when_no_nlp_scores(self):
        """Fallback chart path must render without exception when TOPs exist but are un-scored."""
        from streamlit.testing.v1 import AppTest

        self._seed_top_data_no_nlp()

        at = AppTest.from_function(self._make_render_fn(), default_timeout=30)
        at.run()
        assert not at.exception, (
            f"render_top_analyse raised an exception with un-scored data: {at.exception}"
        )
        info_texts = [el.value for el in at.info]
        assert any("NLP" in t for t in info_texts), (
            "Expected an info banner mentioning NLP when sentiment scores are absent; "
            f"got info messages: {info_texts}"
        )

    def test_top_analyse_with_nlp_scores_renders_aggression_chart(self):
        """Aggression-score bar chart path must render without exception."""
        from streamlit.testing.v1 import AppTest
        from src.database import init_db, get_session
        from src.models import Sitzung, Redner, Rede, Zwischenruf
        from datetime import date

        init_db()
        with get_session() as s:
            sitzung = Sitzung(wahlperiode=20, sitzungsnr=2,
                              datum=date(2022, 2, 1), gesamtwortzahl=200)
            redner = Redner(vorname="Hans", nachname="Scored", fraktion="CDU/CSU")
            reden = [
                Rede(
                    sitzung=sitzung, redner=redner, text=f"Rede {i}",
                    tagesordnungspunkt="TOP 2 – Wirtschaft", wortanzahl=4,
                )
                for i in range(4)
            ]
            zwischenrufe = [
                Zwischenruf(rede=r, text="Widerspruch", fraktion="AfD",
                            sentiment_score=-0.8, kategorie="Widerspruch")
                for r in reden
            ]
            s.add_all([sitzung, redner] + reden + zwischenrufe)

        at = AppTest.from_function(self._make_render_fn(), default_timeout=30)
        at.run()
        assert not at.exception, (
            f"render_top_analyse raised an exception with scored data: {at.exception}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Deep Dashboard Coverage (Happy Paths)
# ─────────────────────────────────────────────────────────────────────────────
import pytest

@pytest.fixture
def analytics_dataset(_isolated_db):
    """
    Creates a rich, deterministic dataset.
    Explicitly clears tables before seeding to prevent UNIQUE constraint violations.
    """
    from datetime import date
    from src.database import get_session, init_db
    from src.models import Sitzung, Redner, Rede, Zwischenruf
    
    init_db()  # Ensure tables exist
    with get_session() as s:
        # 1. Clean current tables to ensure no primary key collisions
        # We delete in reverse order of foreign key dependency
        s.query(Zwischenruf).delete()
        s.query(Rede).delete()
        s.query(Redner).delete()
        s.query(Sitzung).delete()
        s.commit()
        
        # 2. Seed fresh data (3 Sitzungen for windowing logic)
        dates = [date(2023, 1, 10), date(2023, 2, 10), date(2023, 3, 10)]
        redner_list = [
            Redner(vorname="Alice", nachname="A", fraktion="SPD"),
            Redner(vorname="Bob", nachname="B", fraktion="CDU/CSU")
        ]
        s.add_all(redner_list)
        s.commit()
        
        for i, d in enumerate(dates):
            sz = Sitzung(wahlperiode=20, sitzungsnr=i+1, datum=d, gesamtwortzahl=500)
            s.add(sz)
            s.commit() # Flush to get ID
            
            rd = Rede(
                sitzung=sz,
                redner=redner_list[i % 2],
                text="Sample speech.",
                wortanzahl=100,
            )

            zr = Zwischenruf(
                rede=rd,
                text="Widerspruch",
                fraktion="CDU/CSU",
                sentiment_score=-0.8,
                ton_label="Aggression",
                kategorie="Widerspruch",
            )
            
            s.add_all([rd, zr])
            
            
        s.commit()


class TestDashboardHappyPaths:
    """Diagnostic tests that print DataFrame shapes to debug filtering logic."""

    def test_render_ton_analyse_diagnostics(self, analytics_dataset):
        from streamlit.testing.v1 import AppTest
        
        def _render():
            from src.app import render_ton_analyse
            
            # 1. Manually trigger the data retrieval logic used in app.py
            # This mimics what render_ton_analyse does internally
            from src.database import get_session
            from src.analytics import TonAnalyse
            with get_session() as s:
                ta = TonAnalyse(s)
                df = ta.ton_by_fraktion(wahlperiode=20)
                print(f"\n[DIAGNOSTIC] TonAnalyse raw df shape: {df.shape}")
            
            render_ton_analyse(selected_wp=20, datum_von=None, datum_bis=None)
            
        at = AppTest.from_function(_render)
        at.run(timeout=15)
        
        assert not at.exception
        # Only assert chart existence if the diagnostic print shows df.shape > (0,0)


class TestAnalyticsLayer:
    """Verify that analytics classes handle the fixture data correctly."""

    def test_ton_analyse_returns_data(self, analytics_dataset):
        from src.analytics import TonAnalyse
        from src.database import get_session
        
        with get_session() as s:
            ta = TonAnalyse(s)
            # Use the same parameters as the UI
            df = ta.ton_by_fraktion(wahlperiode=20)
            
            # Print for debugging if it fails
            if df.empty:
                print("\n[DEBUG] TonAnalyse returned empty DataFrame!")
            
            assert not df.empty, "TonAnalyse should return data for the fixture"

    def test_aggressions_radar_returns_data(self, analytics_dataset):
        from src.analytics import AggressionsIndex
        from src.database import get_session
        
        with get_session() as s:
            ai = AggressionsIndex(s)
            df = ai.top_targets(wahlperiode=20)
            assert not df.empty, "AggressionsIndex should return target data"


class TestSidebarFiltering:
    """Verifies that global sidebar settings correctly manipulate raw data before rendering."""

    def test_apply_party_filter_comprehensive(self):
        """Comprehensive unit tests for _apply_party_filter fast-path unique-value mapping."""
        import src.app as app
        import pandas as pd
        import numpy as np

        # Test 1: Empty DataFrame
        df_empty = pd.DataFrame()
        assert app._apply_party_filter(df_empty).empty

        # Test 2: Missing target column
        df_no_col = pd.DataFrame({"other": [1, 2, 3]})
        assert app._apply_party_filter(df_no_col, col="fraktion").equals(df_no_col)

        # Test 3: Unfiltered (fraktion_allow_list is None), raw variants get canonicalized
        df_raw = pd.DataFrame({"fraktion": ["SPDSPD", "CDU", "GRÜNE", "LINKE"], "val": [1, 2, 3, 4]})
        app.fraktion_allow_list = None
        res_unfiltered = app._apply_party_filter(df_raw, col="fraktion")
        assert list(res_unfiltered["fraktion"]) == ["SPD", "CDU/CSU", "BÜNDNIS 90/DIE GRÜNEN", "Die Linke"]

        # Test 4: Already canonicalized (allow_list is None), no unnecessary copy/map
        df_canonical = pd.DataFrame({"fraktion": ["SPD", "CDU/CSU", "AfD"], "val": [1, 2, 3]})
        app.fraktion_allow_list = None
        res_canonical = app._apply_party_filter(df_canonical, col="fraktion")
        assert list(res_canonical["fraktion"]) == ["SPD", "CDU/CSU", "AfD"]

        # Test 5: Filtered with allow_list, purging non-official parties and canonicalizing raw names
        app.fraktion_allow_list = frozenset(["SPD", "CDU/CSU", "AfD"])
        df_mixed = pd.DataFrame({"fraktion": ["SPDSPD", "CDU", "Tribüne", None, np.nan, "AfD"], "val": range(6)})
        res_filtered = app._apply_party_filter(df_mixed, col="fraktion")
        assert list(res_filtered["fraktion"]) == ["SPD", "CDU/CSU", "AfD"]
        assert list(res_filtered["val"]) == [0, 1, 5]

        # Test 6: No matching parties in allow_list
        app.fraktion_allow_list = frozenset(["FDP"])
        res_empty_match = app._apply_party_filter(df_mixed, col="fraktion")
        assert res_empty_match.empty

    def test_official_parties_toggle_filters_unwanted_factions(self, analytics_dataset):
        from streamlit.testing.v1 import AppTest
        import src.app as app
        import pandas as pd
        
        # 1. Direct Unit Test of the filter function
        df_mock = pd.DataFrame({"fraktion": ["SPD", "CDU/CSU", "Tribüne"]})
        
        # Without filter: "Tribüne" remains
        app.fraktion_allow_list = None
        res_unfiltered = app._apply_party_filter(df_mock)
        assert len(res_unfiltered) == 3
        
        # With filter: "Tribüne" is purged
        app.fraktion_allow_list = frozenset(["SPD", "CDU/CSU", "FDP", "BÜNDNIS 90/DIE GRÜNEN", "AfD", "Die Linke", "BSW"])
        res_filtered = app._apply_party_filter(df_mock)
        assert len(res_filtered) == 2
        assert "Tribüne" not in res_filtered["fraktion"].values
        
        # 2. Integration Test via the UI Sidebar
        at = AppTest.from_file(APP_PATH)
        at.query_params["page"] = "ton-analyse"
        at.run()
        
        # Locate the specific toggle by checking its label
        toggles = [t for t in at.toggle if "Nur eingetragene Parteien" in getattr(t, "label", "")]
        if toggles:
            toggles[0].set_value(True).run()
        
        # Ensure the filter pipeline does not crash any underlying Plotly traces
        assert not at.exception


class TestLicenseAndAttribution:
    """Verifies that open data license attribution (dl-de/by-2-0) and DCAT-AP.de metadata standards are properly documented."""

    def test_readme_contains_full_attribution(self):
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()

        assert "Datenlizenz Deutschland – Namensnennung – Version 2.0" in content
        assert "dl-de/by-2-0" in content
        assert "http://www.dcat-ap.de/def/licenses/dl-by-de/2.0" in content
        assert "dl-by-de/2.0" in content
        assert "https://www.bundestag.de/services/opendata" in content
        assert "Deutscher Bundestag" in content
        assert "§ 3 dl-de/by-2-0" in content

    def test_app_py_contains_license_and_dcat_terms(self):
        with open("src/app.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert "Datenlizenz Deutschland – Namensnennung – Version 2.0" in content
        assert "dl-de/by-2-0" in content
        assert "http://www.dcat-ap.de/def/licenses/dl-by-de/2.0" in content
        assert "dl-by-de/2.0" in content
        assert "https://www.bundestag.de/services/opendata" in content

    def test_docs_contain_license_specifications(self):
        for filepath in ["docs/requirements/Lastenheft.md", "docs/specs/Pflichtenheft.md", "docs/project/Project.md"]:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            assert "dl-de/by-2-0" in content, f"Missing dl-de/by-2-0 in {filepath}"
            assert "dl-by-de/2.0" in content, f"Missing dl-by-de/2.0 in {filepath}"

class TestSegmentedFramingDashboard:
    """Verifies that segmented TF-IDF tabs, controls, and comparisons render properly."""

    def test_render_framing_analyse_segmented_options(self, monkeypatch):
        from src.database import init_db, get_session
        from src.models import FramingResult
        init_db()
        import src.app as app

        with get_session() as session:
            session.add_all([
                FramingResult(wahlperiode=20, fraktion="SPD", segment="speech", variant="global", term="arbeit", score=0.8),
                FramingResult(wahlperiode=20, fraktion="SPD", segment="interjection", variant="global", term="unsinn", score=0.9),
                FramingResult(wahlperiode=20, fraktion="CDU/CSU", segment="speech", variant="global", term="wirtschaft", score=0.85),
                FramingResult(wahlperiode=20, fraktion="CDU/CSU", segment="interjection", variant="global", term="heuchelei", score=0.95),
            ])
            session.commit()

        # Call render_framing_analyse for WP 20 with precomputed results present
        # It must execute without crashing
        app.render_framing_analyse(selected_wp=20)

    def test_render_framing_analyse_empty_db_shows_info(self, monkeypatch):
        from unittest.mock import patch
        from src.database import init_db
        init_db()
        import src.app as app

        # For an empty Wahlperiode 99, a clean warning/info is shown without live calculation
        with patch("streamlit.info") as mock_info, patch("streamlit.warning") as mock_warn:
            app.render_framing_analyse(selected_wp=99)
            mock_warn.assert_called_with("Keine vorberechneten Framing-Daten für diese Wahlperiode verfügbar.")
            assert any("precompute_framing.py --wahlperiode 99" in str(c) for c in mock_info.call_args_list)

    def test_cached_segmented_framing_helpers(self, monkeypatch):
        import pandas as pd
        from src.database import init_db, get_session
        from src.models import FramingResult
        init_db()
        from src.app import _cached_segmented_framing, _cached_segment_comparison

        with get_session() as session:
            session.add_all([
                FramingResult(wahlperiode=20, fraktion="SPD", segment="speech", variant="global", term="digitalisierung", score=0.9),
                FramingResult(wahlperiode=20, fraktion="SPD", segment="interjection", variant="global", term="dazwischen", score=0.7),
            ])
            session.commit()

        df_seg = _cached_segmented_framing(20, segment="speech", variant="global", n_terms=5)
        assert isinstance(df_seg, pd.DataFrame)
        assert not df_seg.empty
        assert "digitalisierung" in df_seg["term"].values

        df_comp = _cached_segment_comparison(20, segment_a="speech", segment_b="interjection", variant="global", n_terms=5)
        assert isinstance(df_comp, pd.DataFrame)
        assert not df_comp.empty
        assert "diff" in df_comp.columns
