"""Tests for scripts/export_dashboard_plots.py."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# noqa: E402
from scripts.export_dashboard_plots import (  # noqa: E402
    WAHLPERIODEN,
    ManifestJSONEncoder,
    discover_widgets_for_page,
    export_plots_for_page,
    generate_configurations,
    sanitize_filename,
)


class SimplePageWrapper:
    def __init__(self, func, title="Simple Page"):
        self._page = func
        self.title = title


def test_sanitize_filename() -> None:
    assert sanitize_filename("<b>Hello World!</b>") == "hello_world"
    assert sanitize_filename("🔥 Aggressions-Radar") == "aggressions_radar"
    assert sanitize_filename("   ---   ") == "plot"


def test_manifest_json_encoder() -> None:
    data = {
        "tuple": (1, 2),
        "set": {3, 4},
        "np_int": np.int64(42),
        "np_float": np.float64(3.14),
        "np_bool": np.bool_(True),
        "datetime": datetime.datetime(2026, 9, 1, 12, 0, 0),
        "date": datetime.date(2026, 9, 1),
        "path": Path("some/file.txt"),
    }
    encoded = json.dumps(data, cls=ManifestJSONEncoder)
    parsed = json.loads(encoded)
    assert parsed["tuple"] == [1, 2]
    assert sorted(parsed["set"]) == [3, 4]
    assert parsed["np_int"] == 42
    assert parsed["np_float"] == 3.14
    assert parsed["np_bool"] is True
    assert "2026-09-01" in parsed["datetime"]


def test_cartesian_product_exact_count() -> None:
    """Test that 2 checkboxes x 2 checkboxes x 3 selectbox options x 2 radio options = 24 configurations."""

    def dummy_page():
        cb1 = st.checkbox("Checkbox 1", value=False, key="cb1")
        cb2 = st.checkbox("Checkbox 2", value=False, key="cb2")
        sb = st.selectbox("Select 1", options=["A", "B", "C"], index=0, key="sb1")
        rd = st.radio("Radio 1", options=["X", "Y"], index=0, key="rd1")
        if cb1 or cb2 or sb or rd:
            fig = px.scatter(x=[1, 2], y=[3, 4], title=f"Plot {sb} {rd}")
            st.plotly_chart(fig)

    disc_info = discover_widgets_for_page(dummy_page)
    widgets = disc_info["widgets"]
    assert len(widgets) == 4

    configs, candidate_count, reachable_count, skipped = generate_configurations(widgets, dummy_page)
    assert candidate_count == 24  # 2 * 2 * 3 * 2 = 24
    assert reachable_count == 24
    assert len(configs) == 24
    assert skipped == 0


def test_conditional_widget_branches_and_reachable_counts(tmp_path: Path) -> None:
    """Test that conditional widget branches are discovered and reachable configurations pruned."""

    def conditional_page():
        show_details = st.checkbox("Show details", value=False, key="show_details")
        if show_details:
            detail = st.selectbox("Detail", options=["A", "B"], key="detail_type")
            fig = px.scatter(x=[1], y=[1], title=f"Detail {detail}")
            st.plotly_chart(fig)
        else:
            fig = px.scatter(x=[0], y=[0], title="Overview")
            st.plotly_chart(fig)

    page_obj = SimplePageWrapper(conditional_page, "conditional")

    summary = export_plots_for_page(
        page_key="conditional",
        page_obj=page_obj,
        output_dir=tmp_path,
        save_png=False,
        save_json=True,
    )

    assert summary["widgets_discovered"] == 2
    # Candidate configurations: 2 (show_details) x 2 (detail_type) = 4
    assert summary["candidate_configuration_count"] == 4
    # Reachable configurations: show_details=False (1) + show_details=True (detail=A, detail=B) (2) = 3
    assert summary["reachable_configuration_count"] == 3
    assert summary["configurations_executed"] == 3

    # Verify captured records
    records = summary["records"]
    assert len(records) == 3


def test_duplicate_chart_titles_collision_prevention_and_annotation(tmp_path: Path) -> None:
    """Test that two charts with identical titles obtain distinct indices/filenames and annotation is added for PNG."""

    def duplicate_title_page():
        fig1 = go.Figure(layout=dict(title=dict(text="Revenue")))
        fig2 = go.Figure(layout=dict(title=dict(text="Revenue")))
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    page_obj = SimplePageWrapper(duplicate_title_page, "overview")

    summary = export_plots_for_page(
        page_key="overview",
        page_obj=page_obj,
        output_dir=tmp_path,
        save_png=False,
        save_json=True,
    )

    records = summary["records"]
    assert len(records) == 2

    filenames = [r["base_filename"] for r in records]
    assert len(filenames) == len(set(filenames))  # all filenames unique
    assert "_c001_" in filenames[0]
    assert "_c002_" in filenames[1]


def test_error_isolation_between_configurations(tmp_path: Path) -> None:
    """Test that a failure in one configuration does not stop subsequent configurations."""

    def flaky_page():
        val = st.checkbox("Fail trigger", value=False, key="fail_cb")
        if val:
            raise ValueError("Intentional configuration failure")
        fig = px.scatter(x=[1], y=[1], title="Success Plot")
        st.plotly_chart(fig)

    page_obj = SimplePageWrapper(flaky_page, "flaky_page")

    summary = export_plots_for_page(
        page_key="flaky_page",
        page_obj=page_obj,
        output_dir=tmp_path,
        save_png=False,
        save_json=True,
    )

    assert summary["configurations_executed"] == 2
    assert summary["failed_configurations"] == 1
    assert summary["charts_captured"] == 1


def test_export_manifest_writing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "export_manifest.json"
    manifest_data = {
        "export_started_at": "2026-09-01T12:00:00+02:00",
        "export_finished_at": "2026-09-01T12:01:00+02:00",
        "capture_timezone": "CEST",
        "summary": {
            "pages_scanned": 1,
            "widgets_discovered": 2,
            "candidate_configuration_count": 12,
            "reachable_configuration_count": 12,
            "configurations_executed": 12,
            "skipped_configurations": 0,
            "failed_configurations": 0,
            "charts_captured": 12,
            "png_failures": 0,
            "json_failures": 0,
        },
        "pages": [],
        "records": [
            {
                "page": "overview",
                "config_id": "config_0000",
                "configuration": {"key:cb": False, "key:sb": "A", "key:rd": "X"},
                "chart_index": 1,
                "title": "Revenue",
                "captured_at": "2026-09-01T12:00:00+02:00",
                "capture_timezone": "CEST",
                "base_filename": "overview_c001_revenue_config_0000_2026-09-01_12-00-00",
                "json_path": str(tmp_path / "overview_c001_revenue_config_0000_2026-09-01_12-00-00.json"),
            }
        ],
    }

    manifest_path.write_text(json.dumps(manifest_data, cls=ManifestJSONEncoder, indent=2), encoding="utf-8")
    assert manifest_path.exists()

    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert content["summary"]["candidate_configuration_count"] == 12
    assert content["summary"]["reachable_configuration_count"] == 12
    assert len(content["records"]) == 1
    assert content["records"][0]["config_id"] == "config_0000"


def test_chart_deduplication_and_subdirectory_output(tmp_path: Path) -> None:
    """Test that identical charts across configurations are deduplicated and saved to png/ and json/ subfolders."""

    def static_chart_page():
        st.selectbox("Select 1", options=["Option 1", "Option 2"], key="sb_static")
        # Same chart figure regardless of selectbox option
        fig = px.scatter(x=[10, 20], y=[30, 40], title="Static Plot")
        st.plotly_chart(fig)

    page_obj = SimplePageWrapper(static_chart_page, "static_page")

    summary = export_plots_for_page(
        page_key="static_page",
        page_obj=page_obj,
        output_dir=tmp_path,
        save_png=False,
        save_json=True,
    )

    records = summary["records"]
    assert len(records) == 2

    # First record should be canonical (is_duplicate: False)
    assert records[0]["is_duplicate"] is False
    json_p0 = Path(records[0]["json_path"])
    assert json_p0.parent.name == "json"
    assert json_p0.exists()

    # Second record should be deduplicated (is_duplicate: True)
    assert records[1]["is_duplicate"] is True
    assert records[1]["deduplicated_from"] == records[0]["base_filename"]
    assert records[1]["json_path"] == records[0]["json_path"]

    # Only 1 JSON file should exist in the json/ subfolder
    json_files = list((tmp_path / "json").glob("*.json"))
    assert len(json_files) == 1


def test_early_max_configs_termination() -> None:
    """Test that max_configs limits the number of executed configurations early including max_configs=0."""

    def multi_widget_page():
        st.checkbox("CB", value=False, key="cb")
        st.selectbox("SB", options=["A", "B", "C", "D", "E"], key="sb")
        st.radio("RD", options=["1", "2", "3"], key="rd")

    disc_info = discover_widgets_for_page(multi_widget_page)
    widgets = disc_info["widgets"]

    # 2 * 5 * 3 = 30 candidates
    configs, candidate_count, reachable_count, skipped = generate_configurations(
        widgets, multi_widget_page, max_configs=5
    )

    assert candidate_count == 30
    assert reachable_count == 5
    assert len(configs) == 5
    assert skipped == 25

    # Test max_configs = 0
    configs_zero, cand_zero, reach_zero, skipped_zero = generate_configurations(
        widgets, multi_widget_page, max_configs=0
    )
    assert cand_zero == 30
    assert reach_zero == 0
    assert len(configs_zero) == 0
    assert skipped_zero == 30


def test_selected_wp_configuration_dimension(tmp_path: Path) -> None:
    """Test that functions accepting selected_wp iterate across WAHLPERIODEN as a configuration dimension."""

    def page_with_wp(selected_wp: int) -> None:
        show_trend = st.checkbox("Show trend", value=False, key="show_trend")
        fig = px.scatter(x=[selected_wp], y=[1 if show_trend else 0], title=f"WP {selected_wp} Plot")
        st.plotly_chart(fig)

    disc_info = discover_widgets_for_page(page_with_wp)
    widgets = disc_info["widgets"]

    # Must discover param:selected_wp and key:show_trend
    widget_ids = [w["widget_id"] for w in widgets]
    assert "param:selected_wp" in widget_ids
    assert "key:show_trend" in widget_ids

    wp_widget = next(w for w in widgets if w["widget_id"] == "param:selected_wp")
    assert wp_widget["type"] == "parameter"
    assert wp_widget["candidates"] == WAHLPERIODEN

    # 5 WAHLPERIODEN x 2 checkbox states = 10 candidate configurations
    configs, candidate_count, reachable_count, skipped = generate_configurations(widgets, page_with_wp)
    assert candidate_count == len(WAHLPERIODEN) * 2
    assert reachable_count == len(WAHLPERIODEN) * 2

    # Verify execution records
    page_obj = SimplePageWrapper(page_with_wp, "wp_analysis")
    summary = export_plots_for_page(
        page_key="wp_analysis",
        page_obj=page_obj,
        output_dir=tmp_path,
        save_png=False,
        save_json=True,
    )

    assert summary["configurations_executed"] == len(WAHLPERIODEN) * 2
    records = summary["records"]
    assert len(records) == len(WAHLPERIODEN) * 2

    executed_wps = {r["configuration"]["param:selected_wp"] for r in records}
    assert executed_wps == set(WAHLPERIODEN)
