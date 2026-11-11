#!/usr/bin/env python3
"""
Export Dashboard Plots Script for OpenParlament.

Automated Plotly chart exporter using a two-pass widget discovery and Cartesian-product
configuration space explorer.

Pass 1 (Discovery):
    Iteratively renders each page with instrumented Streamlit mocks to discover all
    widgets—including conditional widget branches—and records call-site/key widget IDs,
    candidate values, and enumeration strategies.

Pass 2 (Execution):
    Generates the Cartesian product of candidate widget values, prunes duplicate inactive
    branch configurations to reach valid states, executes the page/tabs for every reachable
    configuration, captures Plotly charts, adds unobtrusive PNG annotations, and saves them
    as PNG and/or JSON with collision-proof filenames and manifest metadata.

Usage:
    python scripts/export_dashboard_plots.py [--output-dir data/dashboard_plots] [--format png|json|both] [--max-pages N] [--max-configs N]
"""

from __future__ import annotations

import argparse
import datetime
import enum
import hashlib
import inspect
import itertools
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import plotly.graph_objects as go

# Supported Bundestag Wahlperioden configuration dimension for dashboard plot exports
WAHLPERIODEN: List[int] = [18, 19, 20, 21, 22]

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Mock get_script_run_ctx before importing src.app so st.Page object initializations retain page functions
fake_ctx = MagicMock()
fake_ctx.pages_manager.main_script_parent = "."

with patch("streamlit.navigation.page.get_script_run_ctx", return_value=fake_ctx):
    import src.app as app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_dashboard_plots")


class ManifestJSONEncoder(json.JSONEncoder):
    """JSON encoder supporting sets, tuples, numpy scalars, datetimes, dates, enums, and paths."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def sanitize_filename(name: str) -> str:
    """Sanitize string for filesystem safety."""
    name = re.sub(r"<[^>]+>", "", name)  # strip HTML tags
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[-\s]+", "_", name).strip("_")
    return name.lower() or "plot"


def compute_figure_hash(fig: Any) -> str:
    """Compute a deterministic SHA-256 hash for a Plotly figure."""
    try:
        if hasattr(fig, "to_dict"):
            data_dict = fig.to_dict()
        elif isinstance(fig, dict):
            data_dict = fig
        else:
            data_dict = str(fig)
        json_bytes = json.dumps(data_dict, sort_keys=True, cls=ManifestJSONEncoder).encode("utf-8")
        return hashlib.sha256(json_bytes).hexdigest()
    except Exception:
        return hashlib.sha256(str(fig).encode("utf-8")).hexdigest()


class DummyContextManager:
    """Mock context manager for Streamlit containers (e.g. st.spinner, st.expander)."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TabContextManager:
    """Mock context manager for st.tabs."""

    def __init__(self, label: str, *args, **kwargs):
        self.label = label

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def _get_page_callable(page_obj: Any) -> Optional[Tuple[str, Any]]:
    """Resolve callable page function from st.Page object or function."""
    if callable(page_obj):
        return getattr(page_obj, "__name__", "custom_page"), page_obj

    page_func = getattr(page_obj, "_page", None) or getattr(page_obj, "_page_func", None)
    if callable(page_func):
        title = getattr(page_obj, "title", None) or getattr(page_func, "__name__", "custom_page")
        return str(title), page_func

    return None


def get_deterministic_widget_id(
    widget_type: str,
    label: str,
    key: Optional[str],
    occurrence_counts: Dict[Tuple[str, str], int],
) -> str:
    """Generate a stable, deterministic widget ID based on key or call-site stack inspection."""
    if key:
        return f"key:{key}"

    # Inspect call-site frame outside exporter script
    callsite_id = None
    try:
        frame = inspect.currentframe()
        while frame:
            filename = frame.f_code.co_filename
            if "export_dashboard_plots" not in filename and "unittest" not in filename and "importlib" not in filename:
                func_name = frame.f_code.co_name
                line_no = frame.f_lineno
                short_fn = Path(filename).name
                clean_lbl = sanitize_filename(label)
                callsite_id = f"callsite:{short_fn}:{func_name}:{line_no}:{widget_type}:{clean_lbl}"
                break
            frame = frame.f_back
    except Exception:
        pass

    if callsite_id:
        return callsite_id

    # Fallback to occurrence count
    count = occurrence_counts.get((widget_type, label), 0)
    occurrence_counts[(widget_type, label)] = count + 1
    clean_lbl = sanitize_filename(label)
    return f"fallback:{widget_type}:{clean_lbl}:{count}"


def get_multiselect_candidates(options: List[Any], default: Any) -> Tuple[List[Any], str]:
    """Determine candidate options and enumeration strategy for multiselect."""
    opts_list = list(options) if options else []
    default_val = default if default is not None else []

    if len(opts_list) <= 5:
        powerset = []
        for r in range(len(opts_list) + 1):
            for combo in itertools.combinations(opts_list, r):
                powerset.append(list(combo))
        return powerset, "exhaustive_powerset"

    candidates = [default_val]
    if [] not in candidates:
        candidates.append([])
    if opts_list and [opts_list[0]] not in candidates:
        candidates.append([opts_list[0]])
    if len(opts_list) >= 2 and opts_list[:2] not in candidates:
        candidates.append(opts_list[:2])
    if opts_list[:5] not in candidates:
        candidates.append(opts_list[:5])

    return candidates, "representative_sample"


def get_slider_candidates(min_value: Any, max_value: Any, value: Any, step: Any) -> Tuple[List[Any], str]:
    """Determine candidate options and enumeration strategy for sliders."""
    default_val = value if value is not None else min_value

    if isinstance(min_value, int) and isinstance(max_value, int) and (max_value - min_value) <= 10:
        step_val = step if isinstance(step, int) and step > 0 else 1
        candidates = list(range(min_value, max_value + 1, step_val))
        if max_value not in candidates:
            candidates.append(max_value)
        return candidates, "exhaustive_range"

    candidates = [min_value, default_val, max_value]
    unique_candidates = []
    for c in candidates:
        if c not in unique_candidates:
            unique_candidates.append(c)
    return unique_candidates, "representative_points"


def discover_widgets_for_page(page_func: Any, max_discovery_runs: int = 30) -> Dict[str, Any]:
    """Pass 1: Iteratively renders page function to discover all widgets across conditional branches."""
    discovered_widgets_map: Dict[str, Dict[str, Any]] = {}
    discovery_queue: List[Dict[str, Any]] = [{}]
    visited_assignments: List[Dict[str, Any]] = []

    runs = 0
    discovery_complete = True

    while discovery_queue:
        if runs >= max_discovery_runs:
            logger.warning(
                "Widget discovery reached max_discovery_runs limit (%d). Discovery may not be exhaustive.",
                max_discovery_runs,
            )
            discovery_complete = False
            break

        runs += 1
        current_assignment = discovery_queue.pop(0)
        visited_assignments.append(current_assignment)
        occurrence_counts: Dict[Tuple[str, str], int] = {}

        def _register_and_get_val(
            widget_type: str,
            label: str,
            key: Optional[str],
            default_val: Any,
            candidate_options: List[Any],
            strategy: str,
        ) -> Any:
            wid = get_deterministic_widget_id(widget_type, label, key, occurrence_counts)
            if wid not in discovered_widgets_map:
                unique_candidates = []
                for opt in candidate_options:
                    if opt not in unique_candidates:
                        unique_candidates.append(opt)
                if not unique_candidates:
                    unique_candidates = [default_val]

                discovered_widgets_map[wid] = {
                    "widget_id": wid,
                    "type": widget_type,
                    "label": label,
                    "key": key,
                    "default": default_val,
                    "strategy": strategy,
                    "candidates": unique_candidates,
                }

                # Enqueue new assignments to explore newly discovered conditional branches
                for c_val in unique_candidates:
                    if c_val != default_val:
                        new_assign = dict(current_assignment)
                        new_assign[wid] = c_val
                        if new_assign not in discovery_queue and new_assign not in visited_assignments:
                            discovery_queue.append(new_assign)

            return current_assignment.get(wid, default_val)

        def mock_toggle(label, value=False, key=None, **kwargs):
            return _register_and_get_val("toggle", str(label), key, value, [False, True], "exhaustive_boolean")

        def mock_checkbox(label, value=False, key=None, **kwargs):
            return _register_and_get_val("checkbox", str(label), key, value, [False, True], "exhaustive_boolean")

        def mock_radio(label, options, index=0, key=None, **kwargs):
            default_val = options[index] if options and index < len(options) else (options[0] if options else None)
            candidate_opts = list(options) if options else [default_val]
            return _register_and_get_val("radio", str(label), key, default_val, candidate_opts, "exhaustive_options")

        def mock_selectbox(label, options, index=0, key=None, **kwargs):
            default_val = options[index] if options and index < len(options) else (options[0] if options else None)
            candidate_opts = list(options) if options else [default_val]
            return _register_and_get_val("selectbox", str(label), key, default_val, candidate_opts, "exhaustive_options")

        def mock_multiselect(label, options, default=None, key=None, **kwargs):
            default_val = default if default is not None else []
            candidate_opts, strategy = get_multiselect_candidates(options, default_val)
            return _register_and_get_val("multiselect", str(label), key, default_val, candidate_opts, strategy)

        def mock_slider(label, min_value=0, max_value=100, value=None, step=None, key=None, **kwargs):
            default_val = value if value is not None else min_value
            candidate_opts, strategy = get_slider_candidates(min_value, max_value, value, step)
            return _register_and_get_val("slider", str(label), key, default_val, candidate_opts, strategy)

        def mock_tabs(labels):
            return [TabContextManager(str(lbl)) for lbl in labels]

        with patch("streamlit.plotly_chart", lambda *args, **kwargs: None), \
             patch("streamlit.tabs", side_effect=mock_tabs), \
             patch("streamlit.toggle", side_effect=mock_toggle), \
             patch("streamlit.checkbox", side_effect=mock_checkbox), \
             patch("streamlit.radio", side_effect=mock_radio), \
             patch("streamlit.selectbox", side_effect=mock_selectbox), \
             patch("streamlit.multiselect", side_effect=mock_multiselect), \
             patch("streamlit.slider", side_effect=mock_slider), \
             patch("streamlit.spinner", side_effect=DummyContextManager), \
             patch("streamlit.expander", side_effect=DummyContextManager):

            try:
                sig = inspect.signature(page_func)
                has_selected_wp = "selected_wp" in sig.parameters
                if has_selected_wp and "param:selected_wp" not in discovered_widgets_map:
                    discovered_widgets_map["param:selected_wp"] = {
                        "widget_id": "param:selected_wp",
                        "type": "parameter",
                        "label": "selected_wp",
                        "key": "param:selected_wp",
                        "default": 21,
                        "strategy": "exhaustive_options",
                        "candidates": list(WAHLPERIODEN),
                    }
                    for wp_val in WAHLPERIODEN:
                        # exclude as there is no data available for WP22 yet, 
                        # but still include it in the candidates for completeness
                        if wp_val != 22 and wp_val >= 18: 
                            new_assign = dict(current_assignment)
                            new_assign["param:selected_wp"] = wp_val
                            if new_assign not in discovery_queue and new_assign not in visited_assignments:
                                discovery_queue.append(new_assign)

                args_to_pass = []
                for param in sig.parameters.values():
                    if param.name == "selected_wp":
                        wp_val = current_assignment.get("param:selected_wp", 21)
                        args_to_pass.append(wp_val)
                    elif param.name in ("datum_von", "datum_bis", "fraktion_filter"):
                        args_to_pass.append(None)
                    elif param.default != inspect.Parameter.empty:
                        args_to_pass.append(param.default)
                    else:
                        args_to_pass.append(None)

                page_func(*args_to_pass)
            except Exception as exc:
                logger.debug("Discovery run %d exception: %s", runs, exc)

    discovered_list = list(discovered_widgets_map.values())
    return {
        "widgets": discovered_list,
        "discovery_runs": runs,
        "discovery_limit": max_discovery_runs,
        "discovery_complete": discovery_complete,
    }


def generate_configurations(
    discovered_widgets: List[Dict[str, Any]],
    page_func: Any,
    max_configs: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Generates candidate Cartesian product, prunes inactive branch duplicates, and formats execution configs."""
    if not discovered_widgets:
        return [{"config_id": "config_0000", "values": {}}], 1, 1, 0

    candidate_space = [w["candidates"] for w in discovered_widgets]
    candidate_config_count = 1
    for opts in candidate_space:
        candidate_config_count *= len(opts)

    raw_candidates = itertools.product(*candidate_space)

    # Reachability pruning pass: evaluate active widget keys per candidate configuration lazily
    seen_active_signatures = set()
    reachable_configs = []

    # Bounded candidate scanning: if max_configs is set, stop scanning as soon as max_configs reachable configs are collected
    max_candidate_attempts = (max_configs * 10) if max_configs is not None else float("inf")
    candidate_attempts = 0

    for candidate_tuple in raw_candidates:
        if max_configs is not None and len(reachable_configs) >= max_configs:
            break

        candidate_attempts += 1
        if candidate_attempts > max_candidate_attempts:
            logger.info("Reached candidate evaluation limit (%d). Halting lazy configuration generation.", max_candidate_attempts)
            break

        cand_map = {w_info["widget_id"]: val for w_info, val in zip(discovered_widgets, candidate_tuple)}

        active_called_keys: Dict[str, Any] = {}
        occurrence_counts: Dict[Tuple[str, str], int] = {}

        def _get_val_and_record(widget_type: str, label: str, key: Optional[str], fallback: Any) -> Any:
            wid = get_deterministic_widget_id(widget_type, label, key, occurrence_counts)
            val = cand_map.get(wid, fallback)
            active_called_keys[wid] = val
            return val

        def mock_tabs(labels):
            return [TabContextManager(str(lbl)) for lbl in labels]

        with patch("streamlit.plotly_chart", lambda *args, **kwargs: None), \
             patch("streamlit.tabs", side_effect=mock_tabs), \
             patch("streamlit.toggle", lambda label, value=False, key=None, **kw: _get_val_and_record("toggle", str(label), key, value)), \
             patch("streamlit.checkbox", lambda label, value=False, key=None, **kw: _get_val_and_record("checkbox", str(label), key, value)), \
             patch("streamlit.radio", lambda label, options, index=0, key=None, **kw: _get_val_and_record("radio", str(label), key, options[index] if options and index < len(options) else None)), \
             patch("streamlit.selectbox", lambda label, options, index=0, key=None, **kw: _get_val_and_record("selectbox", str(label), key, options[index] if options and index < len(options) else None)), \
             patch("streamlit.multiselect", lambda label, options, default=None, key=None, **kw: _get_val_and_record("multiselect", str(label), key, default if default is not None else [])), \
             patch("streamlit.slider", lambda label, min_value=0, max_value=100, value=None, step=None, key=None, **kw: _get_val_and_record("slider", str(label), key, value if value is not None else min_value)), \
             patch("streamlit.spinner", side_effect=DummyContextManager), \
             patch("streamlit.expander", side_effect=DummyContextManager):

            try:
                sig = inspect.signature(page_func)
                args_to_pass = []
                for param in sig.parameters.values():
                    if param.name == "selected_wp":
                        wp_val = cand_map.get("param:selected_wp", 21)
                        args_to_pass.append(wp_val)
                        active_called_keys["param:selected_wp"] = wp_val
                    elif param.name in ("datum_von", "datum_bis", "fraktion_filter"):
                        args_to_pass.append(None)
                    elif param.default != inspect.Parameter.empty:
                        args_to_pass.append(param.default)
                    else:
                        args_to_pass.append(None)

                page_func(*args_to_pass)
            except Exception:
                pass

        # Create signature for active called widgets
        sig_tuple = tuple(sorted((k, json.dumps(v, cls=ManifestJSONEncoder)) for k, v in active_called_keys.items()))
        if sig_tuple not in seen_active_signatures:
            seen_active_signatures.add(sig_tuple)
            reachable_configs.append(cand_map)

    reachable_config_count = len(reachable_configs)
    skipped = max(0, candidate_config_count - reachable_config_count)

    formatted_configs = [
        {"config_id": f"config_{idx:04d}", "values": cfg_val} for idx, cfg_val in enumerate(reachable_configs)
    ]

    return formatted_configs, candidate_config_count, reachable_config_count, skipped


def export_plots_for_page(
    page_key: str,
    page_obj: Any,
    output_dir: Path,
    save_png: bool = True,
    save_json: bool = True,
    max_configs: Optional[int] = None,
) -> Dict[str, Any]:
    """Pass 1 & Pass 2: Discovers widgets across conditional branches, explores reachable config space, and exports Plotly charts."""
    resolved = _get_page_callable(page_obj)
    if not resolved:
        logger.warning("Page object for '%s' is not callable. Skipping.", page_key)
        return {
            "page_key": page_key,
            "widgets_discovered": 0,
            "discovery_runs": 0,
            "discovery_limit": 30,
            "discovery_complete": True,
            "candidate_configuration_count": 0,
            "reachable_configuration_count": 0,
            "configurations_executed": 0,
            "configurations_skipped": 0,
            "configuration_limit_applied": False,
            "configuration_limit": max_configs,
            "failed_configurations": 0,
            "charts_captured": 0,
            "png_failures": 0,
            "json_failures": 0,
            "widgets": [],
            "records": [],
        }

    page_title, page_func = resolved

    # Pass 1: Discover widgets
    disc_info = discover_widgets_for_page(page_func)
    discovered_widgets = disc_info["widgets"]

    # Generate configurations
    configs, candidate_count, reachable_count, skipped_count = generate_configurations(
        discovered_widgets, page_func, max_configs=max_configs
    )

    logger.info(
        "Page '%s': Discovered %d widgets (%d candidate, %d reachable, %d executing, %d skipped).",
        page_key,
        len(discovered_widgets),
        candidate_count,
        reachable_count,
        len(configs),
        skipped_count,
    )

    page_records = []
    failed_configs_count = 0
    png_failures_count = 0
    json_failures_count = 0
    total_charts_captured = 0

    page_slug = sanitize_filename(page_key)

    # Timezone-aware timestamp setup
    now_tz = datetime.datetime.now().astimezone()
    captured_at_iso = now_tz.isoformat()
    ts_fs = now_tz.strftime("%Y-%m-%d_%H-%M-%S")
    tz_name = now_tz.tzname() or "local"

    png_dir = output_dir / "png"
    json_dir = output_dir / "json"
    if save_png:
        png_dir.mkdir(parents=True, exist_ok=True)
    if save_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    seen_chart_hashes: Dict[str, Dict[str, Any]] = {}

    configs_total = len(configs)
    cfg_start_time = time.time()

    # Pass 2: Execute each configuration independently
    for cfg_idx, cfg_entry in enumerate(configs, start=1):
        config_id = cfg_entry["config_id"]
        widget_vals = cfg_entry["values"]

        captured_plots: List[Tuple[str, Any]] = []
        occurrence_counts: Dict[Tuple[str, str], int] = {}

        def _get_val(widget_type: str, label: str, key: Optional[str], fallback: Any) -> Any:
            wid = get_deterministic_widget_id(widget_type, label, key, occurrence_counts)
            return widget_vals.get(wid, fallback)

        def mock_plotly_chart(figure_or_data, **kwargs):
            title = ""
            if hasattr(figure_or_data, "layout") and figure_or_data.layout:
                if hasattr(figure_or_data.layout, "title") and figure_or_data.layout.title:
                    title = getattr(figure_or_data.layout.title, "text", "") or ""
            captured_plots.append((title, figure_or_data))

        def mock_tabs(labels):
            return [TabContextManager(str(lbl)) for lbl in labels]

        def mock_toggle(label, value=False, key=None, **kwargs):
            return _get_val("toggle", str(label), key, value)

        def mock_checkbox(label, value=False, key=None, **kwargs):
            return _get_val("checkbox", str(label), key, value)

        def mock_radio(label, options, index=0, key=None, **kwargs):
            fallback = options[index] if options and index < len(options) else (options[0] if options else None)
            return _get_val("radio", str(label), key, fallback)

        def mock_selectbox(label, options, index=0, key=None, **kwargs):
            fallback = options[index] if options and index < len(options) else (options[0] if options else None)
            return _get_val("selectbox", str(label), key, fallback)

        def mock_multiselect(label, options, default=None, key=None, **kwargs):
            fallback = default if default is not None else []
            return _get_val("multiselect", str(label), key, fallback)

        def mock_slider(label, min_value=0, max_value=100, value=None, step=None, key=None, **kwargs):
            fallback = value if value is not None else min_value
            return _get_val("slider", str(label), key, fallback)

        with patch("streamlit.plotly_chart", side_effect=mock_plotly_chart), \
             patch("streamlit.tabs", side_effect=mock_tabs), \
             patch("streamlit.toggle", side_effect=mock_toggle), \
             patch("streamlit.checkbox", side_effect=mock_checkbox), \
             patch("streamlit.radio", side_effect=mock_radio), \
             patch("streamlit.selectbox", side_effect=mock_selectbox), \
             patch("streamlit.multiselect", side_effect=mock_multiselect), \
             patch("streamlit.slider", side_effect=mock_slider), \
             patch("streamlit.spinner", side_effect=DummyContextManager), \
             patch("streamlit.expander", side_effect=DummyContextManager):

            try:
                sig = inspect.signature(page_func)
                args_to_pass = []
                for param in sig.parameters.values():
                    if param.name == "selected_wp":
                        wp_val = widget_vals.get("param:selected_wp", 21)
                        args_to_pass.append(wp_val)
                    elif param.name in ("datum_von", "datum_bis", "fraktion_filter"):
                        args_to_pass.append(None)
                    elif param.default != inspect.Parameter.empty:
                        args_to_pass.append(param.default)
                    else:
                        args_to_pass.append(None)

                page_func(*args_to_pass)
            except Exception as exc:
                logger.warning("Error running page '%s' in %s: %s", page_key, config_id, exc)
                failed_configs_count += 1

        total_charts_captured += len(captured_plots)

        # Export captured charts with collision-proof 1-based chart index & timestamp
        for c_idx, (title_raw, fig) in enumerate(captured_plots, start=1):
            clean_title = sanitize_filename(title_raw) if title_raw else "plot"
            base_filename = f"{page_slug}_c{c_idx:03d}_{clean_title}_{config_id}_{ts_fs}"
            chart_hash = compute_figure_hash(fig)

            record = {
                "page": page_key,
                "config_id": config_id,
                "configuration": widget_vals,
                "chart_index": c_idx,
                "title": title_raw or clean_title,
                "captured_at": captured_at_iso,
                "capture_timezone": tz_name,
                "base_filename": base_filename,
                "chart_hash": chart_hash,
            }

            if chart_hash in seen_chart_hashes:
                # Deduplicate identical chart: reference canonical exported file paths
                existing = seen_chart_hashes[chart_hash]
                record["is_duplicate"] = True
                record["deduplicated_from"] = existing["base_filename"]
                if "json_path" in existing:
                    record["json_path"] = existing["json_path"]
                if "png_path" in existing:
                    record["png_path"] = existing["png_path"]
            else:
                record["is_duplicate"] = False

                # Save JSON specification of original figure
                if save_json and hasattr(fig, "to_json"):
                    json_path = json_dir / f"{base_filename}.json"
                    try:
                        json_str = fig.to_json()
                        if isinstance(json_str, str):
                            try:
                                parsed = json.loads(json_str)
                                json_str = json.dumps(parsed, cls=ManifestJSONEncoder, indent=2, ensure_ascii=False)
                            except Exception:
                                pass
                        json_path.write_text(json_str, encoding="utf-8")
                        record["json_path"] = str(json_path)
                    except Exception as exc:
                        logger.warning("Failed to save JSON for %s: %s", base_filename, exc)
                        json_failures_count += 1

                # Save PNG image with unobtrusive layout annotation
                if save_png and hasattr(fig, "write_image"):
                    png_path = png_dir / f"{base_filename}.png"
                    try:
                        fig_png = go.Figure(fig)
                        fig_png.update_layout(
                            template="plotly", # Zurück zum Standard-Plotly-Theme statt reinem Schwarz-Weiss
                            margin=dict(t=50),
                            colorway=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"] # Standard Plotly Farben
                        )
                        annotation_text = (
                            f"Page: {page_key} | Chart: {title_raw or clean_title} | "
                            f"Captured: {now_tz.strftime('%Y-%m-%d %H:%M:%S')} {tz_name}"
                        )
                        fig_png.add_annotation(
                            text=annotation_text,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=1.08,
                            showarrow=False,
                            font=dict(size=10, color="gray"),
                            align="center",
                        )
                        fig_png.write_image(str(png_path))
                        record["png_path"] = str(png_path)
                    except Exception as exc:
                        logger.warning("Failed to save PNG for %s: %s", base_filename, exc)
                        png_failures_count += 1

                entry_cache = {"base_filename": base_filename}
                if "json_path" in record:
                    entry_cache["json_path"] = record["json_path"]
                if "png_path" in record:
                    entry_cache["png_path"] = record["png_path"]
                seen_chart_hashes[chart_hash] = entry_cache

            page_records.append(record)

        # Throttled progress logging for configuration execution
        log_step = max(1, configs_total // 10)
        if configs_total > 0 and (cfg_idx == 1 or cfg_idx == configs_total or cfg_idx % log_step == 0):
            elapsed_cfg = time.time() - cfg_start_time
            avg_cfg = elapsed_cfg / cfg_idx
            eta_cfg = avg_cfg * (configs_total - cfg_idx)
            eta_cfg_str = time.strftime('%M:%S', time.gmtime(eta_cfg)) if eta_cfg < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta_cfg))
            logger.info(
                "DashboardExportConfigs: %d/%d | %d%% | ETA: %s",
                cfg_idx,
                configs_total,
                int((cfg_idx / configs_total) * 100),
                eta_cfg_str
            )

    return {
        "page_key": page_key,
        "widgets_discovered": len(discovered_widgets),
        "discovery_runs": disc_info["discovery_runs"],
        "discovery_limit": disc_info["discovery_limit"],
        "discovery_complete": disc_info["discovery_complete"],
        "candidate_configuration_count": candidate_count,
        "reachable_configuration_count": reachable_count,
        "configurations_executed": len(configs),
        "configurations_skipped": skipped_count,
        "configuration_limit_applied": max_configs is not None and reachable_count > max_configs,
        "configuration_limit": max_configs,
        "failed_configurations": failed_configs_count,
        "charts_captured": total_charts_captured,
        "png_failures": png_failures_count,
        "json_failures": json_failures_count,
        "widgets": discovered_widgets,
        "records": page_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scans and exports all dashboard plots in all configurations automatically."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dashboard_plots"),
        help="Directory where plot images and JSON files will be stored.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "json", "both"],
        default="both",
        help="Export format (png, json, or both).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to scan (for quick testing).",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of widget configurations per page to execute (default: None, exhaustive).",
    )

    args = parser.parse_args()

    if args.max_pages is not None and args.max_pages <= 0:
        raise ValueError("max-pages must be a positive integer")
    if args.max_configs is not None and args.max_configs < 0:
        raise ValueError("max-configs must be a non-negative integer")

    start_time = datetime.datetime.now().astimezone()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "png").mkdir(parents=True, exist_ok=True)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)

    registered_pages = list(app._PAGE_REGISTRY.keys())
    logger.info("Discovered %d registered pages from app registry.", len(registered_pages))

    if args.max_pages:
        registered_pages = registered_pages[:args.max_pages]
        logger.info("Limiting scan to first %d pages.", args.max_pages)

    save_png = args.format in ["png", "both"]
    save_json = args.format in ["json", "both"]

    all_exported_records: List[Dict] = []
    page_summaries: List[Dict] = []

    total_widgets = 0
    total_candidate = 0
    total_reachable = 0
    total_executed = 0
    total_skipped = 0
    total_failed_configs = 0
    total_charts = 0
    total_png_failures = 0
    total_json_failures = 0

    total_pages = len(registered_pages)
    pages_start_time = time.time()

    for p_idx, page_key in enumerate(registered_pages, start=1):
        page_obj = app._PAGE_REGISTRY[page_key]
        logger.info("--- [%d/%d] Processing Page: %s ---", p_idx, total_pages, page_key)
        summary = export_plots_for_page(
            page_key=page_key,
            page_obj=page_obj,
            output_dir=output_dir,
            save_png=save_png,
            save_json=save_json,
            max_configs=args.max_configs,
        )

        elapsed_p = time.time() - pages_start_time
        avg_p = elapsed_p / p_idx
        eta_p = avg_p * (total_pages - p_idx)
        eta_p_str = time.strftime('%M:%S', time.gmtime(eta_p)) if eta_p < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta_p))
        logger.info(
            "DashboardExportPages: %d/%d | %d%% | ETA: %s",
            p_idx,
            total_pages,
            int((p_idx / total_pages) * 100),
            eta_p_str
        )

        total_widgets += summary["widgets_discovered"]
        total_candidate += summary["candidate_configuration_count"]
        total_reachable += summary["reachable_configuration_count"]
        total_executed += summary["configurations_executed"]
        total_skipped += summary["configurations_skipped"]
        total_failed_configs += summary["failed_configurations"]
        total_charts += summary["charts_captured"]
        total_png_failures += summary["png_failures"]
        total_json_failures += summary["json_failures"]

        all_exported_records.extend(summary["records"])
        page_summaries.append({
            "page": summary["page_key"],
            "widgets_discovered": summary["widgets_discovered"],
            "discovery_runs": summary["discovery_runs"],
            "discovery_limit": summary["discovery_limit"],
            "discovery_complete": summary["discovery_complete"],
            "candidate_configuration_count": summary["candidate_configuration_count"],
            "reachable_configuration_count": summary["reachable_configuration_count"],
            "configurations_executed": summary["configurations_executed"],
            "configurations_skipped": summary["configurations_skipped"],
            "failed_configurations": summary["failed_configurations"],
            "charts_captured": summary["charts_captured"],
            "widgets": summary["widgets"],
        })

    end_time = datetime.datetime.now().astimezone()

    manifest_data = {
        "export_started_at": start_time.isoformat(),
        "export_finished_at": end_time.isoformat(),
        "capture_timezone": start_time.tzname() or "local",
        "summary": {
            "pages_scanned": len(registered_pages),
            "widgets_discovered": total_widgets,
            "candidate_configuration_count": total_candidate,
            "reachable_configuration_count": total_reachable,
            "configurations_executed": total_executed,
            "configurations_skipped": total_skipped,
            "failed_configurations": total_failed_configs,
            "charts_captured": total_charts,
            "png_failures": total_png_failures,
            "json_failures": total_json_failures,
        },
        "pages": page_summaries,
        "records": all_exported_records,
    }

    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, cls=ManifestJSONEncoder, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved export manifest with %d records to %s.", len(all_exported_records), manifest_path)


if __name__ == "__main__":
    main()
