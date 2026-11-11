"""
OpenParlament - Streamlit Dashboard
Copyright (C) 2026 Jonas-dpp

Unit tests for code quality, syntax compliance, and flake8/linter standards.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_ast_parse_and_syntax_warnings():
    """Verify that all Python files compile cleanly via AST without SyntaxWarning or SyntaxError."""
    py_files = []
    for root, _, files in os.walk(_PROJECT_ROOT):
        if any(ignored in root for ignored in [".git", "venv", ".venv", "tmp", "__pycache__", ".ruff_cache"]):
            continue
        for file in files:
            if file.endswith(".py"):
                py_files.append(Path(root) / file)

    assert len(py_files) > 0, "Expected to find Python files in project"

    syntax_warnings = []
    syntax_errors = []

    for py_file in py_files:
        with open(py_file, "r", encoding="utf-8") as f:
            code = f.read()

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            try:
                ast.parse(code, filename=str(py_file))
            except Exception as exc:
                syntax_errors.append((py_file, str(exc)))

            for w in captured_warnings:
                if issubclass(w.category, SyntaxWarning):
                    syntax_warnings.append((py_file, str(w.message)))

    assert not syntax_errors, f"Syntax errors found during AST compilation: {syntax_errors}"
    assert not syntax_warnings, f"Syntax warnings found during AST compilation: {syntax_warnings}"


def test_flake8_code_linting():
    """Run flake8 on src/, scripts/, and tests/ for critical lint errors (F821, F401, F841, E9, F7)."""
    if importlib.util.find_spec("flake8") is None:
        pytest.skip("flake8 package is not installed in the current Python environment")

    cmd = [
        sys.executable,
        "-m",
        "flake8",
        "src/",
        "scripts/",
        "tests/",
        "--select=F821,F822,F823,E9,F7,F401,F841,E741",
    ]
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True)
    if "No module named flake8" in proc.stderr:
        pytest.skip("flake8 package is not installed in the current Python environment")
    assert proc.returncode == 0, f"flake8 check failed:\nOutput:\n{proc.stdout}\nError:\n{proc.stderr}"


def test_flake_pyproject_config():
    """Validate that pyproject.toml includes flake8 in dev optional dependencies and configures ruff/linter."""
    pyproject_path = _PROJECT_ROOT / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml not found"

    content = pyproject_path.read_text(encoding="utf-8")
    assert "flake8" in content, "pyproject.toml missing flake8 in optional-dependencies"
    assert "[tool.ruff]" in content, "pyproject.toml missing [tool.ruff] configuration section"
