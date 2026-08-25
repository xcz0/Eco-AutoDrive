"""Regression: offline evaluation artifact readers avoid runtime-only imports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROBE = """
import sys

import eco_planner.evaluation.artifacts  # noqa: F401
import eco_planner.evaluation.analysis  # noqa: F401

assert "torch" not in sys.modules
assert not any(name == "metadrive" or name.startswith("metadrive.") for name in sys.modules)
assert not any(name == "panda3d" or name.startswith("panda3d.") for name in sys.modules)
"""


def test_offline_modules_do_not_import_runtime_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
