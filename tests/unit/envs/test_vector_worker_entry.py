"""Regression tests for the Windows multiprocessing import boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROBE = """
import sys

import eco_planner.vector_worker_entry  # noqa: F401

assert "torch" not in sys.modules
assert not any(name == "metadrive" or name.startswith("metadrive.") for name in sys.modules)
assert not any(name == "panda3d" or name.startswith("panda3d.") for name in sys.modules)
"""


def test_vector_worker_entry_stays_runtime_dependency_free() -> None:
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
