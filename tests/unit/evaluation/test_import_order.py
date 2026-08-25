"""Regression: offline evaluation artifact readers avoid runtime-only imports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROBE = """
import sys

import eco_planner.evaluation.artifacts.readers  # noqa: F401

assert "torch" not in sys.modules
assert not any(name == "metadrive" or name.startswith("metadrive.") for name in sys.modules)
"""


def test_offline_reader_does_not_import_torch_or_metadrive() -> None:
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
