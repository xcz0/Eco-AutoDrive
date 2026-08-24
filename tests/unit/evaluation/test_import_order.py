"""Regression: importing the evaluation artifact reader must load torch before
MetaDrive/Panda3D, otherwise worker processes that import this module first fail
to initialize torch's c10.dll on Windows (WinError 1114)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROBE = r"""
import sys

order = []


class _Finder:
    def find_module(self, name, path=None):
        if name == "torch" and "torch" not in order:
            order.append("torch")
        elif (name == "metadrive" or name.startswith("metadrive.")) and not any(
            n.startswith("metadrive") for n in order
        ):
            order.append("metadrive")
        return None


sys.meta_path.insert(0, _Finder())
import eco_planner.evaluation.artifacts.io  # noqa: F401
print(",".join(order))
"""


def test_imports_torch_before_metadrive() -> None:
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    order = [entry for entry in result.stdout.strip().split(",") if entry]
    assert order.index("torch") < order.index("metadrive")
