"""Validate the four pre-registered closed-loop training runs."""

from __future__ import annotations

from pathlib import Path

from eco_planner.artifacts import write_json
from eco_planner.rl.artifacts import summarize_training_runs


def summarize_and_write_training_runs(root: Path) -> dict[str, object]:
    report = summarize_training_runs(root)
    write_json(root / "training_report.json", report)
    return report
