"""Validate the four pre-registered closed-loop training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eco_planner.artifacts import write_json
from eco_planner.rl.artifacts import summarize_training_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = summarize_training_runs(args.root)
    write_json(args.root / "training_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
