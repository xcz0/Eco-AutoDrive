"""CLI adapter for training-run analysis."""

import argparse
import json
from pathlib import Path

from eco_planner.analysis.training import summarize_and_write_training_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = summarize_and_write_training_runs(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
