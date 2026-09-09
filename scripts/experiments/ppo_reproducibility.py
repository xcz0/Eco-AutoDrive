"""CLI adapter for PPO training reproducibility reports."""

import argparse
import json
from pathlib import Path

from eco_planner.experiments.ppo_reproducibility import summarize_and_write_training_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    report = summarize_and_write_training_runs(args.root, figures=not args.no_figures)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
