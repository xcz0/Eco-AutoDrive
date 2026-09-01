"""CLI adapter for PPO reward A/B analysis."""

import argparse
import json
from pathlib import Path

from eco_planner.analysis.reward_ab import summarize_and_write_ab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = summarize_and_write_ab(args.root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["mechanical_status"] == "passed" else 1)


if __name__ == "__main__":
    main()
