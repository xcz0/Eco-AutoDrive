"""CLI adapter for the PPO reward A/B study workflow."""

import argparse
from pathlib import Path

from eco_planner.studies.reward_ab import DEFAULT_STUDY, run_ab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_ab(args.study.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
