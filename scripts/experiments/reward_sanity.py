"""CLI adapter for the reward-sanity experiment."""

import argparse
from pathlib import Path

from eco_planner.experiments.reward_sanity import DEFAULT_CONFIG, run_sanity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_sanity(args.config.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
