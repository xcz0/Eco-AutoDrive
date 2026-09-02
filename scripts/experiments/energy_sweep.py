"""CLI adapter for the fixed energy-sweep experiment."""

import argparse
from pathlib import Path

from eco_planner._repository import LOCAL_ENVIRONMENT_PATH
from eco_planner.configuration import load_local_environment
from eco_planner.experiments.energy_sweep import DEFAULT_STUDY, run_study


def main() -> None:
    load_local_environment(LOCAL_ENVIRONMENT_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_study(args.config.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
