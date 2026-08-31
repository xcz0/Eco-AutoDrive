"""CLI adapter for the energy study workflow."""

import argparse
from pathlib import Path

from eco_planner.studies.energy import DEFAULT_STUDY, run_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_study(args.study.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
