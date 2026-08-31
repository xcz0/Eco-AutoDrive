"""CLI adapter for the evaluation benchmark report."""

import argparse
import json
from pathlib import Path

from eco_planner.benchmarking.evaluation_report import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("serial", type=Path)
    parser.add_argument("job_level", type=Path)
    parser.add_argument("vector", type=Path)
    parser.add_argument("--serial-wall-s", type=float, required=True)
    parser.add_argument("--job-level-wall-s", type=float, required=True)
    parser.add_argument("--vector-wall-s", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation_modes.json"))
    args = parser.parse_args()
    try:
        report = write_report(
            args.serial,
            args.job_level,
            args.vector,
            serial_wall_s=args.serial_wall_s,
            job_level_wall_s=args.job_level_wall_s,
            vector_wall_s=args.vector_wall_s,
            output=args.output,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
