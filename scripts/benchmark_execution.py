"""Compare measured execution backends or regenerate their offline report."""

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("report", "analyze"):
        command = actions.add_parser(action)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--no-figures", action="store_true")
        if action == "analyze":
            command.add_argument("--source-dir", type=Path, required=True)
        else:
            for mode in ("serial", "job-level", "vector"):
                command.add_argument(f"--{mode}-dir", type=Path, required=True)
                command.add_argument(f"--{mode}-wall-s", type=float, required=True)
    return parser


def dispatch(args: argparse.Namespace) -> dict:
    if args.action == "analyze":
        from eco_planner.analysis.runner import analyze

        return analyze(
            "execution-backend",
            args.source_dir,
            args.output_dir,
            figures=not args.no_figures,
        )
    from eco_planner.benchmarking.execution import write_report

    return write_report(
        args.serial_dir,
        args.job_level_dir,
        args.vector_dir,
        serial_wall_s=args.serial_wall_s,
        job_level_wall_s=args.job_level_wall_s,
        vector_wall_s=args.vector_wall_s,
        output=args.output_dir / "evaluation_modes.json",
        figures=not args.no_figures,
    )


def main() -> None:
    print(json.dumps(dispatch(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
