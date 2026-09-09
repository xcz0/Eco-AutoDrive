"""Thin offline analysis CLI; no training or simulator bootstrap."""

import argparse
from pathlib import Path

from eco_planner.analysis.runner import EXPERIMENTS, analyze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path)
    parser.add_argument("--source-file", type=Path, help="Execution-backend comparison JSON")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    analyze(
        args.experiment,
        args.source_dir,
        args.output_dir,
        figures=not args.no_figures,
        comparison_config=args.comparison_config,
        source_file=args.source_file,
    )
    print(args.output_dir.resolve() / "report.md")


if __name__ == "__main__":
    main()
