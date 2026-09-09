"""CLI for Issue 94 Task C fixed-batch objective decomposition."""

import argparse
import json
import os
from pathlib import Path

from eco_planner._repository import CONFIG_ROOT
from eco_planner.experiments.objective_decomposition_runner import run


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_ROOT / "experiments/scalar_reward/objective_decomposition.yaml",
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.source_dir.resolve(),
                args.config.resolve(),
                args.output_dir.resolve(),
                figures=not args.no_figures,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
