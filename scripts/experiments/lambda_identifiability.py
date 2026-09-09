"""CLI for a single fixed-batch lambda identifiability diagnostic."""

import argparse
import json
import os
from pathlib import Path

from eco_planner._repository import CONFIG_ROOT, LOCAL_ENVIRONMENT_PATH
from eco_planner.configuration import load_local_environment
from eco_planner.experiments.lambda_identifiability.runner import run


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    load_local_environment(LOCAL_ENVIRONMENT_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_ROOT / "experiments/scalar_reward/identifiability.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
