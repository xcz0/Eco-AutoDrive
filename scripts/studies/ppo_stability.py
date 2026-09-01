"""CLI adapter for the PPO stability study workflow."""

import argparse
import json
import os
from pathlib import Path

from eco_planner._repository import LOCAL_ENVIRONMENT_PATH
from eco_planner.configuration import load_local_environment
from eco_planner.studies.ppo_stability import DEFAULT_STUDY, run_command


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    load_local_environment(LOCAL_ENVIRONMENT_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("stage-a", "stage-b", "stage-c", "diagnose", "summarize")
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diagnostic", choices=("gradient", "guidance"))
    args = parser.parse_args()
    if args.command == "diagnose" and args.diagnostic is None:
        parser.error("diagnose requires --diagnostic")
    payload = run_command(
        args.command,
        args.study.resolve(),
        args.output_root.resolve(),
        args.diagnostic,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
