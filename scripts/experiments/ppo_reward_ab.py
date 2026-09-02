"""CLI adapter for the matched PPO reward A/B experiment."""

import argparse
import json
from pathlib import Path

from eco_planner._repository import LOCAL_ENVIRONMENT_PATH
from eco_planner.configuration import load_local_environment
from eco_planner.experiments.reward_ab.config import DEFAULT_STUDY
from eco_planner.experiments.reward_ab.report import summarize_and_write_ab
from eco_planner.experiments.reward_ab.runner import run_ab


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_STUDY)
    run.add_argument("--output-root", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        load_local_environment(LOCAL_ENVIRONMENT_PATH)
        raise SystemExit(run_ab(args.config.resolve(), args.output_root.resolve()))
    payload = summarize_and_write_ab(args.root.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["mechanical_status"] == "passed" else 1)


if __name__ == "__main__":
    main()
