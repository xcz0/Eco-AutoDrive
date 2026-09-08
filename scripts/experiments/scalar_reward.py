"""CLI adapter for the matched scalar-reward experiment protocol."""

import argparse
import json
import os
from pathlib import Path

from eco_planner._repository import LOCAL_ENVIRONMENT_PATH
from eco_planner.configuration import load_local_environment
from eco_planner.experiments.scalar_reward.config import DEFAULT_PROTOCOL
from eco_planner.experiments.scalar_reward.runner import run_command


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    load_local_environment(LOCAL_ENVIRONMENT_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate-a0", "train", "evaluate-policy"))
    parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("a1", "a2"))
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--checkpoint", choices=("initial", "final"))
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.action == "train" and args.arm is None:
        parser.error("train requires --arm")
    if args.action == "train" and args.training_seed is None:
        parser.error("train requires --training-seed")
    if args.action == "evaluate-policy" and (
        args.arm is None or args.checkpoint is None or args.checkpoint_path is None
    ):
        parser.error("evaluate-policy requires --arm, --checkpoint, and --checkpoint-path")
    payload = run_command(
        args.action,
        args.config.resolve(),
        args.output_dir.resolve(),
        arm=args.arm,
        training_seed=args.training_seed,
        checkpoint_label=args.checkpoint,
        checkpoint_path=None if args.checkpoint_path is None else args.checkpoint_path.resolve(),
        overrides=args.override,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
