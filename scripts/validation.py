"""Reward correctness checks and offline presentation."""

import argparse
import json
from pathlib import Path

from eco_planner._repository import CONFIG_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    targets = parser.add_subparsers(dest="target", required=True)
    actions = targets.add_parser("reward").add_subparsers(dest="action", required=True)
    for action in ("run", "analyze"):
        command = actions.add_parser(action)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--no-figures", action="store_true")
        if action == "run":
            command.add_argument(
                "--config",
                type=Path,
                default=CONFIG_ROOT / "validation" / "reward" / "sanity.yaml",
            )
        else:
            command.add_argument("--source-dir", type=Path, required=True)
    return parser


def dispatch(args: argparse.Namespace) -> dict:
    if args.action == "run":
        from eco_planner.reward_validation import run_sanity

        return run_sanity(args.config, args.output_dir, figures=not args.no_figures)
    from eco_planner.analysis.runner import analyze

    return analyze("reward-sanity", args.source_dir, args.output_dir, figures=not args.no_figures)


def main() -> None:
    args = build_parser().parse_args()
    result = dispatch(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.action == "run":
        raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
