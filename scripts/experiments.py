"""Thin lazy CLI for current scientific workflows."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from eco_planner._repository import CONFIG_ROOT, LOCAL_ENVIRONMENT_PATH


@dataclass(frozen=True)
class Command:
    module: str
    function: str
    config: str | None
    source: bool = False
    environment: bool = False
    cuda: bool = False


COMMANDS = {
    ("compare", "train"): Command(
        "comparison.runner", "run_command", "comparison/default.yaml", environment=True, cuda=True
    ),
    ("compare", "eval"): Command(
        "comparison.runner", "run_command", "comparison/default.yaml", environment=True, cuda=True
    ),
    ("compare", "analyze"): Command("comparison.inputs", "load_comparison", None, source=True),
    ("reward", "collect"): Command(
        "reward.runner", "collect", "reward/collection.yaml", environment=True, cuda=True
    ),
    ("reward", "run"): Command("reward.runner", "run", "reward/default.yaml", source=True),
    ("reward", "analyze"): Command("", "", None, source=True),
    ("credit", "run"): Command(
        "credit.runner", "run", "credit/objectives.yaml", source=True, cuda=True
    ),
    ("credit", "analyze"): Command("", "", None, source=True),
    ("guidance", "authority", "run"): Command(
        "guidance.authority.runner", "run", "guidance/authority.yaml", environment=True, cuda=True
    ),
    ("guidance", "authority", "analyze"): Command("", "", None, source=True),
    ("guidance", "horizon", "run"): Command(
        "guidance.horizon.runner", "run", "guidance/horizon.yaml", environment=True, cuda=True
    ),
    ("guidance", "horizon", "analyze"): Command("", "", None, source=True),
    ("guidance", "deferral", "run"): Command(
        "guidance.deferral.runner", "run", "guidance/deferral.yaml", environment=True, cuda=True
    ),
    ("guidance", "deferral", "analyze"): Command("", "", None, source=True),
    ("guidance", "decomposition", "run"): Command(
        "guidance.decomposition.runner",
        "run",
        "guidance/decomposition.yaml",
        environment=True,
        cuda=True,
    ),
    ("guidance", "decomposition", "analyze"): Command("", "", None, source=True),
    ("guidance", "sweep", "run"): Command(
        "guidance.sweep", "run_study", "guidance/energy-sweep/matrix.yaml", environment=True
    ),
    ("guidance", "sweep", "analyze"): Command("", "", None, source=True),
    ("training", "grid"): Command(
        "training.grid", "run", "training/grid.yaml", environment=True, cuda=True
    ),
    ("training", "diagnose"): Command("training.runner", "diagnose", None),
    ("training", "eval"): Command("training.runner", "evaluate", None, environment=True, cuda=True),
    ("training", "critic-attribution", "run"): Command(
        "training.critic_attribution.runner",
        "run",
        "training/critic-attribution.yaml",
        source=True,
    ),
    ("training", "critic-attribution", "analyze"): Command("", "", None, source=True),
    ("training", "analyze"): Command("", "", None, source=True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run current research comparisons and diagnostics")
    domains = parser.add_subparsers(dest="domain", required=True)
    for domain in ("compare", "reward", "credit", "guidance", "training"):
        sub = domains.add_parser(domain).add_subparsers(required=True)
        if domain == "guidance":
            nested = {
                name: sub.add_parser(name).add_subparsers(required=True)
                for name in ("authority", "decomposition", "deferral", "horizon", "sweep")
            }
        elif domain == "training":
            nested = {
                "critic-attribution": sub.add_parser("critic-attribution").add_subparsers(
                    required=True
                )
            }
        else:
            nested = {}
        for key, spec in COMMANDS.items():
            if key[0] != domain:
                continue
            parent = nested[key[1]] if key[1] in nested else sub
            command = parent.add_parser(key[-1])
            command.set_defaults(command=spec, key=key, action=key[-1])
            command.add_argument("--output-dir", type=Path, required=True)
            if key[-1] != "collect":
                command.add_argument("--no-figures", action="store_true")
            if spec.source:
                command.add_argument("--source-dir", type=Path, required=True)
            if spec.config:
                command.add_argument(
                    "--config", type=Path, default=CONFIG_ROOT / "experiments" / spec.config
                )
            elif key in (("compare", "analyze"), ("training", "diagnose"), ("training", "eval")):
                command.add_argument("--config", type=Path, required=True)
            if domain == "compare" and key[-1] != "analyze":
                command.add_argument("--arm", required=True)
                if key[-1] == "train":
                    command.add_argument("--training-seed", type=int, required=True)
                    command.add_argument("--override", action="append", default=[])
                else:
                    command.add_argument("--checkpoint", choices=("initial", "final"))
                    command.add_argument("--checkpoint-path", type=Path)
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.key == ("compare", "eval") and (
        (args.checkpoint is None) != (args.checkpoint_path is None)
    ):
        parser.error("--checkpoint and --checkpoint-path must be supplied together")


def bootstrap(args: argparse.Namespace) -> None:
    if args.command.cuda:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command.environment:
        from eco_planner.configuration import load_local_environment

        load_local_environment(LOCAL_ENVIRONMENT_PATH)


def dispatch(args: argparse.Namespace) -> dict[str, Any] | int:
    figures = not getattr(args, "no_figures", False)
    if args.action == "analyze":
        from eco_planner.analysis.runner import analyze

        evidence = {
            ("guidance", "authority"): "guidance-control-authority",
            ("guidance", "horizon"): "guidance-horizon",
            ("guidance", "deferral"): "guidance-deferral",
            ("guidance", "decomposition"): "guidance-decomposition",
            ("guidance", "sweep"): "energy-sweep",
            ("training", "critic-attribution"): "training-critic-attribution",
        }
        kind = evidence.get(
            args.key[:-1], "scalar-reward" if args.domain == "compare" else args.domain
        )
        comparison = None
        if args.domain == "compare":
            from eco_planner.experiments.comparison.inputs import load_comparison

            comparison = load_comparison(args.config)
        return analyze(
            kind, args.source_dir, args.output_dir, figures=figures, scalar_comparison=comparison
        )
    function = getattr(
        import_module("eco_planner.experiments." + args.command.module), args.command.function
    )
    if args.domain == "compare":
        kwargs = {"arm": args.arm, "figures": figures}
        if args.action == "train":
            kwargs.update(training_seed=args.training_seed, overrides=args.override)
        else:
            kwargs.update(checkpoint_label=args.checkpoint, checkpoint_path=args.checkpoint_path)
        return function(
            "train" if args.action == "train" else "evaluate",
            args.config,
            args.output_dir,
            **kwargs,
        )
    if args.action == "collect":
        return function(args.config, args.output_dir)
    inputs = [args.source_dir] if args.command.source else []
    return function(*inputs, args.config, args.output_dir, figures=figures)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    bootstrap(args)
    result = dispatch(args)
    if isinstance(result, int):
        raise SystemExit(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(1 if result.get("status") == "failed" else 0)


if __name__ == "__main__":
    main()
