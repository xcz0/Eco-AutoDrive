"""Thin, lazy command adapters for the three research domains."""

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
    evidence: str
    module: str
    function: str
    actions: tuple[str, ...]
    config: str | None = None
    source: bool = False
    reference: bool = False
    cuda: bool = False
    environment: bool = False


COMMANDS = {
    ("reward", "scalar"): Command(
        "scalar-reward",
        "reward.scalar.runner",
        "run_command",
        ("run", "analyze"),
        "protocol.yaml",
        cuda=True,
        environment=True,
    ),
    ("reward", "fixed-batch"): Command(
        "fixed-batch",
        "reward.fixed_batch.collection",
        "collect",
        ("collect",),
        "collect.yaml",
        cuda=True,
        environment=True,
    ),
    ("reward", "lambda-identifiability"): Command(
        "lambda-identifiability",
        "reward.lambda_identifiability.runner",
        "run",
        ("run", "analyze"),
        "diagnostic.yaml",
        source=True,
        cuda=True,
    ),
    ("reward", "calibration"): Command(
        "reward-calibration",
        "reward.calibration.runner",
        "run",
        ("run", "analyze"),
        "calibration.yaml",
        source=True,
        reference=True,
        cuda=True,
    ),
    ("reward", "objective-decomposition"): Command(
        "objective-decomposition",
        "reward.objective_decomposition.runner",
        "run",
        ("run", "analyze"),
        "diagnostic.yaml",
        source=True,
        cuda=True,
    ),
    ("reward", "critic-gae-ablation"): Command(
        "critic-gae-ablation",
        "reward.critic_gae_ablation.runner",
        "run",
        ("run", "analyze"),
        "diagnostic.yaml",
        source=True,
        reference=True,
        cuda=True,
    ),
    ("guidance", "energy-sweep"): Command(
        "energy-sweep",
        "guidance.energy_sweep.runner",
        "run_study",
        ("run", "analyze"),
        "matrix.yaml",
        environment=True,
    ),
    ("guidance", "control-authority"): Command(
        "guidance-control-authority",
        "guidance.control_authority.runner",
        "run",
        ("run", "analyze"),
        "intervention.yaml",
        cuda=True,
        environment=True,
    ),
    ("training", "stability"): Command(
        "ppo-stability",
        "training.stability.runner",
        "run_command",
        ("run", "analyze"),
        "study.yaml",
        cuda=True,
        environment=True,
    ),
    ("training", "reproducibility"): Command(
        "ppo-reproducibility",
        "training.reproducibility",
        "summarize_and_write_training_runs",
        ("validate", "analyze"),
        source=True,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or analyze repository research studies")
    domains = parser.add_subparsers(dest="domain", required=True)
    for domain in ("reward", "guidance", "training"):
        studies = domains.add_parser(domain).add_subparsers(dest="study", required=True)
        for (owner, name), spec in COMMANDS.items():
            if owner != domain:
                continue
            actions = studies.add_parser(name).add_subparsers(dest="action", required=True)
            for action in spec.actions:
                command = actions.add_parser(action)
                command.set_defaults(command=spec)
                command.add_argument("--output-dir", type=Path, required=True)
                if action != "collect":
                    command.add_argument("--no-figures", action="store_true")
                if action == "analyze":
                    command.add_argument("--source-dir", type=Path, required=True)
                    if spec.evidence == "scalar-reward":
                        command.add_argument("--config", type=Path, required=True)
                    continue
                if spec.config:
                    command.add_argument(
                        "--config",
                        type=Path,
                        default=CONFIG_ROOT / "experiments" / domain / name / spec.config,
                    )
                if spec.source:
                    command.add_argument("--source-dir", type=Path, required=True)
                if spec.reference:
                    command.add_argument("--reference-dir", type=Path, required=True)
                if spec.evidence == "scalar-reward":
                    command.add_argument(
                        "--operation", choices=("train", "evaluate"), required=True
                    )
                    command.add_argument("--arm", choices=("a0", "a1", "a2"), required=True)
                    command.add_argument("--training-seed", type=int)
                    command.add_argument("--override", action="append", default=[])
                    command.add_argument("--checkpoint", choices=("initial", "final"))
                    command.add_argument("--checkpoint-path", type=Path)
                if spec.evidence == "ppo-stability":
                    command.add_argument(
                        "--operation",
                        choices=("search", "confirm", "held-out", "diagnostic"),
                        required=True,
                    )
                    command.add_argument("--diagnostic", choices=("gradient", "guidance"))
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.action != "run":
        return
    if args.command.evidence == "scalar-reward":
        if args.operation == "train":
            if args.arm == "a0" or args.training_seed is None:
                parser.error("train requires --arm a1/a2 and --training-seed")
            if args.checkpoint is not None or args.checkpoint_path is not None:
                parser.error("train does not accept checkpoint arguments")
        else:
            if args.training_seed is not None or args.override:
                parser.error("evaluate does not accept training arguments")
            if args.arm == "a0":
                if args.checkpoint is not None or args.checkpoint_path is not None:
                    parser.error("a0 does not accept checkpoint arguments")
            elif args.checkpoint is None or args.checkpoint_path is None:
                parser.error("a1/a2 evaluation requires --checkpoint and --checkpoint-path")
    if args.command.evidence == "ppo-stability":
        if (args.operation == "diagnostic") != (args.diagnostic is not None):
            parser.error("--diagnostic is required only for --operation diagnostic")


def bootstrap(args: argparse.Namespace) -> None:
    if args.action in ("analyze", "validate"):
        return
    if args.command.cuda:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command.environment:
        from eco_planner.configuration import load_local_environment

        load_local_environment(LOCAL_ENVIRONMENT_PATH)


def dispatch(args: argparse.Namespace) -> dict[str, Any] | int:
    spec = args.command
    figures = not getattr(args, "no_figures", False)
    if args.action == "analyze":
        from eco_planner.analysis.runner import analyze

        comparison = None
        if spec.evidence == "scalar-reward":
            from eco_planner.experiments.reward.scalar.comparison import load_comparison

            comparison = load_comparison(args.config)
        return analyze(
            spec.evidence,
            args.source_dir,
            args.output_dir,
            figures=figures,
            scalar_comparison=comparison,
        )
    function = getattr(import_module("eco_planner.experiments." + spec.module), spec.function)
    if spec.evidence == "scalar-reward":
        return function(
            args.operation,
            args.config,
            args.output_dir,
            figures=figures,
            arm=args.arm,
            training_seed=args.training_seed,
            checkpoint_label=args.checkpoint,
            checkpoint_path=args.checkpoint_path,
            overrides=args.override,
        )
    if spec.evidence == "ppo-stability":
        return function(
            args.operation,
            args.config,
            args.output_dir,
            args.diagnostic,
            figures=figures,
        )
    if args.action == "collect":
        return function(args.config, args.output_dir)
    if args.action == "validate":
        return function(args.source_dir, args.output_dir, figures=figures)
    inputs = [args.source_dir] if spec.source else []
    if spec.reference:
        inputs.append(args.reference_dir)
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
