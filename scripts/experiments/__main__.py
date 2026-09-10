"""Unified CLI adapters; experiment execution lives in eco_planner.experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from eco_planner._repository import CONFIG_ROOT, LOCAL_ENVIRONMENT_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or analyze repository experiments")
    experiments = parser.add_subparsers(dest="experiment", required=True)
    actions = {
        "fixed-batch": ("collect",),
        "lambda-identifiability": ("run", "analyze"),
        "reward-calibration": ("run", "analyze"),
        "objective-decomposition": ("run", "analyze"),
        "critic-gae-ablation": ("run", "analyze"),
        "energy-sweep": ("run", "analyze"),
        "reward-sanity": ("run", "analyze"),
        "scalar-reward": ("evaluate-a0", "train", "evaluate-policy", "analyze"),
        "ppo-stability": ("stage-a", "stage-b", "stage-c", "diagnose", "summarize", "analyze"),
        "ppo-reproducibility": ("report", "analyze"),
        "execution-backend": ("report", "analyze"),
    }
    defaults = {
        "fixed-batch": "collect.yaml",
        "lambda-identifiability": "diagnostic.yaml",
        "reward-calibration": "calibration.yaml",
        "objective-decomposition": "diagnostic.yaml",
        "critic-gae-ablation": "diagnostic.yaml",
        "energy-sweep": "matrix.yaml",
        "reward-sanity": "sanity.yaml",
        "scalar-reward": "protocol.yaml",
        "ppo-stability": "study.yaml",
    }
    for name, choices in actions.items():
        commands = experiments.add_parser(name).add_subparsers(dest="action", required=True)
        for action in choices:
            command = commands.add_parser(action)
            command.add_argument("--output-dir", type=Path, required=True)
            if action != "collect":
                command.add_argument("--no-figures", action="store_true")
            if action == "analyze":
                command.add_argument("--source-dir", type=Path, required=True)
                if name == "scalar-reward":
                    command.add_argument(
                        "--config", type=Path, required=True, help="Cross-arm comparison config"
                    )
                continue
            if name in defaults:
                command.add_argument(
                    "--config",
                    type=Path,
                    default=CONFIG_ROOT / "experiments" / name / defaults[name],
                )
            if name in (
                "lambda-identifiability",
                "reward-calibration",
                "objective-decomposition",
                "critic-gae-ablation",
                "ppo-reproducibility",
            ):
                command.add_argument("--source-dir", type=Path, required=True)
            if name in ("reward-calibration", "critic-gae-ablation"):
                command.add_argument("--reference-dir", type=Path, required=True)
            if name == "scalar-reward":
                if action in ("train", "evaluate-policy"):
                    command.add_argument("--arm", choices=("a1", "a2"), required=True)
                if action == "train":
                    command.add_argument("--training-seed", type=int, required=True)
                    command.add_argument("--override", action="append", default=[])
                if action == "evaluate-policy":
                    command.add_argument(
                        "--checkpoint", choices=("initial", "final"), required=True
                    )
                    command.add_argument("--checkpoint-path", type=Path, required=True)
            if name == "ppo-stability" and action == "diagnose":
                command.add_argument(
                    "--diagnostic", choices=("gradient", "guidance"), required=True
                )
            if name == "execution-backend":
                for mode in ("serial", "job-level", "vector"):
                    command.add_argument(f"--{mode}-dir", type=Path, required=True)
                    command.add_argument(f"--{mode}-wall-s", type=float, required=True)
    return parser


def bootstrap(args: argparse.Namespace) -> None:
    if args.action in ("analyze", "report", "summarize"):
        return
    if args.experiment in (
        "fixed-batch",
        "lambda-identifiability",
        "reward-calibration",
        "objective-decomposition",
        "critic-gae-ablation",
        "scalar-reward",
        "ppo-stability",
    ):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.experiment in ("fixed-batch", "energy-sweep", "scalar-reward", "ppo-stability"):
        from eco_planner.configuration import load_local_environment

        load_local_environment(LOCAL_ENVIRONMENT_PATH)


def dispatch(args: argparse.Namespace) -> dict[str, Any] | int:
    figures = not getattr(args, "no_figures", False)
    name = args.experiment
    if args.action == "analyze":
        from eco_planner.analysis.runner import analyze

        return analyze(
            name,
            args.source_dir,
            args.output_dir,
            figures=figures,
            comparison_config=getattr(args, "config", None),
        )
    if name == "fixed-batch":
        from eco_planner.experiments.fixed_batch.collection import collect

        return collect(args.config, args.output_dir)
    if name == "lambda-identifiability":
        from eco_planner.experiments.lambda_identifiability.runner import run

        return run(args.source_dir, args.config, args.output_dir, figures=figures)
    if name == "reward-calibration":
        from eco_planner.experiments.reward_calibration.runner import run

        return run(
            args.source_dir, args.reference_dir, args.config, args.output_dir, figures=figures
        )
    if name == "objective-decomposition":
        from eco_planner.experiments.objective_decomposition.runner import run

        return run(args.source_dir, args.config, args.output_dir, figures=figures)
    if name == "critic-gae-ablation":
        from eco_planner.experiments.critic_gae_ablation.runner import run

        return run(
            args.source_dir, args.reference_dir, args.config, args.output_dir, figures=figures
        )
    if name == "energy-sweep":
        from eco_planner.experiments.energy_sweep.runner import run_study

        return run_study(args.config, args.output_dir, figures=figures)
    if name == "reward-sanity":
        from eco_planner.experiments.reward_sanity.runner import run_sanity

        return run_sanity(args.config, args.output_dir, figures=figures)
    if name == "scalar-reward":
        from eco_planner.experiments.scalar_reward.runner import run_command

        return run_command(
            args.action,
            args.config,
            args.output_dir,
            figures=figures,
            arm=getattr(args, "arm", None),
            training_seed=getattr(args, "training_seed", None),
            checkpoint_label=getattr(args, "checkpoint", None),
            checkpoint_path=getattr(args, "checkpoint_path", None),
            overrides=getattr(args, "override", ()),
        )
    if name == "ppo-stability":
        from eco_planner.experiments.ppo_stability.runner import run_command

        return run_command(
            args.action,
            args.config,
            args.output_dir,
            getattr(args, "diagnostic", None),
            figures=figures,
        )
    if name == "ppo-reproducibility":
        from eco_planner.experiments.ppo_reproducibility import summarize_and_write_training_runs

        return summarize_and_write_training_runs(args.source_dir, args.output_dir, figures=figures)
    if name == "execution-backend":
        from eco_planner.experiments.execution_backend.report import write_report

        return write_report(
            args.serial_dir,
            args.job_level_dir,
            args.vector_dir,
            serial_wall_s=args.serial_wall_s,
            job_level_wall_s=args.job_level_wall_s,
            vector_wall_s=args.vector_wall_s,
            output=args.output_dir / "evaluation_modes.json",
            figures=figures,
        )
    raise ValueError(f"unsupported experiment: {name}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
