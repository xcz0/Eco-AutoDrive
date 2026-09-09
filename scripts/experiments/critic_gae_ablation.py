"""CLI for Issue 94 Task C4 fixed-batch critic/GAE common-term ablation."""

import argparse
import json
import os
from pathlib import Path

from eco_planner._repository import CONFIG_ROOT
from eco_planner.experiments.critic_gae_ablation_runner import run


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="E-035 objective-decomposition output used to cross-check the standard-GAE arms",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_ROOT / "experiments/scalar_reward/critic_gae_ablation.yaml",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.source_dir.resolve(),
                args.reference_dir.resolve(),
                args.config.resolve(),
                args.output_dir.resolve(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
