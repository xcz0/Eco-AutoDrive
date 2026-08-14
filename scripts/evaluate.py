"""Run fixed-seed official Diffusion Planner evaluation in MetaDrive."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from eco_planner.evaluation.config import parse_evaluation_config
from eco_planner.evaluation.runner import run_evaluation


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="evaluation/no_traffic",
)
def main(config: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    summary = run_evaluation(parse_evaluation_config(config), output_dir)
    print(summary.model_dump_json(indent=2))
    if summary.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
