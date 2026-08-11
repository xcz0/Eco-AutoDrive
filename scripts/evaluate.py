"""Run fixed-seed official Diffusion Planner evaluation in MetaDrive."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from eco_planner.evaluation import run_evaluation


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="evaluation/no_traffic",
)
def main(config: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    summary = run_evaluation(config, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
