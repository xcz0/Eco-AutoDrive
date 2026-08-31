"""Run fixed-seed official Diffusion Planner evaluation in MetaDrive."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from eco_planner.configuration import load_local_environment
from eco_planner.workflows import run_evaluation_job
from scripts._paths import LOCAL_ENVIRONMENT_PATH

load_local_environment(LOCAL_ENVIRONMENT_PATH)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="jobs/evaluation/no_traffic/full",
)
def main(config: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    summary = run_evaluation_job(config, output_dir)
    print(summary.model_dump_json(indent=2))
    if summary.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
