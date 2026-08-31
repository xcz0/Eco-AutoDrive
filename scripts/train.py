"""Run one configured closed-loop PPO training profile."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from eco_planner._repository import LOCAL_ENVIRONMENT_PATH
from eco_planner.configuration import load_local_environment
from eco_planner.workflows import run_training_job


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="jobs/training/ppo/smoke",
)
def _hydra_main(config: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    summary = run_training_job(config, output_dir)
    print(summary.model_dump_json(indent=2))


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    load_local_environment(LOCAL_ENVIRONMENT_PATH)
    _hydra_main()


if __name__ == "__main__":
    main()
