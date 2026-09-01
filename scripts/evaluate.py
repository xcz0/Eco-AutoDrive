"""Run fixed-seed official Diffusion Planner evaluation in MetaDrive."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from eco_planner._repository import LOCAL_ENVIRONMENT_PATH
from eco_planner.configuration import load_local_environment, with_machine_resource_override
from eco_planner.jobs import run_evaluation_job


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="jobs/evaluation/no_traffic/full",
)
def _hydra_main(config: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    summary = run_evaluation_job(config, output_dir)
    print(summary.model_dump_json(indent=2))
    if summary.status == "failed":
        raise SystemExit(1)


def main() -> None:
    load_local_environment(LOCAL_ENVIRONMENT_PATH)
    sys.argv[:] = with_machine_resource_override(sys.argv)
    _hydra_main()


if __name__ == "__main__":
    main()
