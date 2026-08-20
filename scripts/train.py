"""Run one configured closed-loop PPO training profile."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from eco_planner.rl import parse_training_config, train


@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/train_ppo_smoke")
def main(config: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    parsed = parse_training_config(config)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)
    summary = train(parsed, output_dir)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
