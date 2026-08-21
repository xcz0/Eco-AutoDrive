"""Run fixed-slot rollout and PPO update throughput benchmarks."""

import hydra
from benchmarking.rollout import run
from omegaconf import DictConfig


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="experiment/benchmark_rollout",
)
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
