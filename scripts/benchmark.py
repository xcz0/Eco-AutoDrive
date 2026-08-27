"""Run a configured repository-only benchmark profile."""

from __future__ import annotations

from importlib import import_module

import hydra
from omegaconf import DictConfig, OmegaConf

from eco_planner.configuration import load_local_environment
from scripts._paths import LOCAL_ENVIRONMENT_PATH

_BENCHMARK_MODULES = {
    "environment": "scripts.benchmarking.environment",
    "throughput": "scripts.benchmarking.throughput",
    "rollout": "scripts.benchmarking.rollout",
}

load_local_environment(LOCAL_ENVIRONMENT_PATH)


def _benchmark_module(kind: object) -> str:
    if not isinstance(kind, str) or kind not in _BENCHMARK_MODULES:
        raise ValueError(f"unsupported benchmark kind: {kind!r}")
    return _BENCHMARK_MODULES[kind]


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="jobs/benchmark/throughput",
)
def main(config: DictConfig) -> None:
    module = import_module(_benchmark_module(OmegaConf.select(config, "benchmark.kind")))
    module.run(config)


if __name__ == "__main__":
    main()
