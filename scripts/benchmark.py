"""Run a configured repository-only benchmark profile."""

from __future__ import annotations

import sys
from importlib import import_module

import hydra
from omegaconf import DictConfig, OmegaConf

from eco_planner._repository import LOCAL_ENVIRONMENT_PATH
from eco_planner.configuration import load_local_environment, with_machine_resource_override

_BENCHMARK_MODULES = {
    "environment": "eco_planner.benchmarking.environment",
    "throughput": "eco_planner.benchmarking.throughput",
    "rollout": "eco_planner.benchmarking.rollout",
}


def _benchmark_module(kind: object) -> str:
    if not isinstance(kind, str) or kind not in _BENCHMARK_MODULES:
        raise ValueError(f"unsupported benchmark kind: {kind!r}")
    return _BENCHMARK_MODULES[kind]


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="jobs/benchmark/throughput",
)
def _hydra_main(config: DictConfig) -> None:
    module = import_module(_benchmark_module(OmegaConf.select(config, "benchmark.kind")))
    module.run(config)


def main() -> None:
    load_local_environment(LOCAL_ENVIRONMENT_PATH)
    sys.argv[:] = with_machine_resource_override(sys.argv)
    _hydra_main()


if __name__ == "__main__":
    main()
