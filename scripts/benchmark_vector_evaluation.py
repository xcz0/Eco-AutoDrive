"""Run one vector evaluation with explicit worker-pool and batch-fill profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from eco_planner.envs import VectorMetaDriveEnv
from eco_planner.evaluation import episode
from eco_planner.evaluation.artifacts.io import write_json
from eco_planner.evaluation.config import parse_evaluation_config
from eco_planner.evaluation.runner import run_evaluation


@dataclass
class _VectorProfile:
    pool_initialization_s: list[float] = field(default_factory=list)
    pool_teardown_s: list[float] = field(default_factory=list)
    reset_s: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    batch_wall_s: list[float] = field(default_factory=list)
    worker_wait_s: list[float] = field(default_factory=list)
    worker_busy_s: list[float] = field(default_factory=list)
    worker_imbalance_s: list[float] = field(default_factory=list)


class _ProfiledVectorMetaDriveEnv:
    """Delegate to the real vector environment while collecting benchmark-only timings."""

    def __init__(self, *args: object, profile: _VectorProfile, **kwargs: object) -> None:
        started = perf_counter()
        self._env = VectorMetaDriveEnv(*args, **kwargs)
        profile.pool_initialization_s.append(perf_counter() - started)
        self._profile = profile

    def __enter__(self) -> _ProfiledVectorMetaDriveEnv:
        self._env.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        started = perf_counter()
        self._env.__exit__(*args)
        self._profile.pool_teardown_s.append(perf_counter() - started)

    def reset(self, scenarios: object) -> object:
        started = perf_counter()
        result = self._env.reset(scenarios)
        self._profile.reset_s.append(perf_counter() - started)
        return result

    def reset_at(self, slot: int, scenario: object) -> object:
        started = perf_counter()
        result = self._env.reset_at(slot, scenario)
        self._profile.reset_s.append(perf_counter() - started)
        return result

    def step_slots(self, slots: object, trajectories: object) -> object:
        started = perf_counter()
        result = self._env.step_slots(slots, trajectories)
        self._profile.batch_wall_s.append(perf_counter() - started)
        self._profile.batch_sizes.append(len(result))
        busy = [item.timing.environment_s + item.timing.observation_s for item in result]
        self._profile.worker_wait_s.extend(item.timing.worker_wait_s for item in result)
        self._profile.worker_busy_s.extend(busy)
        slowest = max(busy)
        self._profile.worker_imbalance_s.append(sum(slowest - value for value in busy))
        return result


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="experiment/benchmark_vector_refill",
)
def main(config: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    profile = _VectorProfile()
    original = episode.VectorMetaDriveEnv
    episode.VectorMetaDriveEnv = lambda *args, **kwargs: _ProfiledVectorMetaDriveEnv(
        *args, profile=profile, **kwargs
    )
    try:
        run_evaluation(parse_evaluation_config(config), output_dir)
    finally:
        episode.VectorMetaDriveEnv = original
    write_json(output_dir / "vector_benchmark.json", asdict(profile))


if __name__ == "__main__":
    main()
