"""TorchRL ``ParallelEnv`` proof of concept over the existing MetaDrive slot lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs import ParallelEnv

from eco_planner.envs.observation import PlannerObservationSpec
from eco_planner.envs.slot import MetaDriveEnvSlot, ObservationMode
from eco_planner.envs.torchrl_env import TorchRLMetaDriveEnv


@dataclass(frozen=True, slots=True)
class TorchRLParallelScenario:
    """One fixed scenario selectable by integer index at the TorchRL reset boundary."""

    map_name: str
    seed: int


class TorchRLScenarioMetaDriveEnv(TorchRLMetaDriveEnv):
    """Select a predefined map/seed pair before delegating lifecycle work to one slot."""

    def __init__(
        self,
        *args: object,
        scenarios: tuple[TorchRLParallelScenario, ...],
        **kwargs: object,
    ) -> None:
        if not scenarios:
            raise ValueError("TorchRL ParallelEnv scenarios must be non-empty")
        if not all(isinstance(item, TorchRLParallelScenario) for item in scenarios):
            raise TypeError("TorchRL ParallelEnv scenarios must be TorchRLParallelScenario values")
        super().__init__(*args, **kwargs)
        self._scenarios = scenarios
        self._active_scenario_index = 0
        self.state_spec = Composite(
            scenario_index=Unbounded(shape=(1,), dtype=torch.int64, device="cpu"),
            shape=(),
            device="cpu",
        )

    @property
    def active_scenario_index(self) -> int:
        """Return the scenario selected for the current worker episode."""

        return self._active_scenario_index

    def current_scenario_index(self) -> int:
        """Return the active scenario through ParallelEnv's remote-method boundary."""

        return self.active_scenario_index

    def execution_record(self):
        """Return the latest execution audit through ParallelEnv's remote-method boundary."""

        return self.last_execution

    def warmup_execution_records(self):
        """Return reset warmup records through ParallelEnv's remote-method boundary."""

        return self.last_warmup_executions

    def traffic_observation_audit(self):
        """Return the latest traffic audit through ParallelEnv's remote-method boundary."""

        return self.last_traffic_audit

    def _reset(self, tensordict: TensorDictBase | None, **kwargs: object) -> TensorDictBase:
        del kwargs
        if tensordict is not None and "scenario_index" in tensordict:
            index = _scenario_index(tensordict["scenario_index"], len(self._scenarios))
            self._active_scenario_index = index
            scenario = self._scenarios[index]
            self._map_name = scenario.map_name
            self._seed = scenario.seed
        output = super()._reset(None)
        output["scenario_index"] = torch.tensor([self._active_scenario_index], dtype=torch.int64)
        return output


def create_torchrl_parallel_env_poc(
    worker_count: int,
    env_config: Mapping[str, Any],
    *,
    mode: ObservationMode,
    observation_spec: PlannerObservationSpec,
    map_query_radius_m: float,
    history_warmup_steps: int,
    scenarios: Sequence[TorchRLParallelScenario],
) -> ParallelEnv:
    """Create a spawn-based fixed-worker ``ParallelEnv`` PoC.

    Scenario IDs are intentionally scalar tensors so that TorchRL can route a partial reset through
    its standard shared-memory contract.  Each worker continues to use ``MetaDriveEnvSlot`` for
    map replacement, warmup, traffic history and execution auditing.
    """

    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("worker_count must be a positive integer")
    scenarios_tuple = tuple(scenarios)
    if not scenarios_tuple:
        raise ValueError("scenarios must be non-empty")
    factory = partial(
        _make_torchrl_scenario_env,
        dict(env_config),
        mode,
        observation_spec,
        map_query_radius_m,
        history_warmup_steps,
        scenarios_tuple,
    )
    return ParallelEnv(
        worker_count,
        factory,
        device="cpu",
        mp_start_method="spawn",
    )


def _make_torchrl_scenario_env(
    env_config: Mapping[str, Any],
    mode: ObservationMode,
    observation_spec: PlannerObservationSpec,
    map_query_radius_m: float,
    history_warmup_steps: int,
    scenarios: tuple[TorchRLParallelScenario, ...],
) -> TorchRLScenarioMetaDriveEnv:
    scenario = scenarios[0]
    slot = MetaDriveEnvSlot(
        {**env_config, "map": scenario.map_name},
        mode=mode,
        observation_spec=observation_spec,
        map_query_radius_m=map_query_radius_m,
        history_warmup_steps=history_warmup_steps,
    )
    return TorchRLScenarioMetaDriveEnv(
        slot,
        map_name=scenario.map_name,
        seed=scenario.seed,
        observation_spec=observation_spec,
        scenarios=scenarios,
    )


def _scenario_index(value: torch.Tensor, count: int) -> int:
    if value.dtype != torch.int64 or value.numel() != 1:
        raise ValueError("scenario_index must be one int64 scalar")
    index = int(value.item())
    if not 0 <= index < count:
        raise ValueError(f"scenario_index must be in [0, {count})")
    return index
