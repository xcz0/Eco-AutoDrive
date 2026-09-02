"""TorchRL worker implementation used by the parent vector-environment façade."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded

from eco_planner.envs.domain.metrics import TransitionMetrics
from eco_planner.envs.metadrive.config import MetaDriveBuiltinRewardConfig
from eco_planner.envs.metadrive.execution import TrajectoryExecutionRecord
from eco_planner.envs.metadrive.slot import MetaDriveEnvSlot, ObservationMode
from eco_planner.envs.observation import PlannerObservationSpec, TrafficObservationAudit
from eco_planner.envs.torchrl.adapter import TorchRLMetaDriveEnv


@dataclass(frozen=True, slots=True)
class VectorEnvScenario:
    """The immutable scenario identity assigned to one physical environment slot."""

    name: str
    map: str
    seed: int


@dataclass(frozen=True, slots=True)
class VectorEnvTiming:
    """Worker service timings for one reset or step result."""

    environment_s: float
    observation_s: float


@dataclass(frozen=True, slots=True)
class WorkerResetResult:
    scenario: VectorEnvScenario
    route_completion: float
    route_length_m: float
    warmup_initial_state: np.ndarray
    initial_state: np.ndarray
    warmup_executions: tuple[TrajectoryExecutionRecord, ...]
    traffic_audit: TrafficObservationAudit | None
    programmatic_lane_speed_limit_audit: Mapping[str, object]
    timing: VectorEnvTiming


@dataclass(frozen=True, slots=True)
class WorkerStepResult:
    execution: TrajectoryExecutionRecord
    traffic_audit: TrafficObservationAudit | None
    timing: VectorEnvTiming


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    operation: str
    traceback_text: str


class TorchRLScenarioMetaDriveEnv(TorchRLMetaDriveEnv):
    """Select a catalog scenario and attach one domain result to every operation."""

    def __init__(
        self,
        slot: MetaDriveEnvSlot,
        *,
        map_name: str,
        seed: int,
        observation_spec: PlannerObservationSpec,
        scenarios: tuple[VectorEnvScenario, ...],
    ) -> None:
        if not scenarios:
            raise ValueError("vector environment scenarios must be non-empty")
        super().__init__(
            slot,
            map_name=map_name,
            seed=seed,
            observation_spec=observation_spec,
        )
        self._scenarios = scenarios
        self._active_scenario_index = 0
        self._operation_result: WorkerResetResult | WorkerStepResult | WorkerFailure | None = None
        self.state_spec = Composite(
            scenario_index=Unbounded(
                shape=torch.Size((1,)), dtype=torch.int64, device=torch.device("cpu")
            ),
            shape=(),
            device=torch.device("cpu"),
        )

    def operation_result(self) -> WorkerResetResult | WorkerStepResult | WorkerFailure | None:
        """Return the latest domain result through TorchRL's remote-method channel."""

        return self._operation_result

    def _reset(self, tensordict: TensorDictBase | None, **kwargs: object) -> TensorDictBase:
        del kwargs
        try:
            if tensordict is not None and "scenario_index" in tensordict:
                index = _scenario_index(tensordict["scenario_index"], len(self._scenarios))
                self._active_scenario_index = index
                scenario = self._scenarios[index]
                self._map_name = scenario.map
                self._seed = scenario.seed
            output = super()._reset(None)
            reset = self.last_reset
            self._operation_result = WorkerResetResult(
                scenario=self._scenarios[self._active_scenario_index],
                route_completion=reset.route_completion,
                route_length_m=reset.route_length_m,
                warmup_initial_state=reset.warmup_initial_state,
                initial_state=self.last_initial_state,
                warmup_executions=self.last_warmup_executions,
                traffic_audit=self.last_traffic_audit,
                programmatic_lane_speed_limit_audit=reset.programmatic_lane_speed_limit_audit,
                timing=VectorEnvTiming(
                    environment_s=self.last_environment_s,
                    observation_s=self.last_observation_s,
                ),
            )
            return output
        except BaseException:
            return self._failure_output("reset")

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        try:
            output = super()._step(tensordict)
            self._operation_result = WorkerStepResult(
                execution=self.last_step.execution,
                traffic_audit=self.last_traffic_audit,
                timing=VectorEnvTiming(
                    environment_s=self.last_environment_s,
                    observation_s=self.last_observation_s,
                ),
            )
            return output
        except BaseException:
            return self._failure_output("step")

    def _failure_output(self, operation: str) -> TensorDictBase:
        output = self.observation_spec.zero()
        output.update(self.done_spec.zero())
        if operation == "step":
            output["reward"] = self.reward_spec.zero()
        self._operation_result = WorkerFailure(
            operation=operation,
            traceback_text=traceback.format_exc(),
        )
        return output


def make_torchrl_scenario_env(
    env_config: Mapping[str, Any],
    mode: ObservationMode,
    observation_spec: PlannerObservationSpec,
    map_query_radius_m: float,
    history_warmup_steps: int,
    scenarios: tuple[VectorEnvScenario, ...],
    builtin_reward_config: MetaDriveBuiltinRewardConfig | None,
    reward_objective: Callable[[TransitionMetrics], tuple[float, float]] | None,
) -> TorchRLScenarioMetaDriveEnv:
    scenario = scenarios[0]
    slot = MetaDriveEnvSlot(
        {**env_config, "map": scenario.map},
        mode=mode,
        observation_spec=observation_spec,
        map_query_radius_m=map_query_radius_m,
        history_warmup_steps=history_warmup_steps,
        builtin_reward_config=builtin_reward_config,
        reward_objective=reward_objective,
    )
    return TorchRLScenarioMetaDriveEnv(
        slot,
        map_name=scenario.map,
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
