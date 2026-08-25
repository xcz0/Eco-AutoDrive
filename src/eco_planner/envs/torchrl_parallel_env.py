"""TorchRL ``ParallelEnv`` proof of concept over the existing MetaDrive slot lifecycle."""

from __future__ import annotations

import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Any

import numpy as np
import torch
from tensordict import NonTensorData, TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs import ParallelEnv

from eco_planner.envs.execution import TrajectoryExecutionRecord
from eco_planner.envs.observation import PlannerObservationSpec
from eco_planner.envs.observation_adapter import TrafficObservationAudit
from eco_planner.envs.slot import MetaDriveEnvSlot, ObservationMode
from eco_planner.envs.torchrl_env import TorchRLMetaDriveEnv


@dataclass(frozen=True, slots=True)
class TorchRLParallelScenario:
    """One fixed scenario selectable by integer index at the TorchRL reset boundary."""

    map_name: str
    seed: int


@dataclass(frozen=True, slots=True)
class TorchRLParallelResetResult:
    """Domain metadata returned atomically with one ParallelEnv reset."""

    scenario: TorchRLParallelScenario
    route_completion: float
    route_length_m: float
    warmup_initial_state: np.ndarray
    initial_state: np.ndarray
    warmup_executions: tuple[TrajectoryExecutionRecord, ...]
    traffic_audit: TrafficObservationAudit | None
    programmatic_lane_speed_limit_audit: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TorchRLParallelStepResult:
    """Domain audit returned atomically with one ParallelEnv step."""

    execution: TrajectoryExecutionRecord
    traffic_audit: TrafficObservationAudit | None


@dataclass(frozen=True, slots=True)
class _TorchRLWorkerFailure:
    operation: str
    traceback_text: str


class TorchRLParallelWorkerError(RuntimeError):
    """A ParallelEnv slot failed while executing one environment operation."""


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
        self._worker_failure: _TorchRLWorkerFailure | None = None
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

    def worker_failure(self) -> _TorchRLWorkerFailure | None:
        """Return the latest caught operation failure without raising in the worker."""

        return self._worker_failure

    def reset_result(self) -> TorchRLParallelResetResult:
        """Return metadata produced by the latest successful reset."""

        reset = self.last_reset
        return TorchRLParallelResetResult(
            scenario=self._scenarios[self._active_scenario_index],
            route_completion=reset.route_completion,
            route_length_m=reset.route_length_m,
            warmup_initial_state=reset.warmup_initial_state,
            initial_state=self.last_initial_state,
            warmup_executions=self.last_warmup_executions,
            traffic_audit=self.last_traffic_audit,
            programmatic_lane_speed_limit_audit=reset.programmatic_lane_speed_limit_audit,
        )

    def step_result(self) -> TorchRLParallelStepResult:
        """Return the domain audit produced by the latest successful step."""

        return TorchRLParallelStepResult(
            execution=self.last_step.execution,
            traffic_audit=self.last_traffic_audit,
        )

    def _reset(self, tensordict: TensorDictBase | None, **kwargs: object) -> TensorDictBase:
        del kwargs
        try:
            if tensordict is not None and "scenario_index" in tensordict:
                index = _scenario_index(tensordict["scenario_index"], len(self._scenarios))
                self._active_scenario_index = index
                scenario = self._scenarios[index]
                self._map_name = scenario.map_name
                self._seed = scenario.seed
            output = super()._reset(None)
            output["scenario_index"] = torch.tensor(
                [self._active_scenario_index], dtype=torch.int64
            )
            self._worker_failure = None
            return output
        except BaseException:
            return self._failure_output("reset")

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        try:
            output = super()._step(tensordict)
            self._worker_failure = None
            return output
        except BaseException:
            return self._failure_output("step")

    def _failure_output(self, operation: str) -> TensorDictBase:
        output = self.observation_spec.zero()
        output.update(self.done_spec.zero())
        if operation == "step":
            output.update(self.reward_spec.zero())
        output["scenario_index"] = torch.tensor(
            [self._active_scenario_index], dtype=torch.int64
        )
        self._worker_failure = _TorchRLWorkerFailure(
            operation=operation,
            traceback_text=traceback.format_exc(),
        )
        return output


class TorchRLParallelMetaDriveEnv:
    """Public PoC facade that restores explicit worker-failure semantics."""

    def __init__(self, env: ParallelEnv) -> None:
        self._env = env
        self._closed = False
        self._operation_lock = RLock()

    @property
    def BATCHED_PIPE_TIMEOUT(self) -> float:
        return self._env.BATCHED_PIPE_TIMEOUT

    @BATCHED_PIPE_TIMEOUT.setter
    def BATCHED_PIPE_TIMEOUT(self, value: float) -> None:
        self._env.BATCHED_PIPE_TIMEOUT = value

    def reset(self, tensordict: TensorDictBase | None = None, **kwargs: object) -> TensorDictBase:
        if tensordict is not None and tensordict.device is None:
            tensordict = tensordict.to("cpu")
        with self._operation_lock:
            try:
                output = self._env.reset(tensordict, **kwargs)
            except RuntimeError as error:
                self.close()
                raise TorchRLParallelWorkerError(
                    f"TorchRL ParallelEnv failed during reset: {error}"
                ) from error
            self._raise_worker_failure()
            _set_non_tensor_batch(output, "reset_result", self._env.reset_result())
            return output

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if tensordict.device is None:
            tensordict = tensordict.to("cpu")
        with self._operation_lock:
            try:
                output = self._env.step(tensordict)
            except RuntimeError as error:
                self.close()
                raise TorchRLParallelWorkerError(
                    f"TorchRL ParallelEnv failed during step: {error}"
                ) from error
            self._raise_worker_failure()
            _set_non_tensor_batch(output["next"], "step_result", self._env.step_result())
            return output

    def current_scenario_index(self):
        return self._env.current_scenario_index()

    def execution_record(self):
        return self._env.execution_record()

    def warmup_execution_records(self):
        return self._env.warmup_execution_records()

    def traffic_observation_audit(self):
        return self._env.traffic_observation_audit()

    def close(self) -> None:
        if not self._closed:
            try:
                self._env.close()
            finally:
                self._closed = True

    def _raise_worker_failure(self) -> None:
        for slot, failure in enumerate(self._env.worker_failure()):
            if isinstance(failure, _TorchRLWorkerFailure):
                self.close()
                raise TorchRLParallelWorkerError(
                    f"TorchRL ParallelEnv slot {slot} failed during {failure.operation}:\n"
                    f"{failure.traceback_text}"
                )


def create_torchrl_parallel_env_poc(
    worker_count: int,
    env_config: Mapping[str, Any],
    *,
    mode: ObservationMode,
    observation_spec: PlannerObservationSpec,
    map_query_radius_m: float,
    history_warmup_steps: int,
    scenarios: Sequence[TorchRLParallelScenario],
) -> TorchRLParallelMetaDriveEnv:
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
    return TorchRLParallelMetaDriveEnv(
        ParallelEnv(
            worker_count,
            factory,
            device="cpu",
            mp_start_method="spawn",
        )
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


def _set_non_tensor_batch(
    output: TensorDictBase,
    key: str,
    values: Sequence[object],
) -> None:
    output.set(
        key,
        torch.stack(
            [NonTensorData(value, batch_size=[]) for value in values],
            dim=0,
        ),
    )
