"""Project-facing vector MetaDrive environment backed by TorchRL ``ParallelEnv``."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs import ParallelEnv

from eco_planner.envs.array_types import SingleObservation
from eco_planner.envs.metadrive.execution import TrajectoryExecutionRecord
from eco_planner.envs.metadrive.observation import TrafficObservationAudit
from eco_planner.envs.metadrive.reward import RewardProfileConfig
from eco_planner.envs.metadrive.slot import MetaDriveEnvSlot, ObservationMode
from eco_planner.envs.observation import PlannerObservationSpec
from eco_planner.envs.torchrl.adapter import TorchRLMetaDriveEnv
from eco_planner.execution_contracts import PLANNER_FUTURE_STEPS

_OBSERVATION_KEYS = (
    "ego_current_state",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
)


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
class VectorEnvReset:
    """One slot observation and its reset-domain metadata."""

    slot: int
    scenario: VectorEnvScenario
    observation: SingleObservation
    route_completion: float
    route_length_m: float
    warmup_initial_state: np.ndarray
    initial_state: np.ndarray
    warmup_executions: tuple[TrajectoryExecutionRecord, ...]
    traffic_audit: TrafficObservationAudit | None
    programmatic_lane_speed_limit_audit: Mapping[str, object]
    timing: VectorEnvTiming


@dataclass(frozen=True, slots=True)
class VectorEnvStep:
    """One slot transition and its execution-domain metadata."""

    slot: int
    observation: SingleObservation
    reward: float
    terminated: bool
    truncated: bool
    execution: TrajectoryExecutionRecord
    traffic_audit: TrafficObservationAudit | None
    timing: VectorEnvTiming


@dataclass(frozen=True, slots=True)
class _WorkerResetResult:
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
class _WorkerStepResult:
    execution: TrajectoryExecutionRecord
    traffic_audit: TrafficObservationAudit | None
    timing: VectorEnvTiming


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    operation: str
    traceback_text: str


class VectorMetaDriveWorkerError(RuntimeError):
    """A TorchRL-owned MetaDrive worker failed during one explicit operation."""


class _TorchRLScenarioMetaDriveEnv(TorchRLMetaDriveEnv):
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
        self._operation_result: _WorkerResetResult | _WorkerStepResult | _WorkerFailure | None = (
            None
        )
        self.state_spec = Composite(
            scenario_index=Unbounded(
                shape=torch.Size((1,)), dtype=torch.int64, device=torch.device("cpu")
            ),
            shape=(),
            device=torch.device("cpu"),
        )

    def operation_result(self) -> _WorkerResetResult | _WorkerStepResult | _WorkerFailure | None:
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
            self._operation_result = _WorkerResetResult(
                scenario=self._scenarios[self._active_scenario_index],
                route_completion=reset.route_completion,
                route_length_m=reset.route_length_m,
                warmup_initial_state=reset.warmup_initial_state,
                initial_state=self.last_initial_state,
                warmup_executions=self.last_warmup_executions,
                traffic_audit=self.last_traffic_audit,
                programmatic_lane_speed_limit_audit=(reset.programmatic_lane_speed_limit_audit),
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
            self._operation_result = _WorkerStepResult(
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
        self._operation_result = _WorkerFailure(
            operation=operation,
            traceback_text=traceback.format_exc(),
        )
        return output


class VectorMetaDriveEnv:
    """Operate fixed MetaDrive slots through a TorchRL-owned process pool."""

    def __init__(
        self,
        env_configs: Sequence[Mapping[str, Any]],
        *,
        mode: ObservationMode,
        observation_spec: PlannerObservationSpec,
        map_query_radius_m: float,
        history_warmup_steps: int,
        scenarios: Sequence[VectorEnvScenario],
        torch_threads_per_worker: int | None = None,
        reward_profile: RewardProfileConfig | None = None,
    ) -> None:
        _validate_configuration(
            env_configs,
            mode=mode,
            observation_spec=observation_spec,
            map_query_radius_m=map_query_radius_m,
            history_warmup_steps=history_warmup_steps,
        )
        scenario_catalog = tuple(scenarios)
        if not scenario_catalog:
            raise ValueError("VectorMetaDriveEnv scenarios must be non-empty")
        if not all(isinstance(item, VectorEnvScenario) for item in scenario_catalog):
            raise TypeError("scenarios must contain only VectorEnvScenario values")
        if len(set(scenario_catalog)) != len(scenario_catalog):
            raise ValueError("VectorMetaDriveEnv scenario catalog must not contain duplicates")
        if torch_threads_per_worker is None:
            torch_threads_per_worker = torch.get_num_threads()
        if type(torch_threads_per_worker) is not int or torch_threads_per_worker <= 0:
            raise ValueError("torch_threads_per_worker must be a positive integer")

        _validate_shared_environment_configuration(env_configs)
        self._num_envs = len(env_configs)
        self._scenarios = scenario_catalog
        self._scenario_indices = {scenario: index for index, scenario in enumerate(self._scenarios)}
        self._active_scenario_indices = [0] * self._num_envs
        self._operation_lock = RLock()
        self._closed = False
        factory = partial(
            _make_torchrl_scenario_env,
            dict(env_configs[0]),
            mode,
            observation_spec,
            float(map_query_radius_m),
            history_warmup_steps,
            self._scenarios,
            reward_profile,
        )
        self._env = ParallelEnv(
            self._num_envs,
            factory,
            device="cpu",
            mp_start_method="spawn",
            shared_memory=True,
            use_buffers=True,
            serial_for_single=False,
            num_threads=torch.get_num_threads(),
            num_sub_threads=torch_threads_per_worker,
        )

    @property
    def num_envs(self) -> int:
        """Return the fixed number of physical worker slots."""

        return self._num_envs

    @property
    def uses_shared_buffers(self) -> bool:
        """Report whether the pinned TorchRL backend selected its shared-buffer path."""

        return bool(self._env._use_buffers)  # noqa: SLF001 - pinned TorchRL contract

    def reset(self, scenarios: Sequence[VectorEnvScenario]) -> tuple[VectorEnvReset, ...]:
        """Reset every physical slot to its assigned catalog scenario."""

        if len(scenarios) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} scenarios, got {len(scenarios)}")
        slots = tuple(range(self.num_envs))
        return self._reset_slots(slots, scenarios, partial=False)

    def reset_at(self, slot: int, scenario: VectorEnvScenario) -> VectorEnvReset:
        """Reset one slot without changing another worker's episode state."""

        self._validate_slot(slot)
        return self._reset_slots((slot,), (scenario,), partial=True)[0]

    def step(
        self, trajectories: Sequence[np.ndarray] | np.ndarray
    ) -> tuple[VectorEnvStep, ...]:
        """Step every physical slot once."""

        return self.step_slots(tuple(range(self.num_envs)), trajectories)

    def step_slots(
        self, slots: Sequence[int], trajectories: Sequence[np.ndarray] | np.ndarray
    ) -> tuple[VectorEnvStep, ...]:
        """Let TorchRL route one partial step to the selected physical slots."""

        slots_tuple = tuple(slots)
        if not slots_tuple or len(slots_tuple) != len(trajectories):
            raise ValueError("step_slots requires equally sized non-empty slots and trajectories")
        if len(set(slots_tuple)) != len(slots_tuple):
            raise ValueError("step_slots must not repeat a slot")
        for slot in slots_tuple:
            self._validate_slot(slot)
        actions = torch.zeros((self.num_envs, PLANNER_FUTURE_STEPS, 4), dtype=torch.float32)
        for slot, trajectory in zip(slots_tuple, trajectories, strict=True):
            array = np.asarray(trajectory)
            if array.shape != (PLANNER_FUTURE_STEPS, 4) or array.dtype != np.float32:
                raise ValueError("trajectory must have shape [80, 4] and dtype float32")
            actions[slot].copy_(torch.from_numpy(array))
        mask = _slot_mask(self.num_envs, slots_tuple)
        input_td = TensorDict({"action": actions, "_step": mask}, batch_size=[self.num_envs])
        with self._operation_lock:
            output = _tensordict_field(
                self._call("step", lambda: self._env.step(input_td)), "next"
            )
            domains = self._env.operation_result()
            return tuple(self._step_result(output, slot, domains[slot]) for slot in slots_tuple)

    def step_at(self, slot: int, trajectory: np.ndarray) -> VectorEnvStep:
        """Step one physical slot without advancing the others."""

        self._validate_slot(slot)
        return self.step_slots((slot,), (trajectory,))[0]

    def close(self) -> None:
        """Idempotently delegate pool shutdown and termination to TorchRL."""

        with self._operation_lock:
            if not self._closed:
                try:
                    self._env.close(raise_if_closed=False)
                finally:
                    self._closed = True

    def __enter__(self) -> VectorMetaDriveEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _reset_slots(
        self,
        slots: tuple[int, ...],
        scenarios: Sequence[VectorEnvScenario],
        *,
        partial: bool,
    ) -> tuple[VectorEnvReset, ...]:
        indices = list(self._active_scenario_indices)
        expected: dict[int, VectorEnvScenario] = {}
        for slot, scenario in zip(slots, scenarios, strict=True):
            if not isinstance(scenario, VectorEnvScenario):
                raise TypeError("scenarios must contain only VectorEnvScenario values")
            try:
                index = self._scenario_indices[scenario]
            except KeyError as error:
                raise ValueError(
                    f"scenario {scenario.name!r} is absent from the catalog"
                ) from error
            indices[slot] = index
            expected[slot] = scenario
        values: dict[str, torch.Tensor] = {
            "scenario_index": torch.tensor(indices, dtype=torch.int64).unsqueeze(-1)
        }
        if partial:
            values["_reset"] = _slot_mask(self.num_envs, slots)
        input_td = TensorDict(values, batch_size=[self.num_envs])
        with self._operation_lock:
            output = self._call("reset", lambda: self._env.reset(input_td))
            domains = self._env.operation_result()
            results = tuple(
                self._reset_result(output, slot, expected[slot], domains[slot]) for slot in slots
            )
            self._active_scenario_indices = indices
            return results

    def _call(self, operation: str, function: Callable[[], TensorDictBase]) -> TensorDictBase:
        if self._closed:
            raise RuntimeError("VectorMetaDriveEnv is closed")
        try:
            return function()
        except RuntimeError as error:
            self.close()
            raise VectorMetaDriveWorkerError(
                f"TorchRL ParallelEnv failed during {operation}: {error}"
            ) from error
        except BaseException:
            self.close()
            raise

    def _reset_result(
        self,
        output: TensorDictBase,
        slot: int,
        scenario: VectorEnvScenario,
        domain: object,
    ) -> VectorEnvReset:
        self._raise_worker_failure(slot, domain)
        if not isinstance(domain, _WorkerResetResult):
            self.close()
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {slot} returned invalid reset domain data"
            )
        if domain.scenario != scenario:
            self.close()
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {slot} returned the wrong reset scenario"
            )
        return VectorEnvReset(
            slot=slot,
            scenario=domain.scenario,
            observation=_observation(output, slot),
            route_completion=domain.route_completion,
            route_length_m=domain.route_length_m,
            warmup_initial_state=domain.warmup_initial_state,
            initial_state=domain.initial_state,
            warmup_executions=domain.warmup_executions,
            traffic_audit=domain.traffic_audit,
            programmatic_lane_speed_limit_audit=(domain.programmatic_lane_speed_limit_audit),
            timing=domain.timing,
        )

    def _step_result(self, output: TensorDictBase, slot: int, domain: object) -> VectorEnvStep:
        self._raise_worker_failure(slot, domain)
        if not isinstance(domain, _WorkerStepResult):
            self.close()
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {slot} returned invalid step domain data"
            )
        terminated = bool(output["terminated"][slot].item())
        truncated = bool(output["truncated"][slot].item())
        return VectorEnvStep(
            slot=slot,
            observation=_observation(output, slot),
            reward=float(output["reward"][slot].item()),
            terminated=terminated,
            truncated=truncated,
            execution=domain.execution,
            traffic_audit=domain.traffic_audit,
            timing=domain.timing,
        )

    def _raise_worker_failure(self, slot: int, domain: object) -> None:
        if isinstance(domain, _WorkerFailure):
            self.close()
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {slot} failed during {domain.operation}:\n"
                f"{domain.traceback_text}"
            )

    def _validate_slot(self, slot: int) -> None:
        if type(slot) is not int or not 0 <= slot < self.num_envs:
            raise IndexError(f"slot must be in [0, {self.num_envs})")


def _make_torchrl_scenario_env(
    env_config: Mapping[str, Any],
    mode: ObservationMode,
    observation_spec: PlannerObservationSpec,
    map_query_radius_m: float,
    history_warmup_steps: int,
    scenarios: tuple[VectorEnvScenario, ...],
    reward_profile: RewardProfileConfig | None,
) -> _TorchRLScenarioMetaDriveEnv:
    scenario = scenarios[0]
    slot = MetaDriveEnvSlot(
        {**env_config, "map": scenario.map},
        mode=mode,
        observation_spec=observation_spec,
        map_query_radius_m=map_query_radius_m,
        history_warmup_steps=history_warmup_steps,
        reward_profile=reward_profile,
    )
    return _TorchRLScenarioMetaDriveEnv(
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


def _slot_mask(count: int, slots: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros((count, 1), dtype=torch.bool)
    mask[list(slots)] = True
    return mask


def _observation(output: TensorDictBase, slot: int) -> SingleObservation:
    values = {
        key: _tensor_field(output, key)[slot].detach().clone() for key in _OBSERVATION_KEYS
    }
    return cast(SingleObservation, values)


def _tensor_field(output: TensorDictBase, key: str) -> torch.Tensor:
    value = output.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"TensorDict field {key!r} must be a tensor")
    return value


def _tensordict_field(output: TensorDictBase, key: str) -> TensorDictBase:
    value = output.get(key)
    if not isinstance(value, TensorDictBase):
        raise TypeError(f"TensorDict field {key!r} must be a nested TensorDict")
    return value


def _validate_configuration(
    env_configs: Sequence[Mapping[str, Any]],
    *,
    mode: ObservationMode,
    observation_spec: PlannerObservationSpec,
    map_query_radius_m: float,
    history_warmup_steps: int,
) -> None:
    if not env_configs:
        raise ValueError("VectorMetaDriveEnv requires at least one environment slot")
    if mode not in {"traffic", "no_traffic"}:
        raise ValueError("mode must be either 'traffic' or 'no_traffic'")
    if not isinstance(observation_spec, PlannerObservationSpec):
        raise TypeError("observation_spec must be a PlannerObservationSpec")
    if type(map_query_radius_m) not in {int, float} or map_query_radius_m <= 0.0:
        raise ValueError("map_query_radius_m must be a positive real scalar")
    if type(history_warmup_steps) is not int or history_warmup_steps < 0:
        raise ValueError("history_warmup_steps must be a non-negative integer")
    expected_warmup = observation_spec.time_len - 1 if mode == "traffic" else 0
    if history_warmup_steps != expected_warmup:
        raise ValueError(
            f"{mode} vector environments require history_warmup_steps={expected_warmup}"
        )


def _validate_shared_environment_configuration(
    env_configs: Sequence[Mapping[str, Any]],
) -> None:
    expected = {key: value for key, value in env_configs[0].items() if key != "map"}
    for config in env_configs[1:]:
        actual = {key: value for key, value in config.items() if key != "map"}
        if actual != expected:
            raise ValueError("all vector workers must share one non-map environment configuration")
