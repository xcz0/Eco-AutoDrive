"""Project-facing vector MetaDrive environment backed by TorchRL ``ParallelEnv``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from threading import RLock
from typing import Any, TypeVar, cast

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import ParallelEnv

from eco_planner.contracts import (
    PLANNER_HORIZON,
    TRAFFIC_HISTORY_WARMUP_STEPS,
    ExecutionMode,
)
from eco_planner.envs.metadrive import ObservationMode
from eco_planner.runtime.envs.worker import (
    VectorEnvScenario,
    WorkerFailure,
    WorkerResetResult,
    WorkerStepResult,
    make_torchrl_scenario_env,
)


class VectorMetaDriveWorkerError(RuntimeError):
    """A TorchRL-owned MetaDrive worker failed during one explicit operation."""


_OperationResultT = TypeVar("_OperationResultT")


def operation_results(
    output: TensorDictBase, expected_type: type[_OperationResultT]
) -> tuple[_OperationResultT, ...]:
    """Return the parent-side domain sidecars paired with one vector TensorDict batch."""

    value = output.get_non_tensor("operation_results")
    if not isinstance(value, tuple) or len(value) != output.batch_size[0]:
        raise TypeError("vector output operation_results must match its batch dimension")
    if not all(isinstance(item, expected_type) for item in value):
        raise TypeError(f"vector output contains non-{expected_type.__name__} operation results")
    return cast(tuple[_OperationResultT, ...], value)


class VectorMetaDriveEnv:
    """Operate fixed MetaDrive slots through a TorchRL-owned process pool."""

    def __init__(
        self,
        env_configs: Sequence[Mapping[str, Any]],
        *,
        mode: ObservationMode,
        execution_mode: ExecutionMode,
        map_query_radius_m: float,
        history_warmup_steps: int,
        scenarios: Sequence[VectorEnvScenario],
        torch_threads_per_worker: int | None = None,
    ) -> None:
        _validate_configuration(
            env_configs,
            mode=mode,
            execution_mode=execution_mode,
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
            make_torchrl_scenario_env,
            dict(env_configs[0]),
            mode,
            execution_mode,
            float(map_query_radius_m),
            history_warmup_steps,
            self._scenarios,
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

    def reset(
        self,
        scenarios: Sequence[VectorEnvScenario],
        *,
        slots: Sequence[int] | None = None,
    ) -> TensorDictBase:
        """Reset the requested physical slots and return one ordered TensorDict batch."""

        slots_tuple = self._resolve_slots(slots)
        if len(scenarios) != len(slots_tuple):
            raise ValueError(
                f"expected {len(slots_tuple)} scenarios for requested slots, got {len(scenarios)}"
            )
        return self._reset_slots(slots_tuple, scenarios)

    def step(
        self,
        trajectories: Sequence[np.ndarray] | np.ndarray,
        *,
        slots: Sequence[int] | None = None,
    ) -> TensorDictBase:
        """Step the requested physical slots and return one ordered TensorDict batch."""

        slots_tuple = self._resolve_slots(slots)
        arrays = np.asarray(trajectories)
        if arrays.shape != (len(slots_tuple), PLANNER_HORIZON, 4) or arrays.dtype != np.float32:
            raise ValueError(
                "trajectories must have shape "
                f"[{len(slots_tuple)}, {PLANNER_HORIZON}, 4] and dtype float32"
            )
        actions = torch.zeros((self.num_envs, PLANNER_HORIZON, 4), dtype=torch.float32)
        actions.index_copy_(
            0,
            torch.tensor(slots_tuple, dtype=torch.int64),
            torch.from_numpy(arrays),
        )
        mask = _slot_mask(self.num_envs, slots_tuple)
        input_td = TensorDict({"action": actions, "_step": mask}, batch_size=[self.num_envs])
        with self._operation_lock:
            output = _tensordict_field(self._call("step", lambda: self._env.step(input_td)), "next")
            domains = self._env.operation_result()
            selected_domains = tuple(self._step_result(slot, domains[slot]) for slot in slots_tuple)
            return _selected_output(output, slots_tuple).set_non_tensor(
                "operation_results", selected_domains
            )

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
    ) -> TensorDictBase:
        indices = list(self._active_scenario_indices)
        expected: dict[int, VectorEnvScenario] = {}
        for slot, scenario in zip(slots, scenarios, strict=True):
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
        if len(slots) != self.num_envs:
            values["_reset"] = _slot_mask(self.num_envs, slots)
        input_td = TensorDict(values, batch_size=[self.num_envs])
        with self._operation_lock:
            output = self._call("reset", lambda: self._env.reset(input_td))
            domains = self._env.operation_result()
            selected_domains = tuple(
                self._reset_result(slot, expected[slot], domains[slot]) for slot in slots
            )
            self._active_scenario_indices = indices
            return _selected_output(output, slots).set_non_tensor(
                "operation_results", selected_domains
            )

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
        slot: int,
        scenario: VectorEnvScenario,
        domain: object,
    ) -> WorkerResetResult:
        self._raise_worker_failure(slot, domain)
        if not isinstance(domain, WorkerResetResult):
            self.close()
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {slot} returned invalid reset domain data"
            )
        if domain.scenario != scenario:
            self.close()
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {slot} returned the wrong reset scenario"
            )
        return domain

    def _step_result(self, slot: int, domain: object) -> WorkerStepResult:
        self._raise_worker_failure(slot, domain)
        if not isinstance(domain, WorkerStepResult):
            self.close()
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {slot} returned invalid step domain data"
            )
        return domain

    def _raise_worker_failure(self, slot: int, domain: object) -> None:
        if isinstance(domain, WorkerFailure):
            self.close()
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {slot} failed during {domain.operation}:\n"
                f"{domain.traceback_text}"
            )

    def _validate_slot(self, slot: int) -> None:
        if type(slot) is not int or not 0 <= slot < self.num_envs:
            raise IndexError(f"slot must be in [0, {self.num_envs})")

    def _resolve_slots(self, slots: Sequence[int] | None) -> tuple[int, ...]:
        resolved = tuple(range(self.num_envs)) if slots is None else tuple(slots)
        if not resolved:
            raise ValueError("vector operation requires at least one slot")
        if len(set(resolved)) != len(resolved):
            raise ValueError("vector operation must not repeat a slot")
        for slot in resolved:
            self._validate_slot(slot)
        return resolved


def _slot_mask(count: int, slots: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros((count, 1), dtype=torch.bool)
    mask[list(slots)] = True
    return mask


def _selected_output(output: TensorDictBase, slots: Sequence[int]) -> TensorDictBase:
    return cast(TensorDictBase, output[list(slots)]).detach().clone()


def _tensordict_field(output: TensorDictBase, key: str) -> TensorDictBase:
    value = output.get(key)
    if not isinstance(value, TensorDictBase):
        raise TypeError(f"TensorDict field {key!r} must be a nested TensorDict")
    return value


def _validate_configuration(
    env_configs: Sequence[Mapping[str, Any]],
    *,
    mode: ObservationMode,
    execution_mode: ExecutionMode,
    map_query_radius_m: float,
    history_warmup_steps: int,
) -> None:
    if not env_configs:
        raise ValueError("VectorMetaDriveEnv requires at least one environment slot")
    if mode not in {"traffic", "no_traffic"}:
        raise ValueError("mode must be either 'traffic' or 'no_traffic'")
    if not isinstance(execution_mode, ExecutionMode):
        raise TypeError("execution_mode must be an ExecutionMode")
    if type(map_query_radius_m) not in {int, float} or map_query_radius_m <= 0.0:
        raise ValueError("map_query_radius_m must be a positive real scalar")
    if type(history_warmup_steps) is not int or history_warmup_steps < 0:
        raise ValueError("history_warmup_steps must be a non-negative integer")
    expected_warmup = TRAFFIC_HISTORY_WARMUP_STEPS if mode == "traffic" else 0
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
