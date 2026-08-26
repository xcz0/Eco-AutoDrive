"""Thin TorchRL structural adapter around one MetaDrive environment slot."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs import EnvBase

from eco_planner.envs.metadrive.execution import TrajectoryExecutionRecord
from eco_planner.envs.metadrive.slot import EnvSlotReset, EnvSlotStep, MetaDriveEnvSlot
from eco_planner.envs.observation import PlannerObservationSpec
from eco_planner.execution_contracts import PLANNER_FUTURE_STEPS

_CPU_DEVICE = torch.device("cpu")


class TorchRLMetaDriveEnv(EnvBase):
    """Expose one fixed MetaDrive slot through TorchRL's structural tensor contract."""

    def __init__(
        self,
        slot: MetaDriveEnvSlot,
        *,
        map_name: str,
        seed: int,
        observation_spec: PlannerObservationSpec,
    ) -> None:
        if not isinstance(observation_spec, PlannerObservationSpec):
            raise TypeError("observation_spec must be a PlannerObservationSpec")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        super().__init__(device=_CPU_DEVICE, batch_size=[])
        self._slot = slot
        self._map_name = map_name
        self._seed = seed
        self._last_reset: EnvSlotReset | None = None
        self._last_initial_state: np.ndarray | None = None
        self._last_step: EnvSlotStep | None = None
        self._last_execution: TrajectoryExecutionRecord | None = None
        self._last_warmup_executions: tuple[TrajectoryExecutionRecord, ...] = ()
        self._last_traffic_audit = None
        self._last_programmatic_lane_speed_limit_audit: Mapping[str, object] = {}
        self._last_environment_s = 0.0
        self._last_observation_s = 0.0
        self.observation_spec = _observation_spec(observation_spec)
        self.action_spec = Unbounded(
            shape=(PLANNER_FUTURE_STEPS, 4), dtype=torch.float32, device=_CPU_DEVICE
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=_CPU_DEVICE)
        self.done_spec = Composite(
            done=Binary(1, shape=(1,), dtype=torch.bool, device=_CPU_DEVICE),
            terminated=Binary(1, shape=(1,), dtype=torch.bool, device=_CPU_DEVICE),
            truncated=Binary(1, shape=(1,), dtype=torch.bool, device=_CPU_DEVICE),
            shape=(),
            device=_CPU_DEVICE,
        )

    def _reset(self, tensordict: TensorDictBase | None, **kwargs: object) -> TensorDictBase:
        del tensordict, kwargs
        environment_started = perf_counter()
        reset = self._slot.reset(map_name=self._map_name, seed=self._seed)
        self._last_reset = reset
        self._last_warmup_executions = tuple(self._slot.warmup())
        self._last_environment_s = perf_counter() - environment_started
        self._last_programmatic_lane_speed_limit_audit = reset.programmatic_lane_speed_limit_audit
        observation_started = perf_counter()
        observation = self._slot.observe()
        self._last_observation_s = perf_counter() - observation_started
        self._last_initial_state = self._slot.vehicle_state
        self._last_traffic_audit = observation.traffic_audit
        return _observation_tensordict(observation.observation)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        trajectory = tensordict["action"].detach().numpy()
        environment_started = perf_counter()
        result = self._slot.step(trajectory)
        self._last_environment_s = perf_counter() - environment_started
        self._last_step = result
        self._last_execution = result.execution
        observation_started = perf_counter()
        observation = self._slot.observe()
        self._last_observation_s = perf_counter() - observation_started
        self._last_traffic_audit = observation.traffic_audit
        return TensorDict(
            {
                **observation.observation,
                "reward": torch.tensor([result.reward], dtype=torch.float32),
                "done": torch.tensor([result.terminated or result.truncated], dtype=torch.bool),
                "terminated": torch.tensor([result.terminated], dtype=torch.bool),
                "truncated": torch.tensor([result.truncated], dtype=torch.bool),
            },
            batch_size=[],
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            if type(seed) is not int or seed < 0:
                raise ValueError("seed must be a non-negative integer")
            self._seed = seed

    @property
    def last_reset(self) -> EnvSlotReset:
        """Return metadata emitted by the latest slot reset."""

        if self._last_reset is None:
            raise RuntimeError("TorchRL MetaDrive environment has not reset")
        return self._last_reset

    @property
    def last_initial_state(self) -> np.ndarray:
        """Return the post-warmup state captured by the latest reset."""

        if self._last_initial_state is None:
            raise RuntimeError("TorchRL MetaDrive environment has not reset")
        return self._last_initial_state

    @property
    def last_step(self) -> EnvSlotStep:
        """Return the domain step result produced by the latest transition."""

        if self._last_step is None:
            raise RuntimeError("TorchRL MetaDrive environment has not stepped")
        return self._last_step

    @property
    def last_execution(self) -> TrajectoryExecutionRecord:
        """Return the immutable execution record produced by the latest step."""

        if self._last_execution is None:
            raise RuntimeError("TorchRL MetaDrive environment has not stepped")
        return self._last_execution

    @property
    def last_warmup_executions(self) -> tuple[TrajectoryExecutionRecord, ...]:
        """Return the warmup records emitted by the latest reset."""

        return self._last_warmup_executions

    @property
    def last_traffic_audit(self) -> object:
        """Return the traffic selection audit captured with the latest observation."""

        return self._last_traffic_audit

    @property
    def last_programmatic_lane_speed_limit_audit(self) -> Mapping[str, object]:
        """Return the lane-speed audit captured by the latest reset."""

        return self._last_programmatic_lane_speed_limit_audit

    @property
    def last_environment_s(self) -> float:
        """Return simulator service time for the latest reset or step."""

        return self._last_environment_s

    @property
    def last_observation_s(self) -> float:
        """Return observation-build time for the latest reset or step."""

        return self._last_observation_s

    def close(self, *, raise_if_closed: bool = True) -> None:
        """Close the wrapped slot when this adapter owns the final lifecycle boundary."""

        del raise_if_closed
        self._slot.close()


def _observation_spec(spec: PlannerObservationSpec) -> Composite:
    return Composite(
        ego_current_state=Unbounded(shape=(10,), dtype=torch.float32, device=_CPU_DEVICE),
        neighbor_agents_past=Unbounded(
            shape=(spec.agent_num, spec.time_len, spec.agent_state_dim),
            dtype=torch.float32,
            device=_CPU_DEVICE,
        ),
        static_objects=Unbounded(
            shape=(spec.static_objects_num, spec.static_objects_state_dim),
            dtype=torch.float32,
            device=_CPU_DEVICE,
        ),
        lanes=Unbounded(
            shape=(spec.lane_num, spec.lane_len, spec.lane_state_dim),
            dtype=torch.float32,
            device=_CPU_DEVICE,
        ),
        lanes_speed_limit=Unbounded(
            shape=(spec.lane_num, 1), dtype=torch.float32, device=_CPU_DEVICE
        ),
        lanes_has_speed_limit=Binary(
            1, shape=(spec.lane_num, 1), dtype=torch.bool, device=_CPU_DEVICE
        ),
        route_lanes=Unbounded(
            shape=(spec.route_num, spec.route_len, spec.route_state_dim),
            dtype=torch.float32,
            device=_CPU_DEVICE,
        ),
        route_lanes_speed_limit=Unbounded(
            shape=(spec.route_num, 1), dtype=torch.float32, device=_CPU_DEVICE
        ),
        route_lanes_has_speed_limit=Binary(
            1, shape=(spec.route_num, 1), dtype=torch.bool, device=_CPU_DEVICE
        ),
        shape=(),
        device=_CPU_DEVICE,
    )


def _observation_tensordict(observation: dict[str, torch.Tensor]) -> TensorDictBase:
    return TensorDict(
        {
            **observation,
            "done": torch.zeros(1, dtype=torch.bool),
            "terminated": torch.zeros(1, dtype=torch.bool),
            "truncated": torch.zeros(1, dtype=torch.bool),
        },
        batch_size=[],
    )
