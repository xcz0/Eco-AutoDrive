"""Thin TorchRL structural adapter around one MetaDrive environment slot."""

from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs import EnvBase

from eco_planner.envs.observation import PlannerObservationSpec
from eco_planner.envs.slot import MetaDriveEnvSlot
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
        self._slot.reset(map_name=self._map_name, seed=self._seed)
        tuple(self._slot.warmup())
        return _observation_tensordict(self._slot.observe().observation)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        trajectory = tensordict["action"].detach().numpy()
        result = self._slot.step(trajectory)
        return TensorDict(
            {
                **self._slot.observe().observation,
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

    def close(self) -> None:
        """Close the wrapped slot when this adapter owns the final lifecycle boundary."""

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
