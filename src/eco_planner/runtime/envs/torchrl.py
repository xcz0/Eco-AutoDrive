"""Thin TorchRL structural adapter around one MetaDrive environment slot."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs import EnvBase

from eco_planner.contracts import PLANNER_HORIZON
from eco_planner.envs.metadrive import (
    EnvSlotReset,
    EnvSlotStep,
    LocalRouteUnavailableError,
    MetaDriveEnvSlot,
)
from eco_planner.envs.observation import PLANNER_OBSERVATION_FIELDS

_CPU_DEVICE = torch.device("cpu")


class TorchRLMetaDriveEnv(EnvBase):
    """Expose one fixed MetaDrive slot through TorchRL's structural tensor contract."""

    def __init__(
        self,
        slot: MetaDriveEnvSlot,
        *,
        map_name: str,
        seed: int,
    ) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        super().__init__(device=_CPU_DEVICE, batch_size=torch.Size())
        self._slot = slot
        self._map_name = map_name
        self._seed = seed
        self._last_slot_result: EnvSlotReset | EnvSlotStep | None = None
        self.observation_spec = _observation_spec()
        self.action_spec = Unbounded(
            shape=torch.Size((PLANNER_HORIZON, 4)),
            dtype=torch.float32,
            device=_CPU_DEVICE,
        )
        self.reward_spec = Unbounded(
            shape=torch.Size((1,)), dtype=torch.float32, device=_CPU_DEVICE
        )
        self.done_spec = Composite(
            done=Binary(1, shape=torch.Size((1,)), dtype=torch.bool, device=_CPU_DEVICE),
            terminated=Binary(1, shape=torch.Size((1,)), dtype=torch.bool, device=_CPU_DEVICE),
            truncated=Binary(1, shape=torch.Size((1,)), dtype=torch.bool, device=_CPU_DEVICE),
            shape=(),
            device=_CPU_DEVICE,
        )

    def _reset(self, tensordict: TensorDictBase | None, **kwargs: object) -> TensorDictBase:
        del tensordict, kwargs
        try:
            return self._do_reset()
        except LocalRouteUnavailableError:
            self._slot.recreate_environment()
            return self._do_reset()

    def _do_reset(self) -> TensorDictBase:
        reset = self._slot.reset(map_name=self._map_name, seed=self._seed)
        self._last_slot_result = reset
        return TensorDict(
            {
                "observation": reset.state.observation.clone(),
                "done": torch.zeros(1, dtype=torch.bool),
                "terminated": torch.zeros(1, dtype=torch.bool),
                "truncated": torch.zeros(1, dtype=torch.bool),
            },
            batch_size=[],
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get("action")
        if not isinstance(action, torch.Tensor):
            raise TypeError("TorchRL action must be a tensor")
        result = self._slot.step(action.detach().numpy())
        self._last_slot_result = result
        execution = result.execution
        return TensorDict(
            {
                "observation": result.state.observation.clone(),
                # TorchRL requires a reward field structurally. RL replaces this neutral
                # placeholder with the parent-side reward evaluator result.
                "reward": torch.zeros(1, dtype=torch.float32),
                "done": torch.tensor(
                    [execution.terminated or execution.truncated], dtype=torch.bool
                ),
                "terminated": torch.tensor([execution.terminated], dtype=torch.bool),
                "truncated": torch.tensor([execution.truncated], dtype=torch.bool),
            },
            batch_size=[],
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            if type(seed) is not int or seed < 0:
                raise ValueError("seed must be a non-negative integer")
            self._seed = seed

    @property
    def last_slot_result(self) -> EnvSlotReset | EnvSlotStep:
        """Return the latest complete slot operation for the worker side channel."""

        if self._last_slot_result is None:
            raise RuntimeError("MetaDrive slot has not completed an operation")
        return self._last_slot_result

    def close(self, *, raise_if_closed: bool = True) -> None:
        """Close the wrapped slot when this adapter owns the final lifecycle boundary."""

        del raise_if_closed
        self._slot.close()


def _observation_spec() -> Composite:
    fields: dict[str, Any] = {}
    for name, (shape, dtype) in PLANNER_OBSERVATION_FIELDS.items():
        if dtype == np.dtype(np.bool_):
            fields[name] = Binary(1, shape=torch.Size(shape), dtype=torch.bool, device=_CPU_DEVICE)
        else:
            fields[name] = Unbounded(
                shape=torch.Size(shape), dtype=torch.float32, device=_CPU_DEVICE
            )
    return Composite(
        observation=Composite(**fields, shape=(), device=_CPU_DEVICE),
        shape=(),
        device=_CPU_DEVICE,
    )
