"""Parent-side bridge for TorchRL's synchronous collection lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs import EnvBase

from eco_planner.contracts import PLANNER_HORIZON
from eco_planner.envs.parallel import VectorEnvScenario, VectorMetaDriveEnv
from eco_planner.envs.parallel.torchrl import planner_observation_spec

from .contracts import TRAINING_KEYS


class PlannerCollectionPolicy:
    """Declare outputs without a discovery forward that would consume planner RNG."""

    out_keys = ("action", *TRAINING_KEYS[:-1])

    def __init__(self, plan: Callable[[TensorDictBase], TensorDictBase]) -> None:
        self._plan = plan

    def __call__(self, carrier: TensorDictBase) -> TensorDictBase:
        return self._plan(carrier)


class TorchRLCollectionEnv(EnvBase):
    """Lend a logical wave of an existing worker pool to a synchronous Collector.

    Observations and execution actions stay on CPU; policy outputs and reward may
    remain on the policy device. The bridge owns neither workers nor random streams.
    Domain callbacks run before TorchRL processes the step's reset mask, so final
    observations can be audited and explicitly bootstrapped before any reset.
    """

    def __init__(
        self,
        envs: VectorMetaDriveEnv,
        scenarios: tuple[VectorEnvScenario, ...],
        *,
        transitions_per_slot: int,
        reward_device: torch.device,
        on_step: Callable[[TensorDictBase], torch.Tensor],
        on_reset: Callable[[tuple[int, ...], TensorDictBase], None],
    ) -> None:
        count = len(scenarios)
        super().__init__(batch_size=torch.Size([count]), device=None)
        self._envs = envs
        self._scenarios = scenarios
        self._observation: TensorDictBase | None = None
        self._limit = transitions_per_slot
        self._round = 0
        self._on_step = on_step
        self._on_reset = on_reset
        self.observation_spec = planner_observation_spec().expand(self.batch_size)
        self.action_spec = Unbounded(
            shape=torch.Size([count, PLANNER_HORIZON, 4]), dtype=torch.float32
        )
        self.reward_spec = Unbounded(
            shape=torch.Size([count, 1]), dtype=torch.float32, device=reward_device
        )
        self.done_spec = Composite(
            {
                key: Binary(1, shape=torch.Size([count, 1]), dtype=torch.bool)
                for key in ("done", "terminated", "truncated")
            },
            shape=self.batch_size,
        )
        self._post_step_mdp_hooks = _collection_carriers

    def _reset(self, tensordict: TensorDictBase | None, **kwargs: object) -> TensorDictBase:
        del kwargs
        if self._round < self._limit:
            mask = (
                torch.ones(self.batch_size, dtype=torch.bool)
                if tensordict is None
                else tensordict.get(
                    "_reset", torch.ones((*self.batch_size, 1), dtype=torch.bool)
                ).reshape(self.batch_size)
            )
            slots = tuple(mask.nonzero(as_tuple=True)[0].tolist())
            resets = self._envs.reset(tuple(self._scenarios[slot] for slot in slots), slots=slots)
            self._on_reset(slots, resets)
            if self._observation is None:
                self._observation = cast(TensorDictBase, resets["observation"])
            else:
                observation = self._observation.clone()
                observation[list(slots)] = cast(TensorDictBase, resets["observation"])
                self._observation = observation
        # Collector may request a reset after the last real terminal. At the quota
        # boundary there is no next episode: do not reset or advance any worker.
        return TensorDict(
            {
                "observation": self._observation,
                **{
                    key: torch.zeros((*self.batch_size, 1), dtype=torch.bool)
                    for key in self.done_keys
                },
            },
            batch_size=self.batch_size,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        steps = self._envs.step(
            tensordict["action"].numpy(),
            slots=None
            if len(self._scenarios) == self._envs.num_envs
            else tuple(range(len(self._scenarios))),
        )
        self._round += 1
        reward = self._on_step(steps)
        self._observation = cast(TensorDictBase, steps["observation"])
        return steps.select("observation", "done", "terminated", "truncated").set("reward", reward)

    def _set_seed(self, seed: int | None) -> None:
        raise RuntimeError("collection seeds belong to logical scenarios and explicit generators")


def _collection_carriers(
    transition: TensorDictBase, next_state: TensorDictBase
) -> tuple[TensorDictBase, TensorDictBase]:
    # EnvBase's post-step hook runs before maybe_reset. Policy fields belong to
    # the stored transition, not the next environment state: CPU reset masks must
    # never be applied to GPU context/action tensors. Observations are already
    # owned by next_state and the domain audit, so do not stack a second copy.
    next_state = next_state.select("observation", "done", "terminated", "truncated")
    transition.exclude("observation", ("next", "observation"), inplace=True)
    return transition, next_state
