from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import check_env_specs

from eco_planner.envs import PlannerObservationSpec, TorchRLMetaDriveEnv


def _spec() -> PlannerObservationSpec:
    return PlannerObservationSpec(21, 11, 32, 10, 5, 20, 12, 70, 20, 12, 25)


def test_torchrl_adapter_matches_observation_action_reward_and_done_specs() -> None:
    slot = _FakeSlot()
    env = TorchRLMetaDriveEnv(slot, map_name="S", seed=3, observation_spec=_spec())

    check_env_specs(env)
    transition = env.step(TensorDict({"action": _trajectory()}, batch_size=[]))

    assert slot.resets
    assert slot.trajectories[-1].shape == (80, 4)
    assert transition["next", "reward"].shape == (1,)
    assert not transition["next", "done"].item()
    assert not transition["next", "terminated"].item()
    assert not transition["next", "truncated"].item()


class _FakeSlot:
    def __init__(self) -> None:
        self.resets: list[tuple[str, int]] = []
        self.trajectories: list[np.ndarray] = []
        self.closed = False

    def reset(self, *, map_name: str, seed: int) -> SimpleNamespace:
        self.resets.append((map_name, seed))
        return SimpleNamespace(
            route_completion=0.0,
            route_length_m=100.0,
            warmup_initial_state=np.zeros(7),
            programmatic_lane_speed_limit_audit={},
        )

    @property
    def vehicle_state(self) -> np.ndarray:
        return np.zeros(7)

    @staticmethod
    def warmup():
        return iter(())

    @staticmethod
    def observe() -> SimpleNamespace:
        return SimpleNamespace(observation=_observation(), traffic_audit=None)

    def step(self, trajectory: np.ndarray) -> SimpleNamespace:
        self.trajectories.append(trajectory)
        return SimpleNamespace(
            reward=0.25,
            terminated=False,
            truncated=False,
            execution=object(),
        )

    def close(self) -> None:
        self.closed = True


def _observation() -> dict[str, torch.Tensor]:
    return {
        "ego_current_state": torch.zeros(10),
        "neighbor_agents_past": torch.zeros((32, 21, 11)),
        "static_objects": torch.zeros((5, 10)),
        "lanes": torch.zeros((70, 20, 12)),
        "lanes_speed_limit": torch.zeros((70, 1)),
        "lanes_has_speed_limit": torch.zeros((70, 1), dtype=torch.bool),
        "route_lanes": torch.zeros((25, 20, 12)),
        "route_lanes_speed_limit": torch.zeros((25, 1)),
        "route_lanes_has_speed_limit": torch.zeros((25, 1), dtype=torch.bool),
    }


def _trajectory() -> torch.Tensor:
    trajectory = torch.zeros((80, 4), dtype=torch.float32)
    trajectory[:, 2] = 1.0
    return trajectory
