"""Unit contracts for atomic environment lifecycle edge cases."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from eco_planner.envs.domain import TrafficFrame, TrafficParticipantState
from eco_planner.envs.metadrive import (
    EnvSlotReset,
    EnvSlotState,
    EnvSlotTiming,
    LocalRouteUnavailableError,
)
from eco_planner.envs.metadrive.observation import NoTrafficMetaDriveObservationPipeline
from eco_planner.envs.observation import PLANNER_OBSERVATION_FIELDS
from eco_planner.runtime.envs.torchrl import TorchRLMetaDriveEnv


def _observation() -> TensorDict:
    fields = {
        name: torch.zeros(shape, dtype=torch.bool if dtype == np.dtype(np.bool_) else torch.float32)
        for name, (shape, dtype) in PLANNER_OBSERVATION_FIELDS.items()
    }
    return TensorDict(fields, batch_size=[])


def _frame(
    simulator_step: int,
    *,
    participants: tuple[TrafficParticipantState, ...] = (),
) -> TrafficFrame:
    return TrafficFrame(
        simulator_step=simulator_step,
        ego_center_xy_m=(0.0, 0.0),
        ego_heading_rad=0.0,
        ego_rear_wheelbase_m=1.0,
        participants=participants,
        static_objects=(),
    )


class _RouteRetrySlot:
    def __init__(self, failure: Exception) -> None:
        self._failure = failure
        self.reset_calls = 0
        self.recreate_calls = 0

    def reset(self, *, map_name: str, seed: int) -> EnvSlotReset:
        assert (map_name, seed) == ("S", 7)
        self.reset_calls += 1
        if self.reset_calls == 1:
            raise self._failure
        return EnvSlotReset(
            state=EnvSlotState(_observation(), None, np.zeros(7), 0.0),
            route_length_m=100.0,
            warmup_initial_state=np.zeros(7),
            warmup_steps=(),
            programmatic_lane_speed_limit_audit={},
            timing=EnvSlotTiming(0.0, 0.0),
        )

    def recreate_environment(self) -> None:
        self.recreate_calls += 1

    def close(self) -> None:
        pass


def test_torchrl_reset_recovers_once_from_typed_local_route_failure() -> None:
    slot = _RouteRetrySlot(LocalRouteUnavailableError("typed route failure"))
    env = TorchRLMetaDriveEnv(slot, map_name="S", seed=7)  # type: ignore[arg-type]

    output = env._reset(None)

    assert output["observation"].batch_size == torch.Size([])
    assert slot.reset_calls == 2
    assert slot.recreate_calls == 1


def test_torchrl_reset_does_not_match_runtime_error_text() -> None:
    slot = _RouteRetrySlot(RuntimeError("no connected navigation route lanes exist"))
    env = TorchRLMetaDriveEnv(slot, map_name="S", seed=7)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="no connected navigation route lanes exist"):
        env._reset(None)

    assert slot.reset_calls == 1
    assert slot.recreate_calls == 0


def test_no_traffic_pipeline_rejects_participants_in_later_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = NoTrafficMetaDriveObservationPipeline(100.0)
    monkeypatch.setattr(pipeline._map_adapter, "reset", lambda env: None)
    env = SimpleNamespace(
        config={"traffic_density": 0.0, "random_traffic": False, "accident_prob": 0.0}
    )
    pipeline.reset(env, _frame(0))
    participant = TrafficParticipantState(
        object_id="late-vehicle",
        kind="vehicle",
        position_xy_m=(1.0, 0.0),
        heading_rad=0.0,
        velocity_xy_mps=(0.0, 0.0),
        width_m=2.0,
        length_m=4.0,
    )

    with pytest.raises(RuntimeError, match="late-vehicle"):
        pipeline.append_frames((_frame(1, participants=(participant,)),))
