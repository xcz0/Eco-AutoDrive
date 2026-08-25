from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eco_planner.envs import MetaDriveEnvSlot, PlannerObservationSpec, TrajectoryExecutionRecord
from eco_planner.envs import slot as slot_module
from eco_planner.envs.traffic_state import TrafficFrame


def _spec() -> PlannerObservationSpec:
    return PlannerObservationSpec(21, 11, 32, 10, 5, 20, 12, 70, 20, 12, 25)


def test_planner_observation_spec_copies_only_adapter_dimensions() -> None:
    config = SimpleNamespace(
        time_len=21,
        agent_state_dim=11,
        agent_num=32,
        static_objects_state_dim=10,
        static_objects_num=5,
        lane_len=20,
        lane_state_dim=12,
        lane_num=70,
        route_len=20,
        route_state_dim=12,
        route_num=25,
        state_normalizer=object(),
        observation_normalizer=object(),
    )

    assert PlannerObservationSpec.from_planner_config(config) == _spec()


def test_no_traffic_slot_owns_reset_observe_step_and_map_replacement(monkeypatch) -> None:
    _patch_slot_dependencies(monkeypatch, _FakeNoTrafficAdapter)
    slot = MetaDriveEnvSlot(
        {"map": "S"},
        mode="no_traffic",
        observation_spec=_spec(),
        map_query_radius_m=100.0,
        history_warmup_steps=0,
    )
    first_env = slot.env
    try:
        reset = slot.reset(map_name="S", seed=3)
        observation = slot.observe()
        step = slot.step(_stationary_trajectory())
        slot.reset(map_name="SC", seed=4)

        assert reset.route_length_m == 123.0
        assert reset.warmup_initial_state.shape == (7,)
        assert observation.traffic_audit is None
        assert observation.observation["ego_current_state"].shape == (10,)
        assert step.execution.substep_states.shape == (5, 7)
        assert first_env.closed
        assert slot.env is not first_env
        assert slot.env.config["map"] == "SC"
    finally:
        slot.close()


def test_traffic_slot_yields_complete_stationary_warmup_and_commits_history(
    monkeypatch,
) -> None:
    _patch_slot_dependencies(monkeypatch, _FakeTrafficAdapter)
    with MetaDriveEnvSlot(
        {"map": "S"},
        mode="traffic",
        observation_spec=_spec(),
        map_query_radius_m=100.0,
        history_warmup_steps=20,
    ) as slot:
        slot.reset(map_name="S", seed=3)
        executions = tuple(slot.warmup())
        observation = slot.observe()

    assert len(executions) == 4
    assert sum(execution.substep_states.shape[0] for execution in executions) == 20
    assert observation.traffic_audit is not None
    assert observation.traffic_audit.participant_count_in_radius == 20


def test_traffic_slot_yields_partial_record_before_warmup_failure(monkeypatch) -> None:
    class TerminatingEnv(_FakeEnv):
        def step(self, trajectory: np.ndarray):
            result = super().step(trajectory)
            if self.step_count == 2:
                return result[0], result[1], True, result[3], result[4]
            return result

    _patch_slot_dependencies(monkeypatch, _FakeTrafficAdapter, TerminatingEnv)
    with MetaDriveEnvSlot(
        {"map": "S"},
        mode="traffic",
        observation_spec=_spec(),
        map_query_radius_m=100.0,
        history_warmup_steps=20,
    ) as slot:
        slot.reset(map_name="S", seed=3)
        warmup = slot.warmup()
        first = next(warmup)
        second = next(warmup)
        with pytest.raises(RuntimeError, match="ended before"):
            next(warmup)

    assert first.substep_states.shape == second.substep_states.shape == (5, 7)


def _patch_slot_dependencies(monkeypatch, adapter: type, env: type | None = None) -> None:
    monkeypatch.setattr(slot_module, "TrajectoryMetaDriveEnv", env or _FakeEnv)
    monkeypatch.setattr(slot_module, "MetaDriveObservationAdapter", _FakeTrafficAdapter)
    monkeypatch.setattr(slot_module, "NoTrafficMetaDriveObservationAdapter", adapter)


class _FakeEnv:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = dict(config)
        self.agent = SimpleNamespace(
            position=np.zeros(2), heading_theta=0.0, velocity=np.zeros(2), speed=0.0
        )
        self.route_completion = 0.0
        self.route_length_m = 123.0
        self.programmatic_lane_speed_limit_audit = {"configured": True}
        self.initial_traffic_frame = TrafficFrame(0, (0.0, 0.0), 0.0, 1.0, (), ())
        self.step_count = 0
        self.closed = False

    def reset(self, *, seed: int):
        self.step_count = 0
        return None, {"seed": seed}

    def step(self, trajectory: np.ndarray):
        assert trajectory.shape == (80, 4)
        self.step_count += 1
        execution = _execution(self.step_count)
        return None, 1.0, False, False, {"trajectory_execution": execution}

    def close(self) -> None:
        self.closed = True


class _FakeNoTrafficAdapter:
    def __init__(self, spec: PlannerObservationSpec, radius: float) -> None:
        assert spec == _spec()
        assert radius == 100.0

    def reset(self, env: _FakeEnv) -> None:
        pass

    def build(self, env: _FakeEnv) -> dict[str, torch.Tensor]:
        return _observation()


class _FakeTrafficAdapter(_FakeNoTrafficAdapter):
    def __init__(self, spec: PlannerObservationSpec, radius: float) -> None:
        super().__init__(spec, radius)
        self.frames: list[TrafficFrame] = []

    def reset(self, env: _FakeEnv, initial_frame: TrafficFrame) -> None:
        self.frames = [initial_frame]

    def append_frames(self, frames: tuple[TrafficFrame, ...]) -> None:
        self.frames.extend(frames)

    def build(self, env: _FakeEnv):
        from eco_planner.envs.observation_adapter import TrafficObservationAudit

        return _observation(), TrafficObservationAudit((), len(self.frames) - 1, 0, None)


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


def _execution(step: int) -> TrajectoryExecutionRecord:
    states = np.zeros((5, 7), dtype=np.float64)
    frames = tuple(
        TrafficFrame(index, (0.0, 0.0), 0.0, 1.0, (), ())
        for index in range((step - 1) * 5 + 1, step * 5 + 1)
    )
    return TrajectoryExecutionRecord(
        start_center=np.zeros(2),
        start_heading=0.0,
        world_centers=np.zeros((80, 2)),
        world_headings=np.zeros(80),
        substep_states=states,
        target_centers=states[:, :2],
        target_headings=states[:, 2],
        position_errors_m=np.zeros(5),
        heading_errors_rad=np.zeros(5),
        substep_rewards=np.zeros(5),
        substep_dense_rewards=np.zeros(5),
        substep_energy_ml=np.zeros(5),
        substep_episode_energy_ml=np.zeros(5),
        substep_terminated=np.zeros(5, dtype=np.bool_),
        substep_truncated=np.zeros(5, dtype=np.bool_),
        traffic_frames=frames,
        route_completion=0.0,
        arrive_dest=False,
        out_of_road=False,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        max_step=False,
    )


def _stationary_trajectory() -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory
