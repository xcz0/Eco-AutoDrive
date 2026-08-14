from __future__ import annotations

import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from eco_planner.envs import TrafficFrame
from eco_planner.evaluation import episode
from eco_planner.evaluation.artifacts.trace_recorder import EpisodeTraceRecorder
from eco_planner.evaluation.config import ScenarioConfig, parse_evaluation_config


def test_run_scenario_replans_and_persists_trace(
    tmp_path, fake_runtime: object, evaluation_config: object, patch_episode_dependencies
) -> None:
    patch_episode_dependencies()

    summary = episode.run_scenario(
        ScenarioConfig(name="fake", map="S", seed=3),
        fake_runtime,
        parse_evaluation_config(evaluation_config),
        tmp_path,
    )

    assert (summary.plan_cycles, summary.simulator_steps, summary.terminal_reason) == (
        2,
        10,
        "arrive_dest",
    )
    with np.load(tmp_path / "fake" / "trace.npz") as trace:
        assert trace["initial_noise"].shape == (2, 11, 80, 4)
        assert trace["initial_noise"].dtype == np.float32
        assert trace["predictions_local"].dtype == np.float32
        assert trace["observation_ego_current_state"].dtype == np.float32
        assert trace["executed_states"].shape == (10, 7)
        assert trace["traffic_selected_ids"].shape == (2, 32)
    with zipfile.ZipFile(tmp_path / "fake" / "trace.npz") as archive:
        assert {entry.compress_type for entry in archive.infolist()} == {zipfile.ZIP_STORED}
    assert summary.map_input_audit.speed_limit_mps_min == pytest.approx(50.0 / 3.6)


def test_route_length_accepts_finite_numpy_lane_scalar() -> None:
    lane = SimpleNamespace(length=np.float32(123.5))
    env = SimpleNamespace(
        agent=SimpleNamespace(navigation=SimpleNamespace(checkpoints=["start", "end"])),
        current_map=SimpleNamespace(road_network=SimpleNamespace(graph={"start": {"end": [lane]}})),
    )
    assert episode.route_length_m(env) == pytest.approx(123.5)


def test_traffic_warmup_records_stationary_history() -> None:
    class WarmupEnv:
        def __init__(self) -> None:
            self.agent = SimpleNamespace(
                position=np.zeros(2), heading_theta=0.0, velocity=np.zeros(2), speed=0.0
            )
            self.simulator_step = 0

        def step(self, trajectory: np.ndarray) -> tuple[None, float, bool, bool, dict[str, object]]:
            frames = tuple(
                TrafficFrame(index, (0.0, 0.0), 0.0, 1.0, (), ())
                for index in range(self.simulator_step + 1, self.simulator_step + 6)
            )
            self.simulator_step += 5
            return (
                None,
                0.0,
                False,
                False,
                {
                    "trajectory_start_center": np.zeros(2),
                    "trajectory_start_heading": 0.0,
                    "trajectory_world_centers": np.zeros((80, 2)),
                    "trajectory_world_headings": np.zeros(80),
                    "trajectory_substep_states": np.zeros((5, 7)),
                    "trajectory_substep_rewards": np.zeros(5),
                    "trajectory_substep_dense_rewards": np.zeros(5),
                    "trajectory_substep_terminated": np.zeros(5, dtype=np.bool_),
                    "trajectory_substep_truncated": np.zeros(5, dtype=np.bool_),
                    "traffic_substep_frames": frames,
                    "trajectory_target_centers": np.zeros((5, 2)),
                    "trajectory_target_headings": np.zeros(5),
                    "trajectory_position_errors_m": np.zeros(5),
                    "trajectory_heading_errors_rad": np.zeros(5),
                    "route_completion": 0.0,
                    "arrive_dest": False,
                    "out_of_road": False,
                    "crash_vehicle": False,
                    "crash_object": False,
                    "crash_building": False,
                    "crash_human": False,
                    "max_step": False,
                },
            )

    class WarmupAdapter:
        def __init__(self) -> None:
            self.frames: list[TrafficFrame] = []

        def append_frames(self, frames: tuple[TrafficFrame, ...]) -> None:
            self.frames.extend(frames)

    adapter = WarmupAdapter()
    trace = EpisodeTraceRecorder.from_initial_state(
        np.zeros(7), max_plan_cycles=0, max_warmup_steps=20, guided=False
    )
    episode.run_traffic_warmup(WarmupEnv(), adapter, trace, 20)  # type: ignore[arg-type]
    assert len(adapter.frames) == 20
    assert np.concatenate(trace.warmup_state_arrays).shape == (20, 7)
