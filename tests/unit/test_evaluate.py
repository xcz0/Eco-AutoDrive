from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from eco_planner import evaluate
from eco_planner.models.checkpoint import CheckpointLoadReport


class _FakeAgent:
    def __init__(self) -> None:
        self.position = np.array([0.0, 0.0])
        self.heading_theta = 0.0
        self.velocity = np.array([0.0, 0.0])
        self.speed = 0.0


class _FakeEnv:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.agent = _FakeAgent()
        self._step = 0

    @property
    def programmatic_lane_speed_limit_audit(self) -> dict[str, object]:
        return {
            "speed_limit_sentinel_replaced_count": 18,
            "speed_limit_existing_preserved_count": 0,
            "configured_programmatic_lane_speed_limit_kmh": 50.0,
            "lane_speed_limit_kmh_counts": {"50": 18},
        }

    def reset(self, seed: int) -> tuple[None, dict[str, object]]:
        assert seed == 3
        return None, {}

    def step(self, trajectory: np.ndarray) -> tuple[None, float, bool, bool, dict]:
        assert trajectory.shape == (80, 4)
        self._step += 1
        start = float((self._step - 1) * 5)
        positions = np.column_stack((np.arange(1, 6) + start, np.zeros(5)))
        self.agent.position = positions[-1]
        self.agent.velocity = np.array([10.0, 0.0])
        self.agent.speed = 10.0
        terminated = self._step == 2
        states = np.column_stack(
            (
                positions,
                np.zeros(5),
                np.full(5, 10.0),
                np.zeros(5),
                np.full(5, 10.0),
                np.zeros(5),
            )
        )
        info = {
            "trajectory_world_centers": np.column_stack((np.arange(1, 81) + start, np.zeros(80))),
            "trajectory_world_headings": np.zeros(80),
            "trajectory_substep_states": states,
            "trajectory_substep_rewards": np.ones(5),
            "trajectory_substep_terminated": np.array([False, False, False, False, terminated]),
            "trajectory_substep_truncated": np.zeros(5, dtype=np.bool_),
            "trajectory_target_centers": positions,
            "trajectory_target_headings": np.zeros(5),
            "trajectory_position_errors_m": np.zeros(5),
            "trajectory_heading_errors_rad": np.zeros(5),
            "route_completion": self._step / 2,
            "arrive_dest": terminated,
            "out_of_road": False,
            "crash_vehicle": False,
            "crash_object": False,
            "crash_building": False,
            "crash_human": False,
            "max_step": False,
        }
        return None, 5.0, terminated, False, info

    def close(self) -> None:
        pass


class _FakeAdapter:
    def __init__(self, config: object, radius: float) -> None:
        assert radius == 100.0

    def build(self, env: _FakeEnv, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "ego_current_state": torch.zeros((1, 10)),
            "neighbor_agents_past": torch.zeros((1, 20, 21, 11)),
            "static_objects": torch.zeros((1, 20, 10)),
            "lanes": torch.zeros((1, 70, 20, 12)),
            "lanes_speed_limit": torch.full((1, 70, 1), 50.0 / 3.6),
            "lanes_has_speed_limit": torch.ones((1, 70, 1), dtype=torch.bool),
            "route_lanes": torch.zeros((1, 25, 20, 12)),
        }


class _FakePlanner:
    def __init__(self) -> None:
        self.config = SimpleNamespace(predicted_neighbor_num=10, future_len=80)

    def predict(self, observation: dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
        prediction = torch.zeros_like(noise)
        prediction[..., 2] = 1.0
        return prediction


def _config() -> object:
    return OmegaConf.create(
        {
            "env": {"traffic_density": 0.0},
            "map_query_radius_m": 100.0,
            "model": {"seed": 7},
            "video": {
                "enabled": False,
                "fps": 2,
                "screen_width": 32,
                "screen_height": 32,
                "film_width": 32,
                "film_height": 32,
                "scaling": 1.0,
            },
        }
    )


def test_run_scenario_replans_and_writes_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluate, "TrajectoryMetaDriveEnv", _FakeEnv)
    monkeypatch.setattr(evaluate, "NoTrafficMetaDriveObservationAdapter", _FakeAdapter)
    report = CheckpointLoadReport("a" * 64, "b" * 64, 276, 6_042_628, "cpu")

    summary = evaluate._run_scenario(
        evaluate.ScenarioSpec("fake", "S", 3),
        _FakePlanner(),
        report,
        _config(),
        tmp_path,
        torch.device("cpu"),
    )

    assert summary["plan_cycles"] == 2
    assert summary["simulator_steps"] == 10
    assert summary["terminal_reason"] == "arrive_dest"
    assert (tmp_path / "fake" / "summary.json").is_file()
    with np.load(tmp_path / "fake" / "trace.npz") as trace:
        assert trace["initial_noise"].shape == (2, 11, 80, 4)
        assert trace["predictions_local"].shape == (2, 11, 80, 4)
        assert trace["observation_ego_current_state"].shape == (2, 10)
        assert trace["observation_lanes"].shape == (2, 70, 20, 12)
        assert trace["observation_lanes_speed_limit"].shape == (2, 70, 1)
        assert trace["observation_lanes_has_speed_limit"].dtype == np.bool_
        assert trace["executed_states"].shape == (10, 7)
        assert trace["trajectory_target_centers"].shape == (10, 2)
    payload = json.loads((tmp_path / "fake" / "summary.json").read_text())
    assert payload["noise_seed"] == 7
    assert payload["map_input_audit"]["speed_limit_mps_min"] == pytest.approx(50.0 / 3.6)


def test_world_polyline_draws_on_frame() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    evaluate._draw_world_polyline(
        frame,
        np.array([[0.0, 0.0], [5.0, 0.0]]),
        np.array([0.0, 0.0]),
        1.0,
        (1, 2, 3),
        0,
    )
    assert np.count_nonzero(frame) > 0
