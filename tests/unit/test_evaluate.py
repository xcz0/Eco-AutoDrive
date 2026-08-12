from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from eco_planner.envs import TrafficFrame
from eco_planner.evaluation import EpisodeFailure, rendering, run_evaluation, runner
from eco_planner.evaluation.runtime import InferenceRuntimeReport
from eco_planner.evaluation.trace import EpisodeTraceRecorder
from eco_planner.models.checkpoint import CheckpointLoadReport
from eco_planner.models.guidance import NoGuidanceConfig
from eco_planner.models.pretrained import PlannerInferenceResult
from eco_planner.models.sampling_config import SamplerReport


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

    def reset(self, env: _FakeEnv) -> None:
        pass

    def build(self, env: _FakeEnv) -> dict[str, torch.Tensor]:
        return {
            "ego_current_state": torch.zeros((1, 10)),
            "neighbor_agents_past": torch.zeros((1, 32, 21, 11)),
            "static_objects": torch.zeros((1, 5, 10)),
            "lanes": torch.zeros((1, 70, 20, 12)),
            "lanes_speed_limit": torch.full((1, 70, 1), 50.0 / 3.6),
            "lanes_has_speed_limit": torch.ones((1, 70, 1), dtype=torch.bool),
            "route_lanes": torch.zeros((1, 25, 20, 12)),
            "route_lanes_speed_limit": torch.full((1, 25, 1), 50.0 / 3.6),
            "route_lanes_has_speed_limit": torch.ones((1, 25, 1), dtype=torch.bool),
        }


class _FakeRuntime:
    def __init__(self) -> None:
        self.planner_config = SimpleNamespace(predicted_neighbor_num=10, future_len=80)
        self.report = InferenceRuntimeReport(
            requested_accelerator="cpu",
            resolved_accelerator="cpu",
            requested_precision="32-true",
            resolved_precision="32-true",
            device="cpu",
            seed=7,
            world_size=1,
        )
        self.checkpoint_report = CheckpointLoadReport(276, 6_042_628)
        self.sampler_report = SamplerReport(
            name="dpm10",
            num_steps=10,
            timesteps=None,
            initial_noise_scale=0.5,
            ddim_stochasticity=0.0,
            parity_label="official_diffusion_planner_baseline",
        )
        self.guidance_config = NoGuidanceConfig()

    def new_noise_generator(self) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(self.report.seed)

    def infer(
        self,
        observation: dict[str, torch.Tensor],
        generator: torch.Generator,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, PlannerInferenceResult]:
        noise = torch.randn((1, 11, 80, 4), generator=generator)
        prediction = torch.zeros_like(noise)
        prediction[..., 2] = 1.0
        return observation, noise, PlannerInferenceResult(prediction=prediction)


def _config() -> object:
    return OmegaConf.create(
        {
            "evaluation": {
                "mode": "no_traffic",
                "profile": "standard",
                "history_warmup_steps": 0,
                "evaluated_horizon_steps": 10,
                "execution": {
                    "mode": "serial",
                    "launcher": "basic",
                    "worker_count": 1,
                    "torch_threads_per_worker": None,
                    "deterministic": False,
                },
            },
            "env": {"traffic_density": 0.0, "horizon": 10},
            "map_query_radius_m": 100.0,
            "model": {"args_path": "args.json", "checkpoint_path": "model.pth"},
            "runtime": {
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "seed": 7,
            },
            "sampler": {"name": "dpm10"},
            "guidance": {"name": "none"},
            "scenarios": [{"name": "fake", "map": "S", "seed": 3}],
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


def test_evaluation_package_preserves_public_runner() -> None:
    assert run_evaluation is runner.run_evaluation


def test_run_scenario_replans_and_writes_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "TrajectoryMetaDriveEnv", _FakeEnv)
    monkeypatch.setattr(runner, "NoTrafficMetaDriveObservationAdapter", _FakeAdapter)
    monkeypatch.setattr(runner, "_route_length_m", lambda env: 100.0)
    summary = runner._run_scenario(
        runner.ScenarioSpec("fake", "S", 3),
        _FakeRuntime(),  # type: ignore[arg-type]
        _config(),
        tmp_path,
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
        assert trace["warmup_states"].shape == (0, 7)
        assert trace["traffic_selected_ids"].shape == (2, 32)
    payload = json.loads((tmp_path / "fake" / "summary.json").read_text())
    assert payload["noise_seed"] == 7
    assert payload["map_input_audit"]["speed_limit_mps_min"] == pytest.approx(50.0 / 3.6)


def test_run_evaluation_writes_clean_job_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "TrajectoryMetaDriveEnv", _FakeEnv)
    monkeypatch.setattr(runner, "NoTrafficMetaDriveObservationAdapter", _FakeAdapter)
    monkeypatch.setattr(runner, "_route_length_m", lambda env: 100.0)
    monkeypatch.setattr(
        runner,
        "create_fabric_inference_runtime",
        lambda runtime_config, sampler_config, guidance_config, args_path, checkpoint_path: (
            _FakeRuntime()
        ),
    )
    monkeypatch.setattr(
        runner,
        "write_runtime_metadata",
        lambda output, report, sampler_report, guidance_config, execution, elapsed: None,
    )

    summary = runner.run_evaluation(_config(), tmp_path)

    assert set(summary) == {
        "schema_version",
        "status",
        "runtime",
        "checkpoint",
        "sampler",
        "guidance",
        "episodes",
    }
    assert summary["schema_version"] == 2
    assert summary["status"] == "completed"
    assert summary["runtime"]["seed"] == 7
    assert summary["checkpoint"] == {
        "ema_tensor_count": 276,
        "parameter_count": 6_042_628,
    }
    assert summary["sampler"]["name"] == "dpm10"
    assert summary["guidance"] == {"name": "none"}
    assert summary["episodes"][0]["sampler"] == summary["sampler"]
    assert summary["episodes"][0]["guidance"] == summary["guidance"]
    assert "checkpoint" not in summary["episodes"][0]
    persisted = json.loads((tmp_path / "summary.json").read_text())
    assert persisted == summary
    assert (tmp_path / "resolved_config.yaml").is_file()


def test_run_evaluation_persists_explicit_failure_and_continues(tmp_path, monkeypatch) -> None:
    class FailureEnv(_FakeEnv):
        def reset(self, seed: int) -> tuple[None, dict[str, object]]:
            raise EpisodeFailure("reset", RuntimeError("injected episode failure"))

    def environment(config: dict[str, object]) -> _FakeEnv:
        return FailureEnv(config) if config["map"] == "FAIL" else _FakeEnv(config)

    config = _config()
    config.scenarios = [
        {"name": "failed", "map": "FAIL", "seed": 3},
        {"name": "completed", "map": "S", "seed": 3},
    ]
    monkeypatch.setattr(runner, "TrajectoryMetaDriveEnv", environment)
    monkeypatch.setattr(runner, "NoTrafficMetaDriveObservationAdapter", _FakeAdapter)
    monkeypatch.setattr(runner, "_route_length_m", lambda env: 100.0)
    monkeypatch.setattr(
        runner,
        "create_fabric_inference_runtime",
        lambda runtime_config, sampler_config, guidance_config, args_path, checkpoint_path: (
            _FakeRuntime()
        ),
    )
    monkeypatch.setattr(
        runner,
        "write_runtime_metadata",
        lambda output, report, sampler_report, guidance_config, execution, elapsed: None,
    )

    summary = run_evaluation(config, tmp_path)

    assert summary["status"] == "failed"
    assert [episode["status"] for episode in summary["episodes"]] == [
        "failed",
        "completed",
    ]
    failure = summary["episodes"][0]
    assert failure["termination"] == {"type": "runtime_error", "detail": "reset"}
    assert failure["failure"]["exception_type"] == "RuntimeError"
    assert "injected episode failure" in failure["failure"]["traceback"]
    with np.load(tmp_path / "failed" / "trace.npz") as trace:
        assert trace["trace_status"].item() == "empty"
    assert (tmp_path / "completed" / "trace.npz").is_file()


def test_run_evaluation_does_not_catch_unclassified_errors(tmp_path, monkeypatch) -> None:
    class BrokenEnv(_FakeEnv):
        def reset(self, seed: int) -> tuple[None, dict[str, object]]:
            raise RuntimeError("unclassified")

    monkeypatch.setattr(runner, "TrajectoryMetaDriveEnv", BrokenEnv)
    monkeypatch.setattr(runner, "NoTrafficMetaDriveObservationAdapter", _FakeAdapter)
    monkeypatch.setattr(
        runner,
        "create_fabric_inference_runtime",
        lambda runtime_config, sampler_config, guidance_config, args_path, checkpoint_path: (
            _FakeRuntime()
        ),
    )
    monkeypatch.setattr(
        runner,
        "write_runtime_metadata",
        lambda output, report, sampler_report, guidance_config, execution, elapsed: None,
    )

    with pytest.raises(RuntimeError, match="unclassified"):
        run_evaluation(_config(), tmp_path)

    assert not (tmp_path / "summary.json").exists()


def test_world_polyline_draws_on_frame() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    rendering.draw_world_polyline(
        frame,
        np.array([[0.0, 0.0], [5.0, 0.0]]),
        np.array([0.0, 0.0]),
        1.0,
        (1, 2, 3),
        0,
    )
    assert np.count_nonzero(frame) > 0


def test_route_length_accepts_finite_numpy_lane_scalars() -> None:
    lane = SimpleNamespace(length=np.float32(123.5))
    navigation = SimpleNamespace(checkpoints=["start", "end"])
    agent = SimpleNamespace(navigation=navigation)
    road_network = SimpleNamespace(graph={"start": {"end": [lane]}})
    current_map = SimpleNamespace(road_network=road_network)
    env = SimpleNamespace(agent=agent, current_map=current_map)

    assert runner._route_length_m(env) == pytest.approx(123.5)


def test_evaluation_config_rejects_horizon_mismatch() -> None:
    config = _config()
    config.env.horizon = 9

    with pytest.raises(ValueError, match="env.horizon"):
        runner._validate_evaluation_config(config)


def test_evaluation_config_requires_explicit_sampler() -> None:
    config = _config()
    del config.sampler

    with pytest.raises(ValueError, match="select a sampler"):
        runner._validate_evaluation_config(config)


def test_traffic_warmup_records_exact_stationary_history() -> None:
    class WarmupEnv:
        def __init__(self) -> None:
            self.agent = _FakeAgent()
            self.simulator_step = 0

        def step(self, trajectory: np.ndarray) -> tuple[None, float, bool, bool, dict]:
            frames = []
            for _ in range(5):
                self.simulator_step += 1
                frames.append(
                    TrafficFrame(
                        simulator_step=self.simulator_step,
                        ego_center_xy_m=(0.0, 0.0),
                        ego_heading_rad=0.0,
                        ego_rear_wheelbase_m=1.0,
                        participants=(),
                        static_objects=(),
                    )
                )
            info = {
                "trajectory_substep_states": np.zeros((5, 7)),
                "trajectory_substep_rewards": np.zeros(5),
                "trajectory_substep_terminated": np.zeros(5, dtype=np.bool_),
                "trajectory_substep_truncated": np.zeros(5, dtype=np.bool_),
                "traffic_substep_frames": tuple(frames),
            }
            return None, 0.0, False, False, info

    class WarmupAdapter:
        def __init__(self) -> None:
            self.frames: list[TrafficFrame] = []

        def append_frames(self, frames: tuple[TrafficFrame, ...]) -> None:
            self.frames.extend(frames)

    env = WarmupEnv()
    adapter = WarmupAdapter()
    trace = EpisodeTraceRecorder.from_initial_state(np.zeros(7))

    runner._run_traffic_warmup(env, adapter, trace, 20)  # type: ignore[arg-type]

    assert len(adapter.frames) == 20
    assert np.concatenate(trace.warmup_states).shape == (20, 7)
    np.testing.assert_array_equal(trace.initial_state, np.zeros(7))
