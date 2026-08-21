from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from eco_planner.envs import TrafficFrame, TrajectoryExecutionRecord
from eco_planner.envs.observation_adapter import TrafficObservationAudit
from eco_planner.evaluation import episode
from eco_planner.evaluation.artifacts.trace_recorder import EpisodeTraceRecorder
from eco_planner.evaluation.runtime.contracts import HostExecutionResult, HostInferenceResult
from eco_planner.evaluation.runtime.engine import InferenceDecision, InferenceRuntimeReport
from eco_planner.models import CheckpointLoadReport, NoGuidanceConfig, SamplerReport


class FakeAgent:
    def __init__(self) -> None:
        self.position = np.array([0.0, 0.0])
        self.heading_theta = 0.0
        self.velocity = np.array([0.0, 0.0])
        self.speed = 0.0


class FakeEnv:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.agent = FakeAgent()
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

    def step(self, trajectory: np.ndarray) -> tuple[None, float, bool, bool, dict[str, object]]:
        assert trajectory.shape == (80, 4)
        self._step += 1
        start = float((self._step - 1) * 5)
        positions = np.column_stack((np.arange(1, 6) + start, np.zeros(5)))
        self.agent.position = positions[-1]
        self.agent.velocity = np.array([10.0, 0.0])
        self.agent.speed = 10.0
        terminated = self._step == 2
        states = np.column_stack(
            (positions, np.zeros(5), np.full(5, 10.0), np.zeros(5), np.full(5, 10.0), np.zeros(5))
        )
        execution = TrajectoryExecutionRecord(
            start_center=np.array([start, 0.0]),
            start_heading=0.0,
            world_centers=np.column_stack((np.arange(1, 81) + start, np.zeros(80))),
            world_headings=np.zeros(80),
            substep_states=states,
            target_centers=positions,
            target_headings=np.zeros(5),
            position_errors_m=np.zeros(5),
            heading_errors_rad=np.zeros(5),
            substep_rewards=np.ones(5),
            substep_dense_rewards=np.ones(5),
            substep_terminated=np.array([False, False, False, False, terminated]),
            substep_truncated=np.zeros(5, dtype=np.bool_),
            traffic_frames=tuple(
                TrafficFrame(index, (0.0, 0.0), 0.0, 1.0, (), ())
                for index in range((self._step - 1) * 5 + 1, self._step * 5 + 1)
            ),
            route_completion=self._step / 2,
            arrive_dest=terminated,
            out_of_road=False,
            crash_vehicle=False,
            crash_object=False,
            crash_building=False,
            crash_human=False,
            max_step=False,
        )
        info: dict[str, object] = {"trajectory_execution": execution}
        return None, 5.0, terminated, False, info

    def close(self) -> None:
        pass


class FakeAdapter:
    def __init__(self, config: object, radius: float) -> None:
        assert radius == 100.0

    def reset(self, env: FakeEnv) -> None:
        pass

    def build(self, env: FakeEnv) -> dict[str, torch.Tensor]:
        return {
            "ego_current_state": torch.zeros(10),
            "neighbor_agents_past": torch.zeros((32, 21, 11)),
            "static_objects": torch.zeros((5, 10)),
            "lanes": torch.zeros((70, 20, 12)),
            "lanes_speed_limit": torch.full((70, 1), 50.0 / 3.6),
            "lanes_has_speed_limit": torch.ones((70, 1), dtype=torch.bool),
            "route_lanes": torch.zeros((25, 20, 12)),
            "route_lanes_speed_limit": torch.full((25, 1), 50.0 / 3.6),
            "route_lanes_has_speed_limit": torch.ones((25, 1), dtype=torch.bool),
        }


class FakeRuntime:
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
            implementation="diffusers",
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
        self, observation: dict[str, torch.Tensor], generator: torch.Generator
    ) -> InferenceDecision:
        noise = torch.randn((1, 11, 80, 4), generator=generator)
        prediction = torch.zeros_like(noise)
        prediction[..., 2] = 1.0
        audit = HostInferenceResult(
            initial_noise=noise.numpy(),
            prediction=prediction.numpy(),
        )
        return InferenceDecision(HostExecutionResult(audit.ego_trajectory[None]), lambda: audit)


@pytest.fixture
def evaluation_config() -> object:
    return OmegaConf.create(
        {
            "name": "test-evaluation",
            "evaluation": {
                "mode": "no_traffic",
                "profile": "standard",
                "history_warmup_steps": 0,
                "evaluated_horizon_steps": 10,
                "execution": {
                    "mode": "serial",
                    "torch_threads_per_worker": None,
                    "deterministic": False,
                },
            },
            "env": {"traffic_density": 0.0, "trajectory_execution_steps": 5, "horizon": 10},
            "map_query_radius_m": 100.0,
            "model": {"args_path": "args.json", "checkpoint_path": "model.pth"},
            "runtime": {
                "accelerator": "cpu",
                "precision": "32-true",
                "seed": 7,
            },
            "sampler": {"name": "dpm10", "implementation": "diffusers"},
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


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture
def fake_env_class() -> type[FakeEnv]:
    return FakeEnv


@pytest.fixture
def patch_episode_dependencies(monkeypatch: pytest.MonkeyPatch) -> Callable[[object], None]:
    def patch(environment: object = FakeEnv) -> None:
        monkeypatch.setattr(episode, "TrajectoryMetaDriveEnv", environment)
        monkeypatch.setattr(episode, "NoTrafficMetaDriveObservationAdapter", FakeAdapter)
        monkeypatch.setattr(episode, "route_length_m", lambda env: 100.0)

    return patch


@pytest.fixture
def matrix_trace_arrays() -> dict[str, np.ndarray]:
    environment = FakeEnv({})
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    _, _, _, _, info = environment.step(trajectory)
    execution = info["trajectory_execution"]
    assert isinstance(execution, TrajectoryExecutionRecord)
    recorder = EpisodeTraceRecorder.from_initial_state(
        np.zeros(7), max_plan_cycles=1, max_warmup_steps=20, guided=False
    )
    for _ in range(4):
        recorder.append_warmup(
            execution,
            np.ones(execution.substep_states.shape[0], dtype=np.int64),
            np.zeros(execution.substep_states.shape[0], dtype=np.int64),
        )
    observation = FakeAdapter(None, 100.0).build(environment)
    prediction = np.zeros((1, 11, 80, 4), dtype=np.float32)
    recorder.append_cycle(
        np.zeros(7),
        observation,
        HostInferenceResult(initial_noise=np.zeros_like(prediction), prediction=prediction),
        execution,
        0,
        TrafficObservationAudit(("participant-000000",), 1, 0, 1.0),
    )
    return recorder.finalize()
