"""Fast evaluation smoke coverage at the runner/artifact boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from eco_planner.envs import (
    EnvSlotObservation,
    EnvSlotReset,
    EnvSlotStep,
    TrajectoryExecutionRecord,
)
from eco_planner.evaluation import execution, runner
from eco_planner.evaluation.artifacts import (
    load_episode_summary,
    load_job_summary,
    load_trace_artifact,
)
from eco_planner.evaluation.config import parse_evaluation_config
from eco_planner.evaluation.runtime import InferenceDecision, InferenceRuntimeReport
from eco_planner.models import CheckpointLoadReport, NoGuidanceConfig, SamplerReport
from eco_planner.runtime.contracts import HostTrajectories


@pytest.mark.smoke
def test_runner_writes_readable_finite_short_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _ShortEpisodeRuntime()
    monkeypatch.setattr(execution, "MetaDriveEnvSlot", _ShortEpisodeSlot)
    monkeypatch.setattr(runner, "create_fabric_inference_runtime", lambda *_: runtime)
    monkeypatch.setattr(runner, "write_runtime_metadata", lambda *_: None)

    summary = runner.run_evaluation(parse_evaluation_config(_config()), tmp_path)

    job = load_job_summary(tmp_path / "summary.json")
    episode = load_episode_summary(tmp_path / "short" / "summary.json")
    trace = load_trace_artifact(tmp_path / "short" / "trace.npz")
    assert summary == job
    assert job.status == episode.status == "completed"
    assert episode.plan_cycles == 1
    assert episode.simulator_steps == 5
    assert episode.energy is not None
    assert episode.energy.total_ml == pytest.approx(0.5)
    assert trace.trace_status == "complete"
    assert np.isfinite(trace.arrays["initial_noise"]).all()
    assert np.isfinite(trace.arrays["predictions_local"]).all()
    assert np.isfinite(trace.arrays["executed_fuel_proxy_step_energy_ml"]).all()


def test_vector_topology_derives_capacity_from_resource_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config.evaluation.execution.topology = "vector"
    config.resources = {
        "name": "test-host",
        "rollout_worker_count": 2,
        "evaluation_job_worker_count": 3,
        "evaluation_vector_env_slots": 4,
        "torch_threads_per_worker": 5,
    }
    configured_threads: list[int] = []
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(runner.torch, "set_num_threads", configured_threads.append)

    report = runner.configure_job_execution(parse_evaluation_config(config))

    assert report.mode == "serial"
    assert report.launcher == "basic"
    assert report.worker_count == 1
    assert report.vector_env_slots == 4
    assert report.torch_threads_per_worker == 5
    assert report.resource_profile == "test-host"
    assert configured_threads == [5]


class _ShortEpisodeSlot:
    def __init__(self, *_: object, **__: object) -> None:
        self.env = SimpleNamespace()
        self._state = np.zeros(7, dtype=np.float64)

    @property
    def vehicle_state(self) -> np.ndarray:
        return self._state.copy()

    def reset(self, *, map_name: str, seed: int) -> EnvSlotReset:
        assert (map_name, seed) == ("S", 3)
        return EnvSlotReset(
            route_completion=0.0,
            route_length_m=100.0,
            warmup_initial_state=self.vehicle_state,
            programmatic_lane_speed_limit_audit={
                "speed_limit_sentinel_replaced_count": 18,
                "speed_limit_existing_preserved_count": 0,
                "configured_programmatic_lane_speed_limit_kmh": 50.0,
                "lane_speed_limit_kmh_counts": {"50": 18},
            },
        )

    def observe(self) -> EnvSlotObservation:
        return EnvSlotObservation(_observation(), None)

    def step(self, trajectory: np.ndarray) -> EnvSlotStep:
        assert trajectory.shape == (80, 4)
        self._state = np.array([5.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0])
        return EnvSlotStep(5.0, True, False, _execution_record())

    def close(self) -> None:
        pass


class _ShortEpisodeRuntime:
    planner_config = SimpleNamespace(
        predicted_neighbor_num=10,
        future_len=80,
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
    )
    report = InferenceRuntimeReport(
        requested_accelerator="cpu",
        resolved_accelerator="cpu",
        requested_precision="32-true",
        resolved_precision="32-true",
        device="cpu",
        seed=7,
        world_size=1,
    )
    checkpoint_report = CheckpointLoadReport(276, 6_042_628)
    sampler_report = SamplerReport(
        name="dpm10",
        implementation="diffusers",
        num_steps=10,
        timesteps=None,
        initial_noise_scale=0.5,
        ddim_stochasticity=0.0,
        parity_label="official_diffusion_planner_baseline",
    )
    guidance_config = NoGuidanceConfig()

    def new_noise_generator(self) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(self.report.seed)

    def infer(
        self, observation: dict[str, torch.Tensor], generator: torch.Generator
    ) -> InferenceDecision:
        assert observation["ego_current_state"].shape == (1, 10)
        noise = torch.randn((1, 11, 80, 4), generator=generator)
        prediction = torch.zeros_like(noise)
        prediction[..., 2] = 1.0
        audit = TensorDict({"initial_noise": noise, "prediction": prediction}, batch_size=[1])
        return InferenceDecision(HostTrajectories(prediction[:, 0].numpy()), lambda: audit)


def _config() -> object:
    return OmegaConf.create(
        {
            "name": "short-evaluation",
            "evaluation": {
                "mode": "no_traffic",
                "profile": "standard",
                "history_warmup_steps": 0,
                "evaluated_horizon_steps": 5,
                "execution": {
                    "topology": "serial",
                    "deterministic": False,
                },
            },
            "env": {"traffic_density": 0.0, "trajectory_execution_steps": 5, "horizon": 5},
            "map_query_radius_m": 100.0,
            "model": {"args_path": "args.json", "checkpoint_path": "model.pth"},
            "runtime": {"accelerator": "cpu", "precision": "32-true", "seed": 7},
            "sampler": {"name": "dpm10", "implementation": "diffusers"},
            "guidance": {"name": "none"},
            "scenarios": [{"name": "short", "map": "S", "seed": 3}],
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


def _observation() -> dict[str, torch.Tensor]:
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


def _execution_record() -> TrajectoryExecutionRecord:
    positions = np.column_stack((np.arange(1, 6, dtype=np.float64), np.zeros(5)))
    states = np.column_stack(
        (positions, np.zeros(5), np.full(5, 10.0), np.zeros(5), np.full(5, 10.0), np.zeros(5))
    )
    return TrajectoryExecutionRecord(
        start_center=np.zeros(2),
        start_heading=0.0,
        world_centers=np.column_stack((np.arange(1, 81), np.zeros(80))),
        world_headings=np.zeros(80),
        substep_states=states,
        target_centers=positions,
        target_headings=np.zeros(5),
        position_errors_m=np.zeros(5),
        heading_errors_rad=np.zeros(5),
        substep_rewards=np.ones(5),
        substep_dense_rewards=np.ones(5),
        substep_native_energy_ml=np.full(5, 0.1),
        substep_native_episode_energy_ml=np.arange(1, 6, dtype=np.float64) / 10.0,
        substep_executed_fuel_proxy_energy_ml=np.full(5, 0.1),
        substep_distance_m=np.ones(5),
        substep_terminated=np.array([False, False, False, False, True]),
        substep_truncated=np.zeros(5, dtype=np.bool_),
        traffic_frames=(),
        route_completion=1.0,
        arrive_dest=True,
        out_of_road=False,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        max_step=False,
    )
