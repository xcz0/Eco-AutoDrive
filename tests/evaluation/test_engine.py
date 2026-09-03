"""Fast evaluation smoke coverage at the engine/artifact boundary."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict, TensorDictBase

import eco_planner.evaluation.engine as engine
import eco_planner.evaluation.episodes.serial as episode_engine
from eco_planner.envs import (
    EnvSlotObservation,
    EnvSlotReset,
    TrajectoryExecutionRecord,
    TrajectoryExecutionResult,
)
from eco_planner.envs.domain import (
    EnergyMetrics,
    TrafficFrame,
    TransitionMetricInput,
    TransitionMetrics,
)
from eco_planner.evaluation import (
    InferenceDecision,
    load_episode_summary,
    load_job_summary,
    load_trace_artifact,
    parse_evaluation_config,
)
from eco_planner.evaluation.artifacts import (
    validate_episode_artifact,
    validate_matrix_episode,
)
from eco_planner.models import CheckpointLoadReport, NoGuidanceConfig, SamplerReport
from eco_planner.runtime.contracts import HostTrajectories
from eco_planner.runtime.fabric import InferenceRuntimeReport


@pytest.mark.smoke
def test_runner_writes_readable_finite_short_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _ShortEpisodeRuntime()
    monkeypatch.setattr(episode_engine, "MetaDriveEnvSlot", _ShortEpisodeSlot)
    monkeypatch.setattr(engine, "create_fabric_inference_runtime", lambda *_: runtime)
    monkeypatch.setattr(engine, "write_runtime_metadata", lambda *_: None)

    summary = engine.run_evaluation(parse_evaluation_config(_config()), tmp_path)

    job = load_job_summary(tmp_path / "summary.json")
    episode = load_episode_summary(tmp_path / "short" / "summary.json")
    trace = load_trace_artifact(tmp_path / "short" / "trace.npz")
    assert summary == job
    assert job.status == episode.status == "completed"
    assert episode.plan_cycles == 1
    assert episode.simulator_steps == 5
    assert episode.metrics.energy.total_ml == pytest.approx(0.5)
    assert episode.metrics.stopped_fraction == 0.0
    assert episode.metrics.aggregation_unit == "evaluation_episode"
    assert episode.metrics.distance_m == pytest.approx(5.0)
    assert episode.metrics.energy.ml_per_km == pytest.approx(100.0)
    assert trace.trace_status == "complete"
    assert "artifact_schema_version" not in trace.arrays
    assert np.isfinite(trace.arrays["initial_noise"]).all()
    assert np.isfinite(trace.arrays["predictions_local"]).all()
    assert np.isfinite(trace.arrays["executed_fuel_proxy_step_energy_ml"]).all()

    mismatched = episode.model_copy(update={"plan_cycles": 2})
    with pytest.raises(ValueError, match="planning cycle count"):
        validate_episode_artifact(
            tmp_path / "short" / "trace.npz",
            mismatched,
            warmup_steps=0,
            require_traffic=False,
        )

    paired = episode.model_copy(update={"noise_seed": 3, "route_length_m": 2_500.0})
    with pytest.raises(ValueError, match="density disagrees"):
        validate_matrix_episode(paired, tmp_path, seed=3, density=0.5)


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
    monkeypatch.setattr(engine.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(engine.torch, "set_num_threads", configured_threads.append)

    report = engine.configure_job_execution(parse_evaluation_config(config))

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

    def step(self, trajectory: np.ndarray) -> TrajectoryExecutionResult:
        assert trajectory.shape == (80, 4)
        self._state = np.array([5.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0])
        return _step_result()

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

    def infer(self, observation: TensorDictBase, generator: torch.Generator) -> InferenceDecision:
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


def _observation() -> TensorDictBase:
    return TensorDict(
        {
            "ego_current_state": torch.zeros(10),
            "neighbor_agents_past": torch.zeros((32, 21, 11)),
            "static_objects": torch.zeros((5, 10)),
            "lanes": torch.zeros((70, 20, 12)),
            "lanes_speed_limit": torch.full((70, 1), 50.0 / 3.6),
            "lanes_has_speed_limit": torch.ones((70, 1), dtype=torch.bool),
            "route_lanes": torch.zeros((25, 20, 12)),
            "route_lanes_speed_limit": torch.full((25, 1), 50.0 / 3.6),
            "route_lanes_has_speed_limit": torch.ones((25, 1), dtype=torch.bool),
        },
        batch_size=[],
    )


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


def _step_result() -> TrajectoryExecutionResult:
    metrics = tuple(_transition_metrics(index) for index in range(5))
    return TrajectoryExecutionResult(
        metrics=metrics,
        terminated=True,
        truncated=False,
        execution=_execution_record(),
    )


def _transition_metrics(index: int) -> TransitionMetrics:
    position = (float(index + 1), 0.0)
    transition_input = TransitionMetricInput(
        previous_position_xy_m=(float(index), 0.0),
        position_xy_m=position,
        previous_velocity_xy_mps=(10.0, 0.0),
        velocity_xy_mps=(10.0, 0.0),
        previous_acceleration_xy_mps2=(0.0, 0.0),
        heading_rad=0.0,
        yaw_rate_radps=0.0,
        route_progress_delta_m=1.0,
        route_heading_rad=0.0,
        speed_limit_mps=10.0,
        ego_width_m=2.0,
        ego_length_m=4.0,
        traffic_frame=TrafficFrame(index + 1, position, 0.0, 1.0, (), ()),
        target_position_xy_m=position,
        target_heading_rad=0.0,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        crash_sidewalk=False,
        out_of_road=False,
        native_step_energy_ml=0.1,
        native_episode_energy_ml=float(index + 1) / 10.0,
        timestep_s=0.1,
    )
    return TransitionMetrics(
        input=transition_input,
        speed_mps=10.0,
        longitudinal_acceleration_mps2=0.0,
        lateral_acceleration_mps2=0.0,
        jerk_mps3=0.0,
        step_distance_m=1.0,
        position_error_m=0.0,
        heading_error_rad=0.0,
        energy=EnergyMetrics("metadrive_fuel_proxy", 1.0, None, 0.1),
    )


def test_execution_record_contains_only_execution_facts() -> None:
    names = {field.name for field in fields(TrajectoryExecutionRecord)}
    assert names.isdisjoint(
        {
            "position_errors_m",
            "heading_errors_rad",
            "substep_native_energy_ml",
            "substep_native_episode_energy_ml",
            "substep_executed_fuel_proxy_energy_ml",
            "substep_distance_m",
            "substep_metrics",
        }
    )
