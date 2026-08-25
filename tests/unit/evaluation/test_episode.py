from __future__ import annotations

import zipfile
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from eco_planner.envs import (
    TrajectoryExecutionRecord,
    VectorEnvReset,
    VectorEnvScenario,
    VectorEnvStep,
    VectorEnvTiming,
)
from eco_planner.envs.traffic_state import TrafficFrame
from eco_planner.evaluation import episode
from eco_planner.evaluation.artifacts.trace_recorder import EpisodeTraceRecorder
from eco_planner.evaluation.config import ScenarioConfig, parse_evaluation_config
from eco_planner.evaluation.execution import serial, vector


def test_run_scenario_replans_and_persists_trace(
    tmp_path, fake_runtime: object, evaluation_config: object, patch_episode_dependencies
) -> None:
    patch_episode_dependencies()

    summary = serial.run_scenario(
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
    assert summary.energy.metric == "metadrive_fuel_proxy"
    assert summary.energy.distance_m == pytest.approx(10.0)
    assert summary.energy.total_ml > 0.0


def test_failed_episode_preserves_energy_from_partial_execution_trace(
    tmp_path,
    fake_runtime: object,
    evaluation_config: object,
    patch_episode_dependencies,
    monkeypatch,
) -> None:
    patch_episode_dependencies()

    def fail_after_execution(*args, **kwargs):
        raise episode.EpisodeFailure(episode.FailurePhase.EXECUTION, RuntimeError("injected"))

    monkeypatch.setattr(episode, "build_episode_summary", fail_after_execution)

    summary = serial.run_scenario(
        ScenarioConfig(name="fake", map="S", seed=3),
        fake_runtime,
        parse_evaluation_config(evaluation_config),
        tmp_path,
    )

    assert summary.status == "failed"
    assert summary.trace_status == "partial"
    assert summary.energy is not None
    assert summary.energy.distance_m == pytest.approx(10.0)


def test_vector_evaluation_batches_slots_and_writes_independent_traces(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    fake_env_class: object,
    fake_runtime: object,
    evaluation_config: object,
    patch_episode_dependencies,
) -> None:
    patch_episode_dependencies()

    class FakeVectorEnv:
        instances: list[FakeVectorEnv] = []

        def __init__(self, configs: tuple[dict[str, object], ...], **kwargs: object) -> None:
            self.configs = configs
            self.envs = [fake_env_class(config) for config in configs]  # type: ignore[operator]
            self.reset_at_slots: list[int] = []
            self.step_slot_counts: list[int] = []
            self.instances.append(self)

        def __enter__(self) -> FakeVectorEnv:
            return self

        def __exit__(self, *args: object) -> None:
            for env in self.envs:
                env.close()

        def reset(self, scenarios: tuple[VectorEnvScenario, ...]) -> tuple[VectorEnvReset, ...]:
            return tuple(
                self._reset_slot(slot, scenario) for slot, scenario in enumerate(scenarios)
            )

        def reset_at(self, slot: int, scenario: VectorEnvScenario) -> VectorEnvReset:
            self.reset_at_slots.append(slot)
            return self._reset_slot(slot, scenario)

        def _reset_slot(self, slot: int, scenario: VectorEnvScenario) -> VectorEnvReset:
            env = fake_env_class({**self.configs[slot], "map": scenario.map})  # type: ignore[operator]
            self.envs[slot] = env
            env.reset(scenario.seed)
            adapter = serial.NoTrafficMetaDriveObservationAdapter(None, 100.0)
            adapter.reset(env)
            state = episode.vehicle_state(env)
            return VectorEnvReset(
                slot,
                scenario,
                adapter.build(env),
                0.0,
                100.0,
                state,
                state,
                (),
                None,
                env.programmatic_lane_speed_limit_audit,
                VectorEnvTiming(0.0, 0.0, 0.0, 0.0, 0.0),
            )

        def step_slots(
            self, slots: list[int], trajectories: np.ndarray
        ) -> tuple[VectorEnvStep, ...]:
            self.step_slot_counts.append(len(slots))
            steps = []
            for slot, trajectory in zip(slots, trajectories, strict=True):
                env = self.envs[slot]
                _, reward, terminated, truncated, info = env.step(trajectory)
                if env.config["map"] in {"Q", "L"}:
                    target_step = 1 if env.config["map"] == "Q" else 3
                    terminated = env._step == target_step
                    execution = replace(
                        info["trajectory_execution"],
                        substep_terminated=np.array([False, False, False, False, terminated]),
                        arrive_dest=terminated,
                    )
                    info = {"trajectory_execution": execution}
                adapter = serial.NoTrafficMetaDriveObservationAdapter(None, 100.0)
                steps.append(
                    VectorEnvStep(
                        slot,
                        adapter.build(env),
                        reward,
                        terminated,
                        truncated,
                        info["trajectory_execution"],
                        None,
                        VectorEnvTiming(0.0, 0.0, 0.0, 0.0, 0.0),
                    )
                )
            return tuple(steps)

    monkeypatch.setattr(vector, "VectorMetaDriveEnv", FakeVectorEnv)
    evaluation_config.evaluation.execution.vector_env_slots = 2  # type: ignore[attr-defined]
    evaluation_config.evaluation.evaluated_horizon_steps = 15  # type: ignore[attr-defined]
    evaluation_config.env.horizon = 15  # type: ignore[attr-defined]
    config = parse_evaluation_config(evaluation_config)
    scenarios = (
        ScenarioConfig(name="first", map="Q", seed=3),
        ScenarioConfig(name="second", map="L", seed=3),
        ScenarioConfig(name="third", map="S", seed=3),
    )

    summaries = vector.run_vector_scenarios(scenarios, fake_runtime, config, tmp_path)

    assert [summary.scenario.name for summary in summaries] == ["first", "second", "third"]
    assert [(summary.plan_cycles, summary.simulator_steps) for summary in summaries] == [
        (1, 5),
        (3, 15),
        (2, 10),
    ]
    assert len(FakeVectorEnv.instances) == 1
    assert FakeVectorEnv.instances[0].reset_at_slots == [0]
    assert FakeVectorEnv.instances[0].step_slot_counts == [2, 2, 2]
    for scenario, summary in zip(scenarios, summaries, strict=True):
        with np.load(tmp_path / scenario.name / "trace.npz") as trace:
            assert trace["initial_noise"].shape == (summary.plan_cycles, 11, 80, 4)

    initialize = vector._initialize_vector_slot

    def fail_first_reset(reset, runtime, selected_config):
        if reset.scenario.name == "first":
            raise episode.EpisodeFailure(
                episode.FailurePhase.RESET, RuntimeError("invalid route length")
            )
        return initialize(reset, runtime, selected_config)

    monkeypatch.setattr(vector, "_initialize_vector_slot", fail_first_reset)
    failure_root = tmp_path / "reset-failure"

    failure_summaries = vector.run_vector_scenarios(scenarios, fake_runtime, config, failure_root)

    assert [summary.status for summary in failure_summaries] == ["failed", "completed", "completed"]
    assert failure_summaries[0].failure.phase == episode.FailurePhase.RESET
    assert (failure_root / "first" / "summary.json").exists()
    assert (failure_root / "second" / "summary.json").exists()
    assert (failure_root / "third" / "summary.json").exists()


def test_route_length_accepts_finite_numpy_lane_scalar() -> None:
    lane = SimpleNamespace(length=np.float32(123.5))
    env = SimpleNamespace(
        agent=SimpleNamespace(navigation=SimpleNamespace(checkpoints=["start", "end"])),
        current_map=SimpleNamespace(road_network=SimpleNamespace(graph={"start": {"end": [lane]}})),
    )
    assert episode.route_length_m(env) == pytest.approx(123.5)


def test_stationary_trajectory_satisfies_execution_contract() -> None:
    trajectory = episode.stationary_trajectory()

    assert trajectory.shape == (80, 4)
    assert trajectory.dtype == np.float32
    assert np.isfinite(trajectory).all()
    assert np.all(np.linalg.norm(trajectory[:, 2:4], axis=-1) > 0.0)


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
                {"trajectory_execution": _warmup_execution(frames)},
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


def _warmup_execution(frames: tuple[TrafficFrame, ...]) -> TrajectoryExecutionRecord:
    states = np.zeros((5, 7))
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
