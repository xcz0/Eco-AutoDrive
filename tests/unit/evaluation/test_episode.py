from __future__ import annotations

import zipfile
from dataclasses import replace

import numpy as np
import pytest

from eco_planner.envs import (
    VectorEnvReset,
    VectorEnvScenario,
    VectorEnvStep,
    VectorEnvTiming,
)
from eco_planner.evaluation import execution as evaluation_execution
from eco_planner.evaluation.config import ScenarioConfig, parse_evaluation_config


def test_run_scenario_replans_and_persists_trace(
    tmp_path, fake_runtime: object, evaluation_config: object, patch_episode_dependencies
) -> None:
    patch_episode_dependencies()

    summary = evaluation_execution.run_scenario(
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
    assert summary.energy.total_ml == pytest.approx(1.0)


def test_failed_episode_preserves_energy_from_partial_execution_trace(
    tmp_path,
    fake_runtime: object,
    evaluation_config: object,
    patch_episode_dependencies,
    monkeypatch,
) -> None:
    patch_episode_dependencies()

    def fail_after_execution(*args, **kwargs):
        raise evaluation_execution.EpisodeFailure(
            evaluation_execution.FailurePhase.EXECUTION, RuntimeError("injected")
        )

    monkeypatch.setattr(evaluation_execution, "build_episode_summary", fail_after_execution)

    summary = evaluation_execution.run_scenario(
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
    fake_slot_class: object,
    fake_runtime: object,
    evaluation_config: object,
    patch_episode_dependencies,
) -> None:
    patch_episode_dependencies()

    class FakeVectorEnv:
        instances: list[FakeVectorEnv] = []

        def __init__(self, configs: tuple[dict[str, object], ...], **kwargs: object) -> None:
            self.configs = configs
            self.envs = [fake_slot_class(config) for config in configs]  # type: ignore[operator]
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
            env = fake_slot_class({**self.configs[slot], "map": scenario.map})  # type: ignore[operator]
            self.envs[slot] = env
            reset = env.reset(map_name=scenario.map, seed=scenario.seed)
            observation = env.observe()
            state = env.vehicle_state
            return VectorEnvReset(
                slot,
                scenario,
                observation.observation,
                reset.route_completion,
                reset.route_length_m,
                reset.warmup_initial_state,
                state,
                (),
                None,
                reset.programmatic_lane_speed_limit_audit,
                VectorEnvTiming(0.0, 0.0),
            )

        def step_slots(
            self, slots: list[int], trajectories: np.ndarray
        ) -> tuple[VectorEnvStep, ...]:
            self.step_slot_counts.append(len(slots))
            steps = []
            for slot, trajectory in zip(slots, trajectories, strict=True):
                env = self.envs[slot]
                step = env.step(trajectory)
                execution = step.execution
                terminated = step.terminated
                if env.env.config["map"] in {"Q", "L"}:
                    target_step = 1 if env.env.config["map"] == "Q" else 3
                    terminated = env.env._step == target_step
                    execution = replace(
                        execution,
                        substep_terminated=np.array([False, False, False, False, terminated]),
                        arrive_dest=terminated,
                    )
                observation = env.observe()
                steps.append(
                    VectorEnvStep(
                        slot,
                        observation.observation,
                        step.reward,
                        terminated,
                        step.truncated,
                        execution,
                        None,
                        VectorEnvTiming(0.0, 0.0),
                    )
                )
            return tuple(steps)

    monkeypatch.setattr(evaluation_execution, "VectorMetaDriveEnv", FakeVectorEnv)
    evaluation_config.evaluation.execution.vector_env_slots = 2  # type: ignore[attr-defined]
    evaluation_config.evaluation.evaluated_horizon_steps = 15  # type: ignore[attr-defined]
    evaluation_config.env.horizon = 15  # type: ignore[attr-defined]
    config = parse_evaluation_config(evaluation_config)
    scenarios = (
        ScenarioConfig(name="first", map="Q", seed=3),
        ScenarioConfig(name="second", map="L", seed=3),
        ScenarioConfig(name="third", map="S", seed=3),
    )

    summaries = evaluation_execution.run_vector_scenarios(scenarios, fake_runtime, config, tmp_path)

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

    initialize = evaluation_execution._initialize_vector_slot

    def fail_first_reset(reset, runtime, selected_config):
        if reset.scenario.name == "first":
            raise evaluation_execution.EpisodeFailure(
                evaluation_execution.FailurePhase.RESET, RuntimeError("invalid route length")
            )
        return initialize(reset, runtime, selected_config)

    monkeypatch.setattr(evaluation_execution, "_initialize_vector_slot", fail_first_reset)
    failure_root = tmp_path / "reset-failure"

    failure_summaries = evaluation_execution.run_vector_scenarios(
        scenarios, fake_runtime, config, failure_root
    )

    assert [summary.status for summary in failure_summaries] == ["failed", "completed", "completed"]
    assert failure_summaries[0].failure.phase == evaluation_execution.FailurePhase.RESET
    assert (failure_root / "first" / "summary.json").exists()
    assert (failure_root / "second" / "summary.json").exists()
    assert (failure_root / "third" / "summary.json").exists()
