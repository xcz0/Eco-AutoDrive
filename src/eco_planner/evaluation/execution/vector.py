"""Fixed-slot vector closed-loop evaluation scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from eco_planner.envs import (
    TrafficObservationAudit,
    TrajectoryExecutionRecord,
    VectorEnvReset,
    VectorEnvScenario,
    VectorMetaDriveEnv,
    collate_observations,
)
from eco_planner.evaluation.artifacts.models import (
    CompletedEpisodeSummary,
    FailedEpisodeSummary,
    FailurePhase,
)
from eco_planner.evaluation.artifacts.trace_recorder import EpisodeTraceRecorder
from eco_planner.evaluation.config import EvaluationJobConfig, ScenarioConfig
from eco_planner.evaluation.episode import finalize_completed_episode, persist_failed_episode
from eco_planner.evaluation.failures import EpisodeFailure
from eco_planner.evaluation.runtime.engine import FabricInferenceRuntime
from eco_planner.execution_contracts import evaluation_plan_cycles
from eco_planner.models import NoGuidanceConfig


@dataclass
class _VectorEvaluationSlot:
    spec: ScenarioConfig
    observation: dict[str, torch.Tensor]
    traffic_audit: TrafficObservationAudit | None
    generator: torch.Generator
    trace: EpisodeTraceRecorder
    anchor: np.ndarray
    route_length_m: float
    environment_map_audit: dict[str, object]
    saw_traffic: bool
    total_reward: float = 0.0
    plan_index: int = 0


def run_vector_scenarios(
    specs: tuple[ScenarioConfig, ...],
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    output_root: Path,
) -> tuple[CompletedEpisodeSummary | FailedEpisodeSummary, ...]:
    """Evaluate a scenario queue with persistent fixed slots and batched planning."""

    if not specs:
        raise ValueError("vector evaluation requires at least one scenario")
    if config.video.enabled:
        raise ValueError("vector evaluation requires video.enabled=false")
    configured_slots = config.evaluation.execution.vector_env_slots or len(specs)
    slot_count = min(configured_slots, len(specs))
    initial_specs = specs[:slot_count]
    configured_envs = tuple({**config.env, "map": spec.map} for spec in initial_specs)
    initial_scenarios = tuple(
        VectorEnvScenario(spec.name, spec.map, spec.seed) for spec in initial_specs
    )
    with VectorMetaDriveEnv(
        configured_envs,
        mode=config.evaluation.mode,
        model_config=runtime.planner_config,
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.evaluation.history_warmup_steps,
    ) as envs:
        resets = envs.reset(initial_scenarios)
        slots: dict[int, _VectorEvaluationSlot] = {}
        slot_scenario_indices: dict[int, int] = {}
        summaries: list[CompletedEpisodeSummary | FailedEpisodeSummary | None] = [None] * len(specs)
        next_scenario_index = slot_count

        def assign_next(slot_index: int) -> bool:
            nonlocal next_scenario_index
            while next_scenario_index < len(specs):
                scenario_index = next_scenario_index
                spec = specs[scenario_index]
                next_scenario_index += 1
                reset = envs.reset_at(slot_index, VectorEnvScenario(spec.name, spec.map, spec.seed))
                try:
                    slots[slot_index] = _initialize_vector_slot(reset, runtime, config)
                except EpisodeFailure as failure:
                    summaries[scenario_index] = persist_failed_episode(
                        spec,
                        None,
                        failure,
                        runtime,
                        config,
                        output_root,
                        [],
                    )
                    continue
                slot_scenario_indices[slot_index] = scenario_index
                return True
            return False

        for slot_index, reset in enumerate(resets):
            try:
                slots[slot_index] = _initialize_vector_slot(reset, runtime, config)
            except EpisodeFailure as failure:
                summaries[slot_index] = persist_failed_episode(
                    initial_specs[slot_index],
                    None,
                    failure,
                    runtime,
                    config,
                    output_root,
                    [],
                )
                assign_next(slot_index)
            else:
                slot_scenario_indices[slot_index] = slot_index
        active = list(slots)
        while active:
            observations = [slots[index].observation for index in active]
            generators = tuple(slots[index].generator for index in active)
            noise = _batch_noise(runtime, generators)
            decision = runtime.infer_batch(collate_observations(observations), noise, generators)
            steps = envs.step_slots(active, decision.ego_trajectories)
            audit = decision.audit_result()
            next_active: list[int] = []
            for batch_index, step in enumerate(steps):
                slot_index = active[batch_index]
                slot = slots[slot_index]
                slot.trace.append_cycle(
                    slot.anchor,
                    slot.observation,
                    audit[batch_index : batch_index + 1],
                    step.execution,
                    slot.plan_index,
                    slot.traffic_audit,
                )
                slot.saw_traffic = slot.saw_traffic or _has_traffic(slot.traffic_audit)
                slot.total_reward += step.reward
                slot.plan_index += 1
                if step.terminated or step.truncated:
                    scenario_index = slot_scenario_indices.pop(slot_index)
                    trace_arrays = slot.trace.finalize()
                    try:
                        summaries[scenario_index] = finalize_completed_episode(
                            slot.spec,
                            trace_arrays,
                            step.execution,
                            step.terminated,
                            step.truncated,
                            slot.total_reward,
                            slot.environment_map_audit,
                            slot.route_length_m,
                            slot.saw_traffic,
                            runtime,
                            config,
                            output_root,
                            [],
                        )
                    except EpisodeFailure as failure:
                        summaries[scenario_index] = persist_failed_episode(
                            slot.spec,
                            slot.trace,
                            failure,
                            runtime,
                            config,
                            output_root,
                            [],
                            trace_arrays,
                        )
                    if assign_next(slot_index):
                        next_active.append(slot_index)
                    continue
                slot.observation = step.observation
                slot.traffic_audit = step.traffic_audit
                slot.anchor = _state_from_execution(step.execution)
                next_active.append(slot_index)
            active = next_active
    if any(summary is None for summary in summaries):
        raise RuntimeError("vector evaluation ended with an unfinished scenario")
    return tuple(summary for summary in summaries if summary is not None)


def _initialize_vector_slot(
    reset: VectorEnvReset,
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
) -> _VectorEvaluationSlot:
    if config.evaluation.mode == "traffic" and not 2_000.0 <= reset.route_length_m <= 5_000.0:
        raise EpisodeFailure(
            FailurePhase.RESET,
            RuntimeError(
                f"traffic evaluation route length {reset.route_length_m} m is outside [2000, 5000]"
            ),
        )
    trace = EpisodeTraceRecorder.from_initial_state(
        reset.warmup_initial_state,
        max_plan_cycles=evaluation_plan_cycles(config.evaluation.evaluated_horizon_steps),
        max_warmup_steps=config.evaluation.history_warmup_steps,
        guided=not isinstance(runtime.guidance_config, NoGuidanceConfig),
    )
    for execution in reset.warmup_executions:
        trace.append_warmup(
            execution,
            np.asarray(
                [len(frame.participants) for frame in execution.traffic_frames], dtype=np.int64
            ),
            np.asarray(
                [len(frame.static_objects) for frame in execution.traffic_frames], dtype=np.int64
            ),
        )
    trace.replace_initial_state(reset.initial_state)
    return _VectorEvaluationSlot(
        ScenarioConfig(
            name=reset.scenario.name,
            map=reset.scenario.map,
            seed=reset.scenario.seed,
        ),
        reset.observation,
        reset.traffic_audit,
        runtime.new_noise_generator(),
        trace,
        reset.initial_state,
        reset.route_length_m,
        dict(reset.programmatic_lane_speed_limit_audit),
        _has_traffic(reset.traffic_audit),
    )


def _batch_noise(
    runtime: FabricInferenceRuntime, generators: tuple[torch.Generator, ...]
) -> torch.Tensor:
    config = runtime.planner_config
    return torch.cat(
        [
            torch.randn(
                (1, 1 + config.predicted_neighbor_num, config.future_len, 4),
                dtype=torch.float32,
                device=runtime.device,
                generator=generator,
            )
            for generator in generators
        ]
    )


def _state_from_execution(execution: TrajectoryExecutionRecord) -> np.ndarray:
    state = execution.substep_states[-1]
    if state.shape != (7,):
        raise RuntimeError("trajectory execution did not return a [7] final state")
    return np.asarray(state, dtype=np.float64).copy()


def _has_traffic(audit: TrafficObservationAudit | None) -> bool:
    return audit is not None and audit.participant_count_in_radius > 0
