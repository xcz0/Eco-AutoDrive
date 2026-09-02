"""Fixed-slot vector closed-loop evaluation episode execution."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from eco_planner.contracts import evaluation_plan_cycles
from eco_planner.envs import (
    PlannerObservationSpec,
    TrajectoryExecutionRecord,
    VectorEnvReset,
    VectorEnvScenario,
    VectorMetaDriveEnv,
    collate_observations,
)
from eco_planner.envs.array_types import SingleObservation

from ..artifacts import CompletedEpisodeSummary, FailedEpisodeSummary, FailurePhase
from ..config import EvaluationJobConfig, ScenarioConfig
from ..inference import EvaluationAgent
from .lifecycle import (
    EpisodeFailure,
    EpisodeState,
    audit_slot,
    finalize_completed_episode,
    has_traffic,
    persist_failed_episode,
)
from .recorder import EpisodeTraceRecorder


def run_vector_scenarios(
    specs: tuple[ScenarioConfig, ...],
    agent: EvaluationAgent,
    config: EvaluationJobConfig,
    output_root: Path,
    *,
    vector_env_slots: int,
    torch_threads_per_worker: int | None,
) -> tuple[CompletedEpisodeSummary | FailedEpisodeSummary, ...]:
    """Evaluate a scenario queue with persistent fixed slots and batched planning."""

    slot_count = min(vector_env_slots, len(specs))
    initial_specs = specs[:slot_count]
    configured_envs = tuple({**config.env, "map": spec.map} for spec in initial_specs)
    scenarios = tuple(VectorEnvScenario(spec.name, spec.map, spec.seed) for spec in specs)
    with VectorMetaDriveEnv(
        configured_envs,
        mode=config.evaluation.mode,
        observation_spec=PlannerObservationSpec.from_planner_config(agent.planner_config),
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.evaluation.history_warmup_steps,
        scenarios=scenarios,
        torch_threads_per_worker=torch_threads_per_worker,
    ) as envs:
        resets = envs.reset(scenarios[:slot_count])
        slots: dict[int, EpisodeState] = {}
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
                    slots[slot_index] = _initialize_vector_slot(
                        reset, agent, config, scenario_index
                    )
                except EpisodeFailure as failure:
                    summaries[scenario_index] = persist_failed_episode(
                        spec,
                        None,
                        failure,
                        agent,
                        config,
                        output_root,
                        [],
                        scenario_index=scenario_index,
                    )
                    continue
                slot_scenario_indices[slot_index] = scenario_index
                return True
            return False

        for slot_index, reset in enumerate(resets):
            try:
                slots[slot_index] = _initialize_vector_slot(reset, agent, config, slot_index)
            except EpisodeFailure as failure:
                summaries[slot_index] = persist_failed_episode(
                    initial_specs[slot_index],
                    None,
                    failure,
                    agent,
                    config,
                    output_root,
                    [],
                    scenario_index=slot_index,
                )
                assign_next(slot_index)
            else:
                slot_scenario_indices[slot_index] = slot_index
        active = list(slots)
        while active:
            observations = [_state_observation(slots[index]) for index in active]
            decision = agent.decide_batch(
                collate_observations(observations),
                tuple(slots[index].noise_generator for index in active),
            )
            steps = envs.step_slots(active, decision.ego_trajectories)
            audit = decision.audit_result()
            next_active: list[int] = []
            for batch_index, step in enumerate(steps):
                slot_index = active[batch_index]
                slot = slots[slot_index]
                slot.record_cycle(
                    _state_observation(slot),
                    audit_slot(audit, batch_index),
                    step.execution,
                    step.reward,
                    slot.traffic_audit,
                )
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
                            agent,
                            config,
                            output_root,
                            [],
                            scenario_index=scenario_index,
                        )
                    except EpisodeFailure as failure:
                        summaries[scenario_index] = persist_failed_episode(
                            slot.spec,
                            slot.trace,
                            failure,
                            agent,
                            config,
                            output_root,
                            [],
                            trace_arrays,
                            scenario_index=scenario_index,
                        )
                    if assign_next(slot_index):
                        next_active.append(slot_index)
                    continue
                slot.observation = step.observation
                slot.traffic_audit = step.traffic_audit
                slot.anchor = _state_from_execution(step.execution)
                next_active.append(slot_index)
            active = next_active
    return cast(
        tuple[CompletedEpisodeSummary | FailedEpisodeSummary, ...],
        tuple(summaries),
    )


def _initialize_vector_slot(
    reset: VectorEnvReset,
    agent: EvaluationAgent,
    config: EvaluationJobConfig,
    scenario_index: int,
) -> EpisodeState:
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
        guided=agent.guided,
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
    return EpisodeState(
        spec=ScenarioConfig(
            name=reset.scenario.name,
            map=reset.scenario.map,
            seed=reset.scenario.seed,
        ),
        observation=reset.observation,
        traffic_audit=reset.traffic_audit,
        noise_generator=agent.new_noise_generator(scenario_index),
        trace=trace,
        anchor=reset.initial_state,
        route_length_m=reset.route_length_m,
        environment_map_audit=dict(reset.programmatic_lane_speed_limit_audit),
        saw_traffic=has_traffic(reset.traffic_audit),
    )


def _state_from_execution(execution: TrajectoryExecutionRecord) -> np.ndarray:
    state = execution.substep_states[-1]
    return np.asarray(state, dtype=np.float64).copy()


def _state_observation(state: EpisodeState) -> SingleObservation:
    return cast(SingleObservation, state.observation)
