"""Shared state and artifact lifecycle for evaluation episodes."""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.configuration import ScenarioConfig
from eco_planner.envs import (
    TrafficObservationAudit,
    TrajectoryExecutionRecord,
    TrajectoryExecutionResult,
)

from ..artifacts import (
    CompletedEpisodeSummary,
    FailedEpisodeSummary,
    FailurePhase,
    write_episode_artifacts,
)
from ..artifacts.summary import build_episode_summary, build_failed_episode_summary
from ..config import EvaluationJobConfig
from ..inference import EvaluationAgent
from .recorder import EpisodeTraceRecorder


class EpisodeFailure(RuntimeError):
    """A classified episode failure that may be persisted before continuing the job."""

    def __init__(self, phase: FailurePhase, cause: Exception) -> None:
        self.phase = phase
        self.cause = cause
        super().__init__(f"{phase.value}: {cause}")


@dataclass
class EpisodeState:
    spec: ScenarioConfig
    observation: TensorDictBase | None
    traffic_audit: TrafficObservationAudit | None
    noise_generator: torch.Generator
    trace: EpisodeTraceRecorder
    anchor: np.ndarray
    route_length_m: float
    environment_map_audit: dict[str, object]
    saw_traffic: bool = False
    plan_index: int = 0

    def record_cycle(
        self,
        observation: TensorDictBase,
        inference: TensorDictBase,
        step: TrajectoryExecutionResult,
        traffic_audit: TrafficObservationAudit | None,
    ) -> int:
        cycle = self.plan_index
        self.trace.append_cycle(
            self.anchor,
            observation,
            inference,
            step,
            cycle,
            traffic_audit,
        )
        self.saw_traffic = self.saw_traffic or has_traffic(traffic_audit)
        self.plan_index += 1
        return cycle


def finalize_completed_episode(
    spec: ScenarioConfig,
    trace_arrays: dict[str, np.ndarray],
    final_execution: TrajectoryExecutionRecord,
    terminated: bool,
    truncated: bool,
    environment_map_audit: dict[str, object],
    route_length_m: float,
    saw_traffic: bool,
    agent: EvaluationAgent,
    config: EvaluationJobConfig,
    output_root: Path,
    frames: list[np.ndarray],
    *,
    scenario_index: int,
) -> CompletedEpisodeSummary:
    """Build and persist a completed episode from its execution trace."""

    if config.evaluation.mode == "traffic" and not saw_traffic:
        raise EpisodeFailure(
            FailurePhase.OBSERVATION,
            RuntimeError("traffic evaluation never observed a participant within radius"),
        )
    summary = build_episode_summary(
        _scenario_payload(spec),
        trace_arrays,
        final_execution,
        terminated,
        truncated,
        agent.noise_seed(scenario_index),
        environment_map_audit,
        config.evaluation.mode,
        float(config.env["traffic_density"]),
        route_length_m,
        asdict(agent.sampler_report),
        asdict(agent.guidance_config),
    )
    write_episode_artifacts(output_root / spec.name, trace_arrays, frames, summary, config.video)
    return summary


def persist_failed_episode(
    spec: ScenarioConfig,
    trace: EpisodeTraceRecorder | None,
    failure: EpisodeFailure,
    agent: EvaluationAgent,
    config: EvaluationJobConfig,
    output_root: Path,
    frames: list[np.ndarray],
    finalized_trace_arrays: dict[str, np.ndarray] | None = None,
    *,
    scenario_index: int,
) -> FailedEpisodeSummary:
    """Persist the available trace and failure metadata through one common path."""

    trace_status = "partial" if trace is not None and trace.has_recorded_steps else "empty"
    if finalized_trace_arrays is None:
        recorder = trace if trace is not None else EpisodeTraceRecorder.empty()
        trace_arrays = recorder.finalize(trace_status)
    else:
        trace_arrays = dict(finalized_trace_arrays)
        trace_arrays["trace_status"] = np.asarray(trace_status)
    summary = build_failed_episode_summary(
        _scenario_payload(spec),
        noise_seed=agent.noise_seed(scenario_index),
        evaluation_mode=config.evaluation.mode,
        traffic_density=float(config.env["traffic_density"]),
        sampler=asdict(agent.sampler_report),
        guidance=asdict(agent.guidance_config),
        trace_status=trace_status,
        phase=failure.phase,
        cause=failure.cause,
        traceback_text=traceback.format_exc(),
        trace_arrays=trace_arrays,
    )
    write_episode_artifacts(output_root / spec.name, trace_arrays, frames, summary, config.video)
    return summary


def has_traffic(audit: TrafficObservationAudit | None) -> bool:
    return audit is not None and audit.participant_count_in_radius > 0


def audit_slot(audit: TensorDictBase, index: int) -> TensorDictBase:
    return cast(TensorDictBase, audit[index : index + 1])


def _scenario_payload(spec: ScenarioConfig) -> dict[str, object]:
    return {"name": spec.name, "map_sequence": spec.map, "seed": spec.seed}
