"""Typed domain models for closed-loop evaluation results."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)


class ArtifactModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )


class ScenarioSummary(ArtifactModel):
    name: str = Field(min_length=1)
    map_sequence: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)


class WorkloadScenario(ArtifactModel):
    """Configured map/seed pair retained with the evaluation job workload."""

    name: str = Field(min_length=1)
    map: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)


class MatrixWorkload(ArtifactModel):
    """The declared evaluation matrix grid retained with every job result."""

    seeds: tuple[StrictInt, ...]
    traffic_densities: tuple[StrictFloat, ...]

    @model_validator(mode="after")
    def validate_grid(self) -> MatrixWorkload:
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("matrix seeds must be non-empty non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("matrix seeds must be unique")
        if not self.traffic_densities or any(
            density <= 0.0 or density > 1.0 for density in self.traffic_densities
        ):
            raise ValueError("matrix traffic densities must be in (0, 1]")
        if len(set(self.traffic_densities)) != len(self.traffic_densities):
            raise ValueError("matrix traffic densities must be unique")
        return self


class EvaluationWorkload(ArtifactModel):
    """Typed workload provenance required to interpret one evaluation result."""

    mode: Literal["no_traffic", "traffic"]
    profile: str = Field(min_length=1)
    history_warmup_steps: StrictInt = Field(ge=0)
    evaluated_horizon_steps: StrictInt = Field(gt=0)
    scenarios: tuple[WorkloadScenario, ...] = Field(min_length=1)
    matrix: MatrixWorkload | None = None
    video_enabled: StrictBool


class TerminationSummary(ArtifactModel):
    type: Literal[
        "arrive_dest",
        "time_truncation",
        "collision",
        "out_of_road",
        "runtime_error",
    ]
    detail: str = Field(min_length=1)


class FailurePhase(str, Enum):
    RESET = "reset"
    WARMUP = "warmup"
    OBSERVATION = "observation"
    INFERENCE = "inference"
    EXECUTION = "execution"


class FailureSummary(ArtifactModel):
    phase: FailurePhase
    exception_type: str = Field(min_length=1)
    message: str
    traceback: str = Field(min_length=1)


class InferenceRuntimeSummary(ArtifactModel):
    requested_accelerator: Literal["auto", "cpu", "cuda"]
    resolved_accelerator: Literal["cpu", "cuda"]
    requested_precision: Literal["auto", "32-true", "16-mixed", "bf16-mixed"]
    resolved_precision: Literal["32-true", "16-mixed", "bf16-mixed"]
    device: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    world_size: Literal[1]


class CheckpointSummary(ArtifactModel):
    ema_tensor_count: StrictInt = Field(gt=0)
    parameter_count: StrictInt = Field(gt=0)


class SamplerSummary(ArtifactModel):
    name: Literal["dpm10", "ddim5"]
    implementation: Literal["diffusers"]
    num_steps: StrictInt = Field(gt=0)
    timesteps: tuple[StrictFloat, ...] | None
    initial_noise_scale: StrictFloat = Field(gt=0.0)
    ddim_stochasticity: StrictFloat = Field(ge=0.0, le=1.0)
    parity_label: str = Field(min_length=1)


class NoGuidanceSummary(ArtifactModel):
    name: Literal["none"]


class OrthogonalGuidanceSummary(ArtifactModel):
    name: Literal["orthogonal_reference"]
    formula_label: Literal["centered_energy_gradient_delta_v1"]
    lateral_scale: StrictFloat = Field(ge=-1.0, le=1.0)
    longitudinal_scale: StrictFloat = Field(ge=-1.0, le=1.0)
    lateral_max_offset_m: StrictFloat = Field(gt=0.0)
    longitudinal_max_speed_fraction: StrictFloat = Field(gt=0.0)
    trajectory_dt_s: StrictFloat = Field(gt=0.0)
    gradient_step_coefficient: StrictFloat = Field(ge=1.0, le=1.0)
    reference_refresh_cycles: Literal[1]
    share_scene_encoding: Literal[True]
    share_initial_noise: Literal[True]
    share_transition_noise: Literal[True]
    heading_norm_epsilon: StrictFloat = Field(gt=0.0)
    zero_speed_tolerance_mps: StrictFloat = Field(gt=0.0)


class OrthogonalPolicyGuidanceSummary(ArtifactModel):
    name: Literal["orthogonal_policy"]
    formula_label: Literal["centered_energy_gradient_delta_v1"]
    lateral_max_offset_m: StrictFloat = Field(gt=0.0)
    longitudinal_max_speed_fraction: StrictFloat = Field(gt=0.0)
    trajectory_dt_s: StrictFloat = Field(gt=0.0)
    gradient_step_coefficient: StrictFloat = Field(ge=1.0, le=1.0)
    reference_refresh_cycles: Literal[1]
    share_scene_encoding: Literal[True]
    share_initial_noise: Literal[True]
    share_transition_noise: Literal[True]
    heading_norm_epsilon: StrictFloat = Field(gt=0.0)
    zero_speed_tolerance_mps: StrictFloat = Field(gt=0.0)


GuidanceSummary = Annotated[
    NoGuidanceSummary | OrthogonalGuidanceSummary | OrthogonalPolicyGuidanceSummary,
    Field(discriminator="name"),
]


class ExecutionSummary(ArtifactModel):
    mode: Literal["serial", "parallel"]
    launcher: Literal["basic", "joblib"]
    worker_count: StrictInt = Field(gt=0)
    vector_env_slots: StrictInt | None = Field(default=None, gt=0)
    torch_threads_per_worker: StrictInt | None = Field(default=None, gt=0)
    deterministic: StrictBool
    resolved_accelerator: Literal["cpu", "cuda"]
    process_id: StrictInt = Field(gt=0)
    logical_cpu_count: StrictInt = Field(gt=0)
    resource_profile: str | None = None


class CudaMemorySummary(ArtifactModel):
    peak_allocated_bytes: StrictInt = Field(ge=0)
    peak_reserved_bytes: StrictInt = Field(ge=0)


class MapInputAudit(ArtifactModel):
    speed_limit_sentinel_replaced_count: StrictInt = Field(ge=0)
    speed_limit_existing_preserved_count: StrictInt = Field(ge=0)
    configured_programmatic_lane_speed_limit_kmh: StrictFloat = Field(gt=0.0)
    block_speed_limit_profile_kmh: tuple[StrictFloat, ...] | None = None
    block_speed_limit_profile_applied_lane_count: StrictInt = Field(default=0, ge=0)
    lane_speed_limit_kmh_counts: dict[str, StrictInt]
    valid_lane_count_min: StrictInt = Field(ge=0)
    valid_lane_count_max: StrictInt = Field(ge=0)
    speed_limit_valid_count_min: StrictInt = Field(ge=0)
    speed_limit_valid_count_max: StrictInt = Field(ge=0)
    speed_limit_mps_min: StrictFloat | None = Field(default=None, ge=0.0)
    speed_limit_mps_max: StrictFloat | None = Field(default=None, ge=0.0)
    speed_limit_mps_unique_values: tuple[StrictFloat, ...]


class SpeedSummary(ArtifactModel):
    minimum: StrictFloat
    mean: StrictFloat
    maximum: StrictFloat


class EnergySummary(ArtifactModel):
    metric: Literal["metadrive_fuel_proxy"]
    total_ml: StrictFloat = Field(ge=0.0)
    distance_m: StrictFloat = Field(ge=0.0)
    ml_per_km: StrictFloat | None = Field(default=None, ge=0.0)


class ErrorValues(ArtifactModel):
    maximum: StrictFloat
    mean: StrictFloat
    final: StrictFloat


class ExecutionErrorSummary(ArtifactModel):
    position_m: ErrorValues
    heading_rad: ErrorValues


class WarmupSummary(ArtifactModel):
    simulator_steps: StrictInt = Field(ge=0)
    simulated_seconds: StrictFloat = Field(ge=0.0)
    ego_displacement_m_maximum: StrictFloat = Field(ge=0.0)
    participant_count_minimum: StrictInt = Field(ge=0)
    participant_count_maximum: StrictInt = Field(ge=0)


class TrafficObservationSummary(ArtifactModel):
    planning_frames: StrictInt = Field(ge=0)
    frames_with_participants: StrictInt = Field(ge=0)
    frames_with_participants_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    participant_count_minimum: StrictInt = Field(ge=0)
    participant_count_maximum: StrictInt = Field(ge=0)
    nearest_participant_distance_m_minimum: StrictFloat | None = Field(default=None, ge=0.0)


class EpisodeMetrics(ArtifactModel):
    """One completed closed-loop episode's common trace-derived metrics."""

    aggregation_unit: Literal["evaluation_episode"] = "evaluation_episode"
    simulated_seconds: StrictFloat = Field(gt=0.0)
    distance_m: StrictFloat = Field(ge=0.0)
    speed_mps: SpeedSummary
    stopped_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    route_completion: StrictFloat
    energy: EnergySummary
    arrive_dest: StrictBool
    collision: StrictBool
    out_of_road: StrictBool


class CompletedEpisodeSummary(ArtifactModel):
    status: Literal["completed"] = "completed"
    trace_status: Literal["complete"] = "complete"
    scenario: ScenarioSummary
    evaluation_mode: Literal["no_traffic", "traffic"]
    traffic_density: StrictFloat = Field(ge=0.0, le=1.0)
    route_length_m: StrictFloat = Field(gt=0.0)
    noise_seed: StrictInt = Field(ge=0)
    sampler: SamplerSummary
    guidance: GuidanceSummary
    plan_cycles: StrictInt = Field(gt=0)
    simulator_steps: StrictInt = Field(gt=0)
    environment_steps_including_warmup: StrictInt = Field(gt=0)
    metrics: EpisodeMetrics
    crash_vehicle: StrictBool
    crash_object: StrictBool
    crash_building: StrictBool
    crash_human: StrictBool
    crash_sidewalk: StrictBool
    terminated: StrictBool
    truncated: StrictBool
    terminal_reason: str = Field(min_length=1)
    termination: TerminationSummary
    map_input_audit: MapInputAudit
    history_warmup: WarmupSummary
    traffic_observation: TrafficObservationSummary
    trajectory_execution_error: ExecutionErrorSummary


class FailedEpisodeSummary(ArtifactModel):
    status: Literal["failed"] = "failed"
    scenario: ScenarioSummary
    evaluation_mode: Literal["no_traffic", "traffic"]
    traffic_density: StrictFloat = Field(ge=0.0, le=1.0)
    noise_seed: StrictInt = Field(ge=0)
    sampler: SamplerSummary
    guidance: GuidanceSummary
    trace_status: Literal["partial", "empty"]
    energy: EnergySummary | None = None
    termination: TerminationSummary
    failure: FailureSummary


EpisodeSummary = Annotated[
    CompletedEpisodeSummary | FailedEpisodeSummary,
    Field(discriminator="status"),
]


class JobSummary(ArtifactModel):
    status: Literal["completed", "failed"]
    runtime: InferenceRuntimeSummary
    checkpoint: CheckpointSummary
    sampler: SamplerSummary
    guidance: GuidanceSummary
    workload: EvaluationWorkload
    episodes: tuple[EpisodeSummary, ...]

    @model_validator(mode="after")
    def validate_status(self) -> JobSummary:
        expected = (
            "failed" if any(item.status == "failed" for item in self.episodes) else "completed"
        )
        if self.status != expected:
            raise ValueError("job status must agree with episode statuses")
        if not self.episodes:
            raise ValueError("job summary must contain at least one episode")
        workload_scenarios = tuple(
            (item.name, item.map, item.seed) for item in self.workload.scenarios
        )
        result_scenarios = tuple(
            (item.scenario.name, item.scenario.map_sequence, item.scenario.seed)
            for item in self.episodes
        )
        if result_scenarios != workload_scenarios:
            raise ValueError("job summary episodes must match the declared workload scenarios")
        if any(item.evaluation_mode != self.workload.mode for item in self.episodes):
            raise ValueError("job summary episode modes must match the declared workload")
        return self


class RuntimeMetadata(ArtifactModel):
    git_head: str = Field(min_length=1)
    git_status_short: tuple[str, ...]
    platform: str = Field(min_length=1)
    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    lightning: str = Field(min_length=1)
    metadrive: str = Field(min_length=1)
    pydantic: str = Field(min_length=1)
    inference_runtime: InferenceRuntimeSummary
    sampler: SamplerSummary
    guidance: GuidanceSummary
    execution: ExecutionSummary
    elapsed_seconds: StrictFloat = Field(ge=0.0)
    cuda_memory: CudaMemorySummary | None
