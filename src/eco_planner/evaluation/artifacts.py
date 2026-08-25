"""Lightweight evaluation artifact models, schemas, readers, and writers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    TypeAdapter,
    model_validator,
)

from eco_planner.execution_contracts import EVALUATION_EXECUTION_STEPS, PLANNER_FUTURE_STEPS

if TYPE_CHECKING:
    from eco_planner.evaluation.config import VideoConfig


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
    gradient_step_coefficient: Literal[1.0]
    reference_refresh_cycles: Literal[1]
    share_scene_encoding: Literal[True]
    share_initial_noise: Literal[True]
    share_transition_noise: Literal[True]
    heading_norm_epsilon: StrictFloat = Field(gt=0.0)
    zero_speed_tolerance_mps: StrictFloat = Field(gt=0.0)


GuidanceSummary = Annotated[
    NoGuidanceSummary | OrthogonalGuidanceSummary,
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
    simulated_seconds: StrictFloat = Field(gt=0.0)
    environment_steps_including_warmup: StrictInt = Field(gt=0)
    total_reward: StrictFloat
    distance_m: StrictFloat = Field(ge=0.0)
    energy: EnergySummary
    speed_mps: SpeedSummary
    route_completion: StrictFloat
    arrive_dest: StrictBool
    out_of_road: StrictBool
    crash_vehicle: StrictBool
    crash_object: StrictBool
    crash_building: StrictBool
    crash_human: StrictBool
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


PLANNER_ACTOR_COUNT = 11
PLANNER_STATE_DIM = 4
EXECUTION_PREFIX_STEPS = EVALUATION_EXECUTION_STEPS

_PLAN = "plan"
_SIMULATOR = "simulator"
_WARMUP = "warmup"


@dataclass(frozen=True)
class TraceFieldSpec:
    """Shape and dtype for one persisted trace array."""

    axes: tuple[str | int, ...]
    dtype: np.dtype | None
    guided_only: bool = False
    finite: bool = True


OBSERVATION_FIELDS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "ego_current_state": ((10,), np.dtype(np.float32)),
    "neighbor_agents_past": ((32, 21, 11), np.dtype(np.float32)),
    "static_objects": ((5, 10), np.dtype(np.float32)),
    "lanes": ((70, 20, 12), np.dtype(np.float32)),
    "lanes_speed_limit": ((70, 1), np.dtype(np.float32)),
    "lanes_has_speed_limit": ((70, 1), np.dtype(np.bool_)),
    "route_lanes": ((25, 20, 12), np.dtype(np.float32)),
    "route_lanes_speed_limit": ((25, 1), np.dtype(np.float32)),
    "route_lanes_has_speed_limit": ((25, 1), np.dtype(np.bool_)),
}

_BASE_TRACE_FIELDS: dict[str, TraceFieldSpec] = {
    "trace_status": TraceFieldSpec((), None, finite=False),
    "warmup_initial_state": TraceFieldSpec((7,), np.dtype(np.float64)),
    "warmup_initial_state_valid": TraceFieldSpec((), np.dtype(np.bool_), finite=False),
    "initial_state": TraceFieldSpec((7,), np.dtype(np.float64)),
    "initial_state_valid": TraceFieldSpec((), np.dtype(np.bool_), finite=False),
    "warmup_states": TraceFieldSpec((_WARMUP, 7), np.dtype(np.float64)),
    "warmup_rewards": TraceFieldSpec((_WARMUP,), np.dtype(np.float64)),
    "warmup_step_energy_ml": TraceFieldSpec((_WARMUP,), np.dtype(np.float64)),
    "warmup_episode_energy_ml": TraceFieldSpec((_WARMUP,), np.dtype(np.float64)),
    "warmup_terminated": TraceFieldSpec((_WARMUP,), np.dtype(np.bool_), finite=False),
    "warmup_truncated": TraceFieldSpec((_WARMUP,), np.dtype(np.bool_), finite=False),
    "warmup_participant_counts": TraceFieldSpec((_WARMUP,), np.dtype(np.int64), finite=False),
    "warmup_static_object_counts": TraceFieldSpec((_WARMUP,), np.dtype(np.int64), finite=False),
    "planning_anchors": TraceFieldSpec((_PLAN, 7), np.dtype(np.float64)),
    "initial_noise": TraceFieldSpec(
        (_PLAN, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM), np.dtype(np.float32)
    ),
    "predictions_local": TraceFieldSpec(
        (_PLAN, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM), np.dtype(np.float32)
    ),
    "ego_predictions_world": TraceFieldSpec(
        (_PLAN, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM), np.dtype(np.float64)
    ),
    "executed_states": TraceFieldSpec((_SIMULATOR, 7), np.dtype(np.float64)),
    "executed_rewards": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "executed_step_energy_ml": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "executed_episode_energy_ml": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "executed_terminated": TraceFieldSpec((_SIMULATOR,), np.dtype(np.bool_), finite=False),
    "executed_truncated": TraceFieldSpec((_SIMULATOR,), np.dtype(np.bool_), finite=False),
    "executed_plan_indices": TraceFieldSpec((_SIMULATOR,), np.dtype(np.int64), finite=False),
    "trajectory_target_centers": TraceFieldSpec((_SIMULATOR, 2), np.dtype(np.float64)),
    "trajectory_target_headings": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "trajectory_position_errors_m": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "trajectory_heading_errors_rad": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "traffic_selected_ids": TraceFieldSpec((_PLAN, 32), np.dtype("<U64"), finite=False),
    "traffic_participant_counts": TraceFieldSpec((_PLAN,), np.dtype(np.int64), finite=False),
    "traffic_static_object_counts": TraceFieldSpec((_PLAN,), np.dtype(np.int64), finite=False),
    "traffic_nearest_distance_m": TraceFieldSpec((_PLAN,), np.dtype(np.float64)),
    "traffic_has_nearest": TraceFieldSpec((_PLAN,), np.dtype(np.bool_), finite=False),
}

_GUIDANCE_TRACE_FIELDS: dict[str, TraceFieldSpec] = {
    "reference_predictions_local": TraceFieldSpec(
        (_PLAN, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
        np.dtype(np.float32),
        guided_only=True,
    ),
    "guidance_actions": TraceFieldSpec((_PLAN, 2), np.dtype(np.float32), guided_only=True),
    "guidance_lateral_target_offset_m": TraceFieldSpec(
        (_PLAN,), np.dtype(np.float32), guided_only=True
    ),
    "guidance_longitudinal_target_speed_fraction": TraceFieldSpec(
        (_PLAN,), np.dtype(np.float32), guided_only=True
    ),
    "guidance_longitudinal_target_speed_delta_mps": TraceFieldSpec(
        (_PLAN, PLANNER_FUTURE_STEPS), np.dtype(np.float32), guided_only=True
    ),
    "guidance_lateral_objective_delta": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_longitudinal_objective_delta": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_applied_gradient_l2": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_applied_gradient_max_abs": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_raw_neighbor_gradient_l2": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_zero_speed_count": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.int64), guided_only=True, finite=False
    ),
}

TRACE_FIELDS = {
    **_BASE_TRACE_FIELDS,
    **{
        f"observation_{name}": TraceFieldSpec((_PLAN, *shape), dtype, finite=dtype.kind == "f")
        for name, (shape, dtype) in OBSERVATION_FIELDS.items()
    },
    **_GUIDANCE_TRACE_FIELDS,
}
GUIDED_TRACE_FIELDS = frozenset(name for name, spec in TRACE_FIELDS.items() if spec.guided_only)
STATIC_TRACE_FIELDS = frozenset(
    {
        "trace_status",
        "warmup_initial_state",
        "warmup_initial_state_valid",
        "initial_state",
        "initial_state_valid",
    }
)


def trace_shape(spec: TraceFieldSpec, *, plan: int, simulator: int, warmup: int) -> tuple[int, ...]:
    """Resolve declarative axes to a concrete persisted array shape."""

    capacities = {_PLAN: plan, _SIMULATOR: simulator, _WARMUP: warmup}
    return tuple(capacities.get(axis, axis) for axis in spec.axes)


def allocate_trace_arrays(
    max_plan_cycles: int, max_warmup_steps: int, guided: bool
) -> dict[str, np.ndarray]:
    """Allocate all recorder-owned arrays directly from ``TRACE_FIELDS``."""

    capacities = {
        "plan": max_plan_cycles,
        "simulator": max_plan_cycles * EXECUTION_PREFIX_STEPS,
        "warmup": max_warmup_steps,
    }
    return {
        name: np.empty(trace_shape(spec, **capacities), dtype=spec.dtype)
        for name, spec in TRACE_FIELDS.items()
        if name not in STATIC_TRACE_FIELDS and (guided or not spec.guided_only)
    }


def validate_trace_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_plan_cycles: int | None = None,
    expected_simulator_steps: int | None = None,
    expected_warmup_steps: int | None = None,
    require_traffic: bool = False,
    expected_trace_status: str | None = None,
    require_finite: bool = True,
) -> None:
    """Validate field declarations and cross-array trace invariants."""

    mapping = arrays
    present_guidance = GUIDED_TRACE_FIELDS & set(mapping)
    if present_guidance and present_guidance != GUIDED_TRACE_FIELDS:
        missing = sorted(GUIDED_TRACE_FIELDS - present_guidance)
        raise ValueError(f"guided trace is missing arrays: {missing}")
    expected_fields = {
        name: spec
        for name, spec in TRACE_FIELDS.items()
        if present_guidance or not spec.guided_only
    }
    missing = sorted(set(expected_fields) - set(mapping))
    if missing:
        raise ValueError(f"trace is missing arrays: {missing}")
    unexpected = sorted(set(mapping) - set(expected_fields))
    if unexpected:
        raise ValueError(f"trace contains unexpected arrays: {unexpected}")
    dynamic_shape = {"plan": None, "simulator": None, "warmup": None}
    for name, spec in expected_fields.items():
        value = mapping[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"trace array {name!r} must be a numpy.ndarray")
        expected_shape = tuple(dynamic_shape.get(axis, axis) for axis in spec.axes)
        if len(value.shape) != len(expected_shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, expected_shape, strict=True)
        ):
            raise ValueError(
                f"trace array {name!r} has shape {value.shape}, expected {expected_shape}"
            )
        if spec.dtype is not None and value.dtype != spec.dtype:
            raise TypeError(f"trace array {name!r} has dtype {value.dtype}, expected {spec.dtype}")
        if (
            require_finite
            and spec.finite
            and value.dtype.kind in "fc"
            and not np.isfinite(value).all()
        ):
            raise ValueError(f"trace array {name!r} contains non-finite values")

    plan_cycles = mapping["initial_noise"].shape[0]
    simulator_steps = mapping["executed_states"].shape[0]
    warmup_steps = mapping["warmup_states"].shape[0]
    trace_status = str(mapping["trace_status"].item())
    if trace_status not in {"complete", "partial", "empty"}:
        raise ValueError("trace status is invalid")
    if expected_trace_status is not None and trace_status != expected_trace_status:
        raise ValueError("trace status disagrees with summary")
    if trace_status == "complete" and (plan_cycles <= 0 or simulator_steps <= 0):
        raise ValueError("complete trace must contain planning and simulator steps")
    if trace_status == "empty" and (plan_cycles or simulator_steps or warmup_steps):
        raise ValueError("empty trace contains recorded steps")
    if trace_status == "complete" and not bool(mapping["initial_state_valid"].item()):
        raise ValueError("complete trace requires a valid initial state")
    if expected_plan_cycles is not None and plan_cycles != expected_plan_cycles:
        raise ValueError("trace planning cycle count disagrees with summary")
    if expected_simulator_steps is not None and simulator_steps != expected_simulator_steps:
        raise ValueError("trace simulator step count disagrees with summary")
    if expected_warmup_steps is not None and warmup_steps != expected_warmup_steps:
        raise ValueError(f"trace must contain exactly {expected_warmup_steps} warmup states")
    axis_sizes = {_PLAN: plan_cycles, _SIMULATOR: simulator_steps, _WARMUP: warmup_steps}
    for name, spec in expected_fields.items():
        if spec.axes and isinstance(spec.axes[0], str):
            if mapping[name].shape[0] != axis_sizes[spec.axes[0]]:
                raise ValueError(f"trace array {name!r} is not {spec.axes[0]}-aligned")
    for name in (
        "warmup_participant_counts",
        "warmup_static_object_counts",
        "warmup_step_energy_ml",
        "warmup_episode_energy_ml",
        "executed_plan_indices",
        "traffic_participant_counts",
        "traffic_static_object_counts",
        "guidance_zero_speed_count",
        "executed_step_energy_ml",
        "executed_episode_energy_ml",
    ):
        if name in mapping and np.any(mapping[name] < 0):
            raise ValueError(f"trace array {name!r} must be non-negative")
    plan_indices = mapping["executed_plan_indices"]
    if plan_cycles:
        if not np.array_equal(np.unique(plan_indices), np.arange(plan_cycles)):
            raise ValueError("trace plan indices are not contiguous")
        counts = np.bincount(plan_indices, minlength=plan_cycles)
        if np.any(counts[:-1] != EXECUTION_PREFIX_STEPS) or not (
            1 <= counts[-1] <= EXECUTION_PREFIX_STEPS
        ):
            raise ValueError(
                f"trace plan indices do not encode {EXECUTION_PREFIX_STEPS}-step prefixes"
            )
        if not np.array_equal(plan_indices, np.repeat(np.arange(plan_cycles), counts)):
            raise ValueError("trace plan indices are not ordered by planning cycle")
    elif simulator_steps:
        raise ValueError("trace has simulator steps without planning cycles")
    terminal = mapping["executed_terminated"] | mapping["executed_truncated"]
    if terminal[:-1].any():
        raise ValueError("trace contains a terminal flag before its final simulator step")
    episode_energy = np.concatenate(
        (mapping["warmup_episode_energy_ml"], mapping["executed_episode_energy_ml"])
    )
    if np.any(np.diff(episode_energy) < 0.0):
        raise ValueError("trace episode energy must be cumulative")
    if require_traffic and not np.any(mapping["traffic_participant_counts"] > 0):
        raise ValueError("trace never observed traffic within the query radius")
    nearest = mapping["traffic_nearest_distance_m"][mapping["traffic_has_nearest"]]
    if np.any(nearest < 0.0):
        raise ValueError("trace nearest traffic distances must be non-negative")


_EPISODE_ADAPTER = TypeAdapter(EpisodeSummary)
_Artifact = TypeVar("_Artifact", JobSummary, RuntimeMetadata)


@dataclass(frozen=True)
class LoadedTraceArtifact:
    """Validated current-schema trace arrays."""

    trace_status: str
    arrays: dict[str, np.ndarray]


def load_job_summary(path: Path) -> JobSummary:
    """Load a typed current-schema job summary without compatibility conversion."""

    return _load_json(path, JobSummary)


def load_episode_summary(path: Path) -> EpisodeSummary:
    """Load a typed current-schema episode summary without compatibility conversion."""

    return _load_episode_json(path)


def load_runtime_metadata(path: Path) -> RuntimeMetadata:
    """Load typed current-schema runtime metadata without compatibility conversion."""

    return _load_json(path, RuntimeMetadata)


def load_trace_artifact(path: Path) -> LoadedTraceArtifact:
    """Load current-schema NPZ arrays without synthesizing missing fields."""

    with np.load(path, allow_pickle=False) as trace:
        arrays = {name: trace[name] for name in trace.files}
    status = str(arrays["trace_status"].item())
    validate_trace_arrays(arrays, expected_trace_status=status)
    return LoadedTraceArtifact(status, arrays)


def _load_json(path: Path, model: type[_Artifact]) -> _Artifact:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _load_episode_json(path: Path) -> EpisodeSummary:
    return _EPISODE_ADAPTER.validate_json(path.read_text(encoding="utf-8"))


def write_episode_artifacts(
    output_dir: Path,
    trace_arrays: dict[str, np.ndarray],
    frames: list[np.ndarray],
    summary: CompletedEpisodeSummary | FailedEpisodeSummary,
    video_config: VideoConfig,
) -> None:
    """Persist one finalized episode without recomputing trace arrays."""

    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(output_dir / "trace.npz", **trace_arrays)
    write_json(output_dir / "summary.json", summary)
    if video_config.enabled:
        if summary.status == "completed" and not frames:
            raise RuntimeError("video output was enabled but no frames were rendered")
        if frames:
            from eco_planner.evaluation.rendering import write_gif

            write_gif(frames, output_dir / "closed_loop.gif", video_config.fps)


def write_json(path: Path, payload: Any) -> None:
    """Persist a Pydantic JSON artifact with stable formatting."""

    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
