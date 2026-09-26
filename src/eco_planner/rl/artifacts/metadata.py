"""Strict training runtime artifacts; independent of evaluation schema and execution imports."""

from typing import Literal

from pydantic import Field, StrictFloat, StrictInt

from eco_planner.runtime.resources import ResourceProfileConfig

from .summaries import _ArtifactModel


class InferenceRuntimeSummary(_ArtifactModel):
    requested_accelerator: Literal["auto", "cpu", "cuda"]
    resolved_accelerator: Literal["cpu", "cuda"]
    requested_precision: Literal["auto", "32-true", "16-mixed", "bf16-mixed"]
    resolved_precision: Literal["32-true", "16-mixed", "bf16-mixed"]
    device: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    world_size: Literal[1]


class CheckpointSummary(_ArtifactModel):
    ema_tensor_count: StrictInt = Field(gt=0)
    parameter_count: StrictInt = Field(gt=0)


class SamplerSummary(_ArtifactModel):
    name: Literal["dpm10", "ddim5"]
    implementation: Literal["diffusers"]
    num_steps: StrictInt = Field(gt=0)
    timesteps: tuple[StrictFloat, ...] | None
    initial_noise_scale: StrictFloat = Field(gt=0.0)
    ddim_stochasticity: StrictFloat = Field(ge=0.0, le=1.0)
    parity_label: str = Field(min_length=1)


class OrthogonalPolicyGuidanceSummary(_ArtifactModel):
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


class TrackingIdentity(_ArtifactModel):
    run_id: str = Field(min_length=1)
    tracking_uri: str = Field(min_length=1)


class TrainingRuntimeMetadata(_ArtifactModel):
    git_head: str = Field(min_length=1)
    git_branch: str = Field(min_length=1)
    git_status_short: tuple[str, ...]
    platform: str = Field(min_length=1)
    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    lightning: str = Field(min_length=1)
    metadrive: str = Field(min_length=1)
    pydantic: str = Field(min_length=1)
    inference_runtime: InferenceRuntimeSummary
    checkpoint: CheckpointSummary
    sampler: SamplerSummary
    guidance: OrthogonalPolicyGuidanceSummary
    resources: ResourceProfileConfig
    tracking: TrackingIdentity | None = None
