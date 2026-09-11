"""Persisted training summary models without execution dependencies."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class RewardComponentMeans(_ArtifactModel):
    ttc: StrictFloat = Field(ge=0.0, le=1.0)
    progress: StrictFloat = Field(ge=0.0, le=1.0)
    comfort: StrictFloat = Field(ge=0.0, le=1.0)
    speed: StrictFloat = Field(ge=0.0, le=1.0)
    energy: StrictFloat = Field(ge=0.0, le=1.0)


class RewardDiagnosticMeans(_ArtifactModel):
    collision_score: StrictFloat = Field(ge=0.0, le=1.0)
    drivable_score: StrictFloat = Field(ge=0.0, le=1.0)
    wrong_direction_score: StrictFloat = Field(ge=0.0, le=1.0)


class PPOGradientDiagnosticsSummary(_ArtifactModel):
    actor_head_policy: StrictFloat = Field(ge=0.0)
    shared_trunk_policy: StrictFloat = Field(ge=0.0)
    value_head_critic: StrictFloat = Field(ge=0.0)
    shared_trunk_critic: StrictFloat = Field(ge=0.0)
    actor_head_entropy: StrictFloat = Field(ge=0.0)
    shared_trunk_entropy: StrictFloat = Field(ge=0.0)


class TrainingUpdateSummary(_ArtifactModel):
    update_index: StrictInt = Field(ge=0)
    sample_count: StrictInt = Field(gt=0)
    episode_count: StrictInt = Field(gt=0)
    mean_episode_length: StrictFloat = Field(gt=0.0)
    total_reward: StrictFloat
    base_reward: StrictFloat
    mean_safety_gate: StrictFloat = Field(ge=0.0, le=1.0)
    route_completion_delta: StrictFloat
    distance_m: StrictFloat = Field(ge=0.0)
    mean_speed_mps: StrictFloat = Field(ge=0.0)
    stopped_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    collision_count: StrictInt = Field(ge=0)
    out_of_road_count: StrictInt = Field(ge=0)
    maximum_position_error_m: StrictFloat = Field(ge=0.0)
    maximum_heading_error_rad: StrictFloat = Field(ge=0.0)
    beta_alpha_mean: tuple[StrictFloat, StrictFloat]
    beta_alpha_min: tuple[StrictFloat, StrictFloat]
    beta_alpha_max: tuple[StrictFloat, StrictFloat]
    beta_beta_mean: tuple[StrictFloat, StrictFloat]
    beta_beta_min: tuple[StrictFloat, StrictFloat]
    beta_beta_max: tuple[StrictFloat, StrictFloat]
    action_mean: tuple[StrictFloat, StrictFloat]
    action_std: tuple[StrictFloat, StrictFloat]
    action_min: tuple[StrictFloat, StrictFloat]
    action_max: tuple[StrictFloat, StrictFloat]
    mean_state_value: StrictFloat
    std_state_value: StrictFloat
    mean_policy_loss: StrictFloat
    mean_value_loss: StrictFloat
    mean_entropy_loss: StrictFloat
    mean_total_loss: StrictFloat
    mean_approximate_kl: StrictFloat
    mean_clip_fraction: StrictFloat
    mean_entropy: StrictFloat
    mean_explained_variance: StrictFloat
    maximum_pre_clip_gradient_norm: StrictFloat
    evaluated_minibatch_count: StrictInt = Field(gt=0)
    optimizer_step_count: StrictInt = Field(ge=0)
    final_learning_rate: StrictFloat = Field(ge=0.0)
    raw_advantage_mean: StrictFloat
    raw_advantage_std: StrictFloat = Field(ge=0.0)
    normalized_advantage_mean: StrictFloat
    normalized_advantage_std: StrictFloat = Field(ge=0.0)
    mean_value_target: StrictFloat
    std_value_target: StrictFloat
    kl_early_stopped: StrictBool
    kl_early_stop_trigger: StrictFloat | None
    cumulative_kl_early_stop_count: StrictInt = Field(ge=0)
    policy_ratio_mean: StrictFloat = Field(gt=0.0)
    policy_ratio_std: StrictFloat = Field(ge=0.0)
    policy_ratio_p95: StrictFloat = Field(gt=0.0)
    policy_ratio_max: StrictFloat = Field(gt=0.0)
    gradient_diagnostics: PPOGradientDiagnosticsSummary | None
    reward_profile: Literal[
        "plannerrft_energy_v1",
        "plannerrft_energy_band_lam64_v1",
        "plannerrft_no_energy_v1",
        "plannerrft_no_energy_calibrated_v1",
    ]
    native_step_energy_total_ml: StrictFloat = Field(ge=0.0)
    executed_fuel_proxy_total_ml: StrictFloat = Field(ge=0.0)
    executed_fuel_proxy_distance_m: StrictFloat = Field(ge=0.0)
    executed_fuel_proxy_ml_per_km: StrictFloat | None = Field(ge=0.0)
    reward_component_means: RewardComponentMeans
    reward_diagnostic_means: RewardDiagnosticMeans


class PolicyProbeSummary(_ArtifactModel):
    alpha: tuple[tuple[StrictFloat, StrictFloat], ...]
    beta: tuple[tuple[StrictFloat, StrictFloat], ...]
    guidance_mean: tuple[tuple[StrictFloat, StrictFloat], ...]
    boundary_mass: tuple[tuple[StrictFloat, StrictFloat], ...]


class TrainingRunSummary(_ArtifactModel):
    status: Literal["completed"]
    training_seed: StrictInt = Field(ge=0)
    replay_id: StrictInt = Field(ge=0)
    noise_seeds: tuple[StrictInt, ...]
    policy_action_seeds: tuple[StrictInt, ...]
    total_transitions: StrictInt = Field(gt=0)
    initial_policy_hash: str = Field(min_length=64, max_length=64)
    final_policy_hash: str = Field(min_length=64, max_length=64)
    frozen_planner_hash_before: str = Field(min_length=64, max_length=64)
    frozen_planner_hash_after: str = Field(min_length=64, max_length=64)
    probe_before: PolicyProbeSummary
    probe_after: PolicyProbeSummary
    updates: tuple[TrainingUpdateSummary, ...]
    reward_profile: Literal[
        "plannerrft_energy_v1",
        "plannerrft_energy_band_lam64_v1",
        "plannerrft_no_energy_v1",
        "plannerrft_no_energy_calibrated_v1",
    ]
