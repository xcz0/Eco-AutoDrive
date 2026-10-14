"""Explicit synthetic diagnostics for aggregation, tracking, and report tests."""

from dataclasses import asdict
from typing import TYPE_CHECKING

from eco_planner.rl.artifacts.summaries import (
    RewardComponentMeans,
    RewardDiagnosticMeans,
    TrainingUpdateSummary,
)

if TYPE_CHECKING:
    from eco_planner.rl.optimization.ppo import PPOUpdateReport


def build_ppo_update_report(sample_count: int) -> "PPOUpdateReport":
    """Supply optimizer diagnostics without running a policy or an update."""
    from eco_planner.rl.optimization.ppo import PPOUpdateReport

    return PPOUpdateReport(
        sample_count=sample_count,
        evaluated_minibatch_count=1,
        optimizer_step_count=1,
        mean_policy_loss=-0.1,
        mean_value_loss=0.4,
        mean_entropy_loss=-0.01,
        mean_total_loss=0.29,
        mean_approximate_kl=0.002,
        mean_clip_fraction=0.1,
        mean_entropy=1.0,
        mean_explained_variance=0.2,
        maximum_pre_clip_gradient_norm=0.8,
        final_learning_rate=0.00025,
        raw_advantage_mean=0.125,
        raw_advantage_std=1.237436867,
        normalized_advantage_mean=0.0,
        normalized_advantage_std=1.0,
        mean_value_target=1.125,
        std_value_target=0.875,
        kl_early_stopped=False,
        kl_early_stop_trigger=None,
        cumulative_kl_early_stop_count=0,
        policy_ratio_mean=1.0,
        policy_ratio_std=0.1,
        policy_ratio_p95=1.1,
        policy_ratio_max=1.2,
        gradient_diagnostics=None,
    )


def build_training_update_summary() -> TrainingUpdateSummary:
    """Two one-transition episodes with rewards 0.25 and 2.0; no optimizer execution."""
    return TrainingUpdateSummary(
        **asdict(build_ppo_update_report(sample_count=2)),
        update_index=0,
        episode_count=2,
        mean_episode_length=1.0,
        total_reward=2.25,
        base_reward=2.25,
        mean_safety_gate=1.0,
        route_completion_delta=0.2,
        distance_m=2.0,
        mean_speed_mps=2.0,
        stopped_fraction=0.0,
        collision_count=0,
        out_of_road_count=0,
        maximum_position_error_m=0.0,
        maximum_heading_error_rad=0.0,
        beta_alpha_mean=(2.0, 2.0),
        beta_alpha_min=(2.0, 2.0),
        beta_alpha_max=(2.0, 2.0),
        beta_beta_mean=(2.0, 2.0),
        beta_beta_min=(2.0, 2.0),
        beta_beta_max=(2.0, 2.0),
        action_mean=(-0.5, 0.5),
        action_std=(0.0, 0.0),
        action_min=(-0.5, 0.5),
        action_max=(-0.5, 0.5),
        mean_state_value=1.0,
        std_state_value=0.0,
        reward_profile="plannerrft_energy_v1",
        native_step_energy_total_ml=0.0,
        executed_fuel_proxy_total_ml=0.1,
        executed_fuel_proxy_distance_m=2.0,
        executed_fuel_proxy_ml_per_km=50.0,
        reward_component_means=RewardComponentMeans(
            ttc=1.0, progress=0.5, comfort=1.0, speed=1.0, energy=0.5
        ),
        reward_diagnostic_means=RewardDiagnosticMeans(
            collision_score=1.0, drivable_score=1.0, wrong_direction_score=1.0
        ),
    )
