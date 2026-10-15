"""Synthetic diagnostic inputs, persisted batches, measurements, and reports."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from eco_planner.experiments.training.config import GateConfig, TrainingGridConfig
from eco_planner.rl.artifacts.summaries import (
    RewardComponentMeans,
    RewardDiagnosticMeans,
    TrainingUpdateSummary,
)

if TYPE_CHECKING:
    from eco_planner.rl.optimization.ppo import PPOUpdateReport


def diagnostic_inputs(reward, *, minibatch_size=2):
    import torch

    from eco_planner.jobs import compose_job_config
    from eco_planner.planning.policy import ExplorationPolicy
    from eco_planner.rl.config import parse_training_config
    from tests.training.helpers import _policy_config, _ppo_config

    raw = compose_job_config(
        "jobs/training/ppo",
        [
            "components/resources=rtx3050_laptop",
            "runtime.seed=0",
            "runtime.accelerator=cpu",
            "training.replay_id=0",
            "training.transitions_per_environment=2",
            "training.update_count=1",
        ],
    )
    raw.policy = _policy_config().model_dump()
    raw.ppo = (
        _ppo_config()
        .model_copy(
            update={
                "batch_size": 4,
                "minibatch_size": minibatch_size,
                "scheduler_total_optimizer_steps": 2,
            }
        )
        .model_dump()
    )
    raw.reward = reward.model_dump()
    config = parse_training_config(raw)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(config.policy)
    return config, policy


def persist_diagnostic_batch(path, config, policy, slots):
    from omegaconf import OmegaConf

    from eco_planner.artifacts import write_json
    from eco_planner.planning.policy import save_exploration_policy_checkpoint
    from eco_planner.rl.rollout.fixed_batch import write_batch

    path.mkdir(parents=True)
    episodes, samples = write_batch(path, slots, config.scenarios)
    OmegaConf.save(OmegaConf.create(config.model_dump(mode="json")), path / "resolved_config.yaml")
    save_exploration_policy_checkpoint(path / "policy-initial.pt", policy)
    write_json(path / "runtime_metadata.json", {"fixture": "synthetic CPU"})
    return episodes, samples


def build_ppo_update_report(sample_count: int) -> PPOUpdateReport:
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


def _gate(**overrides: object) -> GateConfig:
    values: dict[str, object] = {
        "target_kl": 0.006,
        "kl_floor": 1.0e-06,
        "within_target_kl_fraction": 0.9,
        "runaway_tail_updates": 10,
        "runaway_tail_factor": 10.0,
        "ratio_change_floor": 1.0e-04,
        "guidance_rms_shift_floor": 0.01,
        "beta_parameter_floor": 0.1,
        "boundary_mass_ceiling": 0.2,
        "episode_length_retention_floor": 0.5,
        "collision_budget": 2,
        "heldout_relative_change_floor": 0.001,
        "heldout_metrics": [
            "mean_speed_mps",
            "distance_m",
            "route_completion",
            "energy_total_ml",
            "energy_ml_per_km",
        ],
    }
    values.update(overrides)
    return GateConfig.model_validate(values)


def _study(**overrides: object) -> TrainingGridConfig:
    values: dict[str, object] = {
        "version": 1,
        "mc_draws": 4096,
        "mc_seed": 1000003,
        "study_name": "unit_effective_update",
        "protocol": "experiments/comparison/default.yaml",
        "arm": "a1",
        "training_seed": 0,
        "update_count": 50,
        "base_overrides": ["components/resources=rtx_a4000"],
        "grid": {
            "learning_rates": [1.0e-05, 1.0e-04],
            "epochs": [1, 2],
            "max_gradient_norms": [0.5, 2.0],
        },
        "gate": _gate().model_dump(),
        "selection": "lowest_learning_rate_then_epochs_then_max_gradient_norm",
    }
    values.update(overrides)
    return TrainingGridConfig.model_validate(values)


def _metrics(**overrides: object) -> dict[str, Any]:
    values: dict[str, Any] = {
        "post_update_kl": [1.0e-05] * 50,
        "post_update_kl_median": 1.0e-05,
        "policy_ratio_change": 1.0e-03,
        "probe_guidance_rms_shift": 0.02,
        "min_beta_alpha": 1.5,
        "min_beta_beta": 1.5,
        "probe_boundary_mass_max_after": 0.02,
        "behavior": {
            "collision_count": 0,
            "out_of_road_count": 0,
            "episode_length_first_median": 8.0,
            "episode_length_last_median": 8.0,
        },
    }
    values.update(overrides)
    return values
