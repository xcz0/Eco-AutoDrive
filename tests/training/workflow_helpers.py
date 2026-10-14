"""Synthetic effective-update measurements and grid configurations."""

from __future__ import annotations

from typing import Any

from eco_planner.experiments.training.config import GateConfig, TrainingGridConfig


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
