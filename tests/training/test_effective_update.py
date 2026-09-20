from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from eco_planner.analysis.training import (
    beta_probe_statistics,
    heldout_metric_values,
    paired_beta_deltas,
)
from eco_planner.experiments.training.config import (
    GateConfig,
    GridConfig,
    TrainingGridConfig,
)
from eco_planner.experiments.training.decisions import evaluate_heldout_change, evaluate_update_gate
from eco_planner.experiments.training.grid import (
    arm_label,
    compose_arm_overrides,
)
from eco_planner.planning.policy import ExplorationPolicy
from eco_planner.planning.policy.distribution import AffineBeta
from eco_planner.rl.optimization.update_diagnostics import (
    extract_arm_metrics,
    policy_ratio_change,
    post_update_kl_series,
)
from tests.training.test_ppo import _policy_config


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


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        "actor_head.weight": torch.ones(4, 8),
        "actor_head.bias": torch.zeros(4),
        "value_head.weight": torch.ones(1, 8),
        "value_head.bias": torch.zeros(1),
        "fusion_trunk.0.weight": torch.zeros(8, 8),
        "reference_mixers.0.weight": torch.zeros(8, 8),
    }


def test_grid_combinations_are_complete_and_sorted() -> None:
    grid = GridConfig(
        learning_rates=[1.0e-05, 1.0e-04], epochs=[1, 2], max_gradient_norms=[0.5, 2.0]
    )
    combinations = grid.combinations()
    assert combinations == (
        (1.0e-05, 1, 0.5),
        (1.0e-05, 1, 2.0),
        (1.0e-05, 2, 0.5),
        (1.0e-05, 2, 2.0),
        (1.0e-04, 1, 0.5),
        (1.0e-04, 1, 2.0),
        (1.0e-04, 2, 0.5),
        (1.0e-04, 2, 2.0),
    )


@pytest.mark.parametrize(
    "learning_rates",
    [[1.0e-04, 1.0e-05], [1.0e-05, 1.0e-05], [-1.0e-05, 1.0e-05]],
)
def test_grid_rejects_invalid_learning_rates(learning_rates: list[float]) -> None:
    with pytest.raises(ValueError, match="grid.learning_rates"):
        GridConfig(learning_rates=learning_rates, epochs=[1], max_gradient_norms=[0.5])


def test_policy_ratio_change_is_median_of_joint_deviation() -> None:
    means = [1.0, 0.999, 1.001]
    stds = [1.0e-07, 1.0e-07, 1.0e-07]
    assert policy_ratio_change(means, stds) == pytest.approx(1.0e-03)
    with pytest.raises(ValueError, match="ratio statistics"):
        policy_ratio_change([1.0], [])


def test_arm_label_and_overrides_carry_grid_values() -> None:
    assert arm_label(1.6301e-05, 2, 0.5) == "lr1.6301e-05-epochs2-mgn0.5"
    study = _study()
    overrides = compose_arm_overrides(study, 1.6301e-05, 2, 1.0)
    assert overrides[:1] == ["components/resources=rtx_a4000"]
    assert "ppo.learning_rate=1.6301e-05" in overrides
    assert "ppo.epochs=2" in overrides
    assert "ppo.max_gradient_norm=1" in overrides
    assert "training.update_count=50" in overrides
    assert "ppo.scheduler_total_optimizer_steps=100" in overrides


def test_gate_f_passes_effective_arm_and_flags_under_update() -> None:
    effective = evaluate_update_gate(_metrics(), _gate())
    assert effective["passed_1_6"] is True
    assert effective["under_update"] is False
    under = evaluate_update_gate(
        _metrics(
            post_update_kl=[1.0e-08] * 50,
            post_update_kl_median=1.0e-08,
            policy_ratio_change=1.0e-05,
            probe_guidance_rms_shift=6.0e-04,
        ),
        _gate(),
    )
    assert under["passed_1_6"] is False
    assert under["under_update"] is True
    assert "c1_median_kl_above_floor" in under["failure_reasons"]
    assert "c3_policy_ratio_moves" in under["failure_reasons"]
    assert "c4_guidance_rms_shift" in under["failure_reasons"]


def test_gate_f_detects_runaway_tail() -> None:
    kl = [1.0e-05] * 40 + [1.0e-03] * 10
    result = evaluate_update_gate(
        _metrics(post_update_kl=kl, post_update_kl_median=1.0e-05), _gate()
    )
    assert result["conditions"]["c2_kl_stable_within_target"] is False
    assert result["kl_runaway"] is True


def test_gate_f_failed_arm_has_no_passing_condition() -> None:
    result = evaluate_update_gate(None, _gate())
    assert result["passed_1_6"] is False
    assert result["failure_reasons"] == ["training_run_failed"]
    assert all(value is False for value in result["conditions"].values())


def test_gate_f_rejects_boundary_collapse_and_behavioral_collapse() -> None:
    boundary = evaluate_update_gate(
        _metrics(min_beta_alpha=0.05, probe_boundary_mass_max_after=0.4), _gate()
    )
    assert boundary["conditions"]["c5_no_beta_boundary_collapse"] is False
    behavioral = evaluate_update_gate(
        _metrics(
            behavior={
                "collision_count": 1,
                "out_of_road_count": 2,
                "episode_length_first_median": 8.0,
                "episode_length_last_median": 2.0,
            }
        ),
        _gate(),
    )
    assert behavioral["conditions"]["c6_no_behavioral_collapse"] is False


def _write_run_dir(path: Path, update_count: int) -> None:
    updates = []
    for index in range(update_count):
        updates.append(
            {
                "mean_approximate_kl": 1.0e-05 + index * 1.0e-07,
                "mean_clip_fraction": 0.01,
                "maximum_pre_clip_gradient_norm": 4.8,
                "policy_ratio_mean": 1.001,
                "policy_ratio_std": 2.0e-03,
                "policy_ratio_p95": 1.004,
                "mean_policy_loss": -1.0e-03,
                "mean_value_loss": 7.1,
                "mean_entropy_loss": -0.011,
                "mean_entropy": 1.13,
                "mean_explained_variance": 0.1,
                "beta_alpha_mean": [2.0, 2.0],
                "beta_alpha_min": [1.9, 1.9],
                "beta_beta_mean": [2.0, 2.0],
                "beta_beta_min": [1.9, 1.9],
                "action_mean": [0.0, 0.0],
                "action_std": [0.4, 0.4],
                "collision_count": 0,
                "out_of_road_count": 0,
                "mean_episode_length": 8.0,
                "kl_early_stopped": False,
            }
        )
    summary = {
        "initial_policy_hash": "a" * 64,
        "final_policy_hash": "b" * 64,
        "probe_before": {
            "guidance_mean": ((0.0, 0.0),),
            "boundary_mass": ((0.01, 0.01),),
        },
        "probe_after": {
            "guidance_mean": ((0.02, 0.03),),
            "boundary_mass": ((0.02, 0.02),),
        },
        "updates": updates,
    }
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    reference = _state_dict()
    _save_checkpoint(path / "policy-initial.pt", reference)
    for index in range(update_count):
        state = dict(reference)
        state["actor_head.bias"] = torch.full((4,), 0.001 * (index + 1))
        _save_checkpoint(path / f"policy-update-{index:03d}.pt", state)
    _save_checkpoint(path / "policy-final.pt", {**reference, "actor_head.bias": torch.zeros(4)})


def _save_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    torch.save({"format_version": 1, "policy_state_dict": state_dict}, path)


def test_extract_arm_metrics_reads_persisted_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "arm"
    _write_run_dir(run_dir, 4)
    metrics = extract_arm_metrics(run_dir, max_gradient_norm=0.5, update_count=4)
    assert metrics["initial_policy_hash"] == "a" * 64
    assert metrics["pre_update_kl_median"] == pytest.approx(1.0e-05 + 1.5e-07)
    assert metrics["post_clip_gradient_norm"] == [0.5] * 4
    assert metrics["pre_clip_gradient_norm_median"] == pytest.approx(4.8)
    assert metrics["probe_guidance_rms_shift"] == pytest.approx(math.sqrt(0.00065))
    assert metrics["min_beta_alpha"] == pytest.approx(1.9)
    assert len(metrics["parameter_delta_vs_initial"]) == 4
    assert metrics["parameter_delta_vs_initial"][3]["actor_head"] == pytest.approx(0.008)
    assert metrics["parameter_delta_vs_initial_final"]["actor_head"] == pytest.approx(0.0)
    assert metrics["behavior"]["collision_count"] == 0
    with pytest.raises(ValueError, match="expected 3"):
        extract_arm_metrics(run_dir, max_gradient_norm=0.5, update_count=3)


def test_post_update_kl_series_recomputes_kl_on_persisted_updates(tmp_path: Path) -> None:
    torch.manual_seed(0)
    policy = ExplorationPolicy(_policy_config())
    run_dir = tmp_path / "kl-arm"
    run_dir.mkdir()
    OmegaConf.save({"policy": policy.config.model_dump()}, run_dir / "resolved_config.yaml")
    batch = 4
    context = {
        "scene_tokens": torch.randn(batch, 2, 12),
        "scene_padding_mask": torch.zeros(batch, 2, dtype=torch.bool),
        "navigation_tokens": torch.randn(batch, 1, 12),
        "navigation_padding_mask": torch.zeros(batch, 1, dtype=torch.bool),
        "reference_trajectory": torch.randn(batch, 80, 4),
    }
    with torch.no_grad():
        alpha, beta, _ = policy.forward_tensors(*[context[key] for key in context])
    old_distribution = AffineBeta(alpha, beta, validate_args=False)
    action = old_distribution.sample(torch.Generator().manual_seed(7)).guidance_action
    old_log_prob = old_distribution.log_prob(action)
    _write_kl_update(run_dir, 0, context, action, old_log_prob, alpha, beta, policy.state_dict())
    perturbed = {key: value.clone() for key, value in policy.state_dict().items()}
    actor_key = next(key for key in perturbed if key.startswith("actor_head."))
    perturbed[actor_key] = perturbed[actor_key] + 1.0
    _write_kl_update(run_dir, 1, context, action, old_log_prob, alpha, beta, perturbed)
    series = post_update_kl_series(run_dir, update_count=2, mc_draws=4096, mc_seed=1000003)
    assert series["post_update_kl"][0] == pytest.approx(0.0, abs=1.0e-12)
    assert series["post_update_kl"][1] > 1.0e-03
    assert all(math.isfinite(value) for value in series["post_update_kl"])
    assert series["post_update_kl_single_draw_k1"][0] == pytest.approx(0.0, abs=1.0e-12)
    assert series["post_update_kl_single_draw_k3_max"] > 1.0e-06
    assert series["post_update_kl_single_draw_k3_median"] == pytest.approx(
        0.5 * series["post_update_kl_single_draw_k3_max"]
    )
    assert series["post_update_kl_median"] == pytest.approx(
        0.5 * (series["post_update_kl"][0] + series["post_update_kl"][1])
    )
    with pytest.raises(ValueError, match="no persisted rollout episodes"):
        post_update_kl_series(run_dir, update_count=3, mc_draws=4096, mc_seed=1000003)


def _write_kl_update(
    run_dir: Path,
    update_index: int,
    context: dict[str, torch.Tensor],
    action: torch.Tensor,
    old_log_prob: torch.Tensor,
    old_alpha: torch.Tensor,
    old_beta: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
) -> None:
    update_dir = run_dir / "updates" / f"update-{update_index:03d}"
    update_dir.mkdir(parents=True)
    arrays = {key: value.numpy() for key, value in context.items()}
    arrays["guidance_action"] = action.numpy()
    arrays["old_joint_guidance_log_prob"] = old_log_prob.numpy()
    arrays["beta_alpha"] = old_alpha.numpy()
    arrays["beta_beta"] = old_beta.numpy()
    np.savez(update_dir / "slot-0-episode-0.npz", **arrays)
    _save_checkpoint(run_dir / f"policy-update-{update_index:03d}.pt", state_dict)


def test_heldout_values_and_noise_exceedance() -> None:
    def episode(speed: float, distance: float, energy: float) -> dict[str, Any]:
        return {
            "metrics": {
                "speed_mps": {"mean": speed},
                "distance_m": distance,
                "route_completion": 0.9,
                "energy": {"total_ml": energy, "ml_per_km": energy / distance * 1000.0},
                "arrive_dest": True,
                "collision": False,
                "out_of_road": False,
            }
        }

    initial = heldout_metric_values([episode(10.0, 100.0, 5.0)])
    final = heldout_metric_values([episode(10.0, 100.0, 5.0)])
    unchanged = evaluate_heldout_change(initial, final, _gate())
    assert unchanged["exceeds_noise"] is False
    moved = heldout_metric_values([episode(10.0, 100.0, 5.05)])
    changed = evaluate_heldout_change(initial, moved, _gate())
    assert changed["exceeds_noise"] is True
    assert changed["exceeds_noise_metrics"] == ["energy_total_ml", "energy_ml_per_km"]
    assert changed["relative_change"]["energy_total_ml"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="at least one episode"):
        heldout_metric_values([])


def test_beta_probe_statistics_aggregates_affine_beta_moments() -> None:
    probe = {
        "alpha": [[2.0, 2.0], [2.0, 2.0]],
        "beta": [[2.0, 2.0], [2.0, 2.0]],
    }

    statistics = beta_probe_statistics(probe)

    assert statistics["context_count"] == 2
    assert statistics["beta_mean"]["mean"] == pytest.approx([0.0, 0.0])
    assert statistics["concentration"]["mean"] == pytest.approx([4.0, 4.0])
    assert statistics["variance"]["mean"] == pytest.approx([0.2, 0.2])


def test_paired_beta_deltas_reports_per_dimension_mean_and_rms() -> None:
    reference = {"alpha": [[2.0, 2.0]], "beta": [[2.0, 2.0]]}
    stress = {"alpha": [[1.0, 2.0]], "beta": [[2.0, 2.0]]}

    deltas = paired_beta_deltas(reference, stress)

    assert deltas["concentration"]["mean_delta_per_dimension"] == pytest.approx([-1.0, 0.0])
    assert deltas["beta_mean"]["mean_delta_per_dimension"][1] == pytest.approx(0.0)
    assert deltas["beta_mean"]["mean_delta_per_dimension"][0] < 0.0
    assert deltas["variance"]["rms"] > 0.0
