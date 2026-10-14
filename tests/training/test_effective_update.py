from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from eco_planner.experiments.training.config import (
    GridConfig,
)
from eco_planner.experiments.training.decisions import evaluate_heldout_change, evaluate_update_gate
from eco_planner.experiments.training.grid import (
    compose_arm_overrides,
)
from eco_planner.experiments.training.measurements import extract_arm_metrics, policy_ratio_change
from eco_planner.experiments.training.statistics import (
    beta_probe_statistics,
    heldout_metric_values,
    paired_beta_deltas,
)
from eco_planner.planning.policy import ExplorationPolicy
from eco_planner.planning.policy.distribution import AffineBeta, beta_kl
from eco_planner.rl.optimization.update_diagnostics import (
    learning_rate_history,
    mc_log_ratio_draws,
    post_update_kl_series,
)
from tests.training.helpers import _policy_config
from tests.training.workflow_helpers import _gate, _metrics, _study


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


def test_overrides_carry_grid_values() -> None:
    study = _study()
    overrides = compose_arm_overrides(study, 1.6301e-05, 2, 1.0)
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


def _cosine_final_learning_rates(step_counts: list[int]) -> list[float]:
    """Final per-update learning rates from a real CosineAnnealingLR schedule."""

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sum(step_counts), eta_min=0.0
    )
    values = []
    for steps in step_counts:
        for _ in range(steps):
            optimizer.step()
            scheduler.step()
        values.append(float(optimizer.param_groups[0]["lr"]))
    return values


def _write_run_dir(path: Path, update_count: int, *, gradient_diagnostics: bool = False) -> None:
    final_learning_rates = _cosine_final_learning_rates([1] * update_count)
    updates = []
    for index in range(update_count):
        updates.append(
            {
                "gradient_diagnostics": (
                    {
                        "actor_head_policy": 0.3 + index,
                        "shared_trunk_policy": 0.4,
                        "value_head_critic": 0.5,
                        "shared_trunk_critic": 0.6,
                        "actor_head_entropy": 0.7,
                        "shared_trunk_entropy": 0.8,
                    }
                    if gradient_diagnostics
                    else None
                ),
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
                "optimizer_step_count": 1,
                "final_learning_rate": final_learning_rates[index],
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
    OmegaConf.save(
        {
            "ppo": {
                "learning_rate": 1.5e-4,
                "scheduler_minimum_learning_rate": 0.0,
                "scheduler_total_optimizer_steps": update_count,
            }
        },
        path / "resolved_config.yaml",
    )
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
    assert [entry["dim0"]["alpha"] for entry in metrics["beta_series"]] == [2.0] * 4
    assert metrics["beta_series"][0] == metrics["beta_initial"]
    assert metrics["beta_series"][-1] == metrics["beta_final"]
    assert len(metrics["parameter_delta_vs_initial"]) == 4
    assert metrics["parameter_delta_vs_initial"][3]["actor_head"] == pytest.approx(0.008)
    assert metrics["parameter_delta_vs_initial_final"]["actor_head"] == pytest.approx(0.0)
    assert metrics["behavior"]["collision_count"] == 0
    assert metrics["gradient_diagnostics"] == {}
    assert metrics["learning_rate_initial"] == pytest.approx(1.5e-4)
    expected_series = [1.5e-4 * (1.0 + math.cos(math.pi * step / 4)) / 2.0 for step in range(4)]
    assert metrics["learning_rate_series"] == pytest.approx(expected_series)
    assert metrics["cumulative_learning_rate_exposure"] == pytest.approx(
        [sum(expected_series[: index + 1]) for index in range(4)]
    )
    assert metrics["cumulative_optimizer_steps"] == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="expected 3"):
        extract_arm_metrics(run_dir, max_gradient_norm=0.5, update_count=3)


def test_extract_arm_metrics_reports_per_group_gradient_diagnostics(tmp_path: Path) -> None:
    run_dir = tmp_path / "arm-gradient"
    _write_run_dir(run_dir, 4, gradient_diagnostics=True)
    metrics = extract_arm_metrics(run_dir, max_gradient_norm=0.5, update_count=4)
    assert metrics["gradient_diagnostics"]["actor_head_policy"] == [0.3, 1.3, 2.3, 3.3]
    assert metrics["gradient_diagnostics"]["shared_trunk_policy"] == [0.4] * 4
    assert metrics["gradient_diagnostics"]["shared_trunk_entropy"] == [0.8] * 4


def test_learning_rate_history_handles_zero_step_update(tmp_path: Path) -> None:
    run_dir = tmp_path / "early-stop-arm"
    run_dir.mkdir()
    step_counts = [1, 1, 0]
    OmegaConf.save(
        {
            "ppo": {
                "learning_rate": 1.5e-4,
                "scheduler_minimum_learning_rate": 0.0,
                "scheduler_total_optimizer_steps": sum(step_counts),
            }
        },
        run_dir / "resolved_config.yaml",
    )
    finals = _cosine_final_learning_rates(step_counts)
    updates = [
        {"optimizer_step_count": steps, "final_learning_rate": final}
        for steps, final in zip(step_counts, finals, strict=True)
    ]
    history = learning_rate_history(run_dir, updates)
    assert history["learning_rate_initial"] == pytest.approx(1.5e-4)
    assert history["cumulative_optimizer_steps"] == [1, 2, 2]
    assert history["learning_rate_series"][2] == pytest.approx(finals[1])
    assert history["cumulative_learning_rate_exposure"][2] == pytest.approx(
        history["cumulative_learning_rate_exposure"][1]
    )
    assert history["cumulative_learning_rate_exposure"][1] == pytest.approx(
        history["learning_rate_series"][0] + history["learning_rate_series"][1]
    )


def test_learning_rate_history_rejects_schedule_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "mismatched-arm"
    run_dir.mkdir()
    finals = _cosine_final_learning_rates([1, 1])
    updates = [
        {"optimizer_step_count": steps, "final_learning_rate": final}
        for steps, final in zip([1, 1], finals, strict=True)
    ]

    def _save_resolved_config(total_optimizer_steps: int, learning_rate: float) -> None:
        OmegaConf.save(
            {
                "ppo": {
                    "learning_rate": learning_rate,
                    "scheduler_minimum_learning_rate": 0.0,
                    "scheduler_total_optimizer_steps": total_optimizer_steps,
                }
            },
            run_dir / "resolved_config.yaml",
        )

    _save_resolved_config(2, 1.5e-4)
    tampered = [{**updates[0], "final_learning_rate": 1.0e-4}, updates[1]]
    with pytest.raises(ValueError, match="disagrees with the resolved cosine schedule"):
        learning_rate_history(run_dir, tampered)
    _save_resolved_config(2, 2.5e-4)
    with pytest.raises(ValueError, match="disagrees with the resolved cosine schedule"):
        learning_rate_history(run_dir, updates)
    _save_resolved_config(1, 1.5e-4)
    with pytest.raises(ValueError, match="outside the scheduler horizon"):
        learning_rate_history(
            run_dir, [{"optimizer_step_count": 2, "final_learning_rate": finals[0]}]
        )
    _save_resolved_config(2, 1.5e-4)
    with pytest.raises(ValueError, match="optimizer step count is invalid"):
        learning_rate_history(
            run_dir, [{"optimizer_step_count": 1.0, "final_learning_rate": finals[0]}]
        )


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
    assert series["post_update_kl_analytic"][0] == pytest.approx(0.0, abs=1.0e-12)
    assert series["post_update_kl_analytic"][1] > series["post_update_kl"][1]
    assert all(math.isfinite(value) for value in series["post_update_kl_analytic"])
    assert series["post_update_kl_analytic_median"] == pytest.approx(
        0.5 * (series["post_update_kl_analytic"][0] + series["post_update_kl_analytic"][1])
    )
    assert series["post_update_kl_analytic_max"] == series["post_update_kl_analytic"][1]
    with pytest.raises(ValueError, match="no persisted rollout episodes"):
        post_update_kl_series(run_dir, update_count=3, mc_draws=4096, mc_seed=1000003)


def test_mc_log_ratio_stays_finite_when_float32_rounds_interior_draw_to_boundary() -> None:
    # E-057 lam4-seed-2 update 86: a legitimate interior base draw 1.45e-08
    # under a near-floor Beta(1.12, 4.32) rounds to guidance -1.0 in float32,
    # and the guidance-coordinate round-trip turned the finite true ratio into
    # NaN.  The base-coordinate evaluation keeps the draw intact.
    old_alpha = torch.tensor([[3.000150203704834, 1.1215640306472778]])
    old_beta = torch.tensor([[1.4681456089019775, 4.3188958168029785]])
    new_alpha = torch.tensor([[3.0026626586914062, 1.1203675270080566]])
    new_beta = torch.tensor([[1.4670534133911133, 4.3306474685668945]])
    base_action = torch.tensor([[0.6829089522361755, 1.4503590328729388e-08]])
    assert (2.0 * base_action - 1.0)[0, 1].item() == -1.0
    ratio = mc_log_ratio_draws(old_alpha, old_beta, new_alpha, new_beta, base_action)
    assert torch.isfinite(ratio).all()
    # Genuinely boundary-singular draws must stay non-finite so the
    # FloatingPointError collapse guard keeps its semantics.
    boundary_ratio = mc_log_ratio_draws(
        old_alpha, old_beta, new_alpha, new_beta, torch.tensor([[0.68, 0.0]])
    )
    assert not torch.isfinite(boundary_ratio).all()


def test_beta_kl_matches_closed_form_and_is_zero_for_identical_parameters() -> None:
    identical = beta_kl(
        torch.tensor([[2.0, 3.0]]),
        torch.tensor([[4.0, 1.5]]),
        torch.tensor([[2.0, 3.0]]),
        torch.tensor([[4.0, 1.5]]),
    )
    assert identical.item() == pytest.approx(0.0, abs=1.0e-12)

    uniform_to_beta = beta_kl(
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[2.0, 3.0]]),
        torch.tensor([[2.0, 5.0]]),
    )
    expected_first_dimension = 2.0 - math.log(6.0)
    expected_second_dimension = 6.0 - math.log(105.0)
    assert uniform_to_beta.item() == pytest.approx(
        expected_first_dimension + expected_second_dimension, rel=1.0e-6
    )
    with pytest.raises(ValueError, match="equally shaped"):
        beta_kl(torch.ones(1, 2), torch.ones(1, 2), torch.ones(2, 2), torch.ones(1, 2))


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
