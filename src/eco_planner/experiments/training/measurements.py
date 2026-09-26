"""Training-study summaries assembled from explicit RL measurements."""

import json
from pathlib import Path
from typing import Any

from eco_planner.planning.policy.statistics import beta_statistics


def extract_arm_metrics(
    run_dir: Path, *, max_gradient_norm: float, update_count: int
) -> dict[str, Any]:
    """Read one completed training run's persisted diagnostics."""

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    updates = summary["updates"]
    if len(updates) != update_count:
        raise ValueError(
            f"training run {run_dir} has {len(updates)} updates, expected {update_count}"
        )
    kl = [update["mean_approximate_kl"] for update in updates]
    clip_fraction = [update["mean_clip_fraction"] for update in updates]
    pre_clip_norm = [update["maximum_pre_clip_gradient_norm"] for update in updates]
    ratio_means = [update["policy_ratio_mean"] for update in updates]
    ratio_stds = [update["policy_ratio_std"] for update in updates]
    ratio_p95 = [update["policy_ratio_p95"] for update in updates]
    episode_lengths = [update["mean_episode_length"] for update in updates]
    early_stops = sum(1 for update in updates if update["kl_early_stopped"])
    from eco_planner.rl.optimization.update_diagnostics import (
        parameter_delta_series,
        post_clip_gradient_norm,
        probe_guidance_rms_shift,
    )

    groups, deltas, final_delta = parameter_delta_series(run_dir, update_count=update_count)
    first, last = updates[0], updates[-1]
    return {
        "run_dir": str(run_dir),
        "update_count": len(updates),
        "initial_policy_hash": summary["initial_policy_hash"],
        "final_policy_hash": summary["final_policy_hash"],
        # TorchRL's in-update kl_approx, measured at the loss forward (before
        # the optimizer step); structurally ~0 with one minibatch per epoch.
        "pre_update_kl": kl,
        "pre_update_kl_median": _median(kl),
        "pre_update_kl_max": max(kl),
        "kl_early_stop_fraction": early_stops / len(updates),
        "clip_fraction": clip_fraction,
        "clip_fraction_median": _median(clip_fraction),
        "pre_clip_gradient_norm": pre_clip_norm,
        "post_clip_gradient_norm": [
            post_clip_gradient_norm(value, max_gradient_norm) for value in pre_clip_norm
        ],
        "pre_clip_gradient_norm_median": _median(pre_clip_norm),
        "policy_ratio_change": policy_ratio_change(ratio_means, ratio_stds),
        "policy_ratio_mean_median": _median(ratio_means),
        "policy_ratio_std_median": _median(ratio_stds),
        "policy_ratio_p95_median": _median(ratio_p95),
        "policy_ratio_mean": ratio_means,
        "policy_ratio_std": ratio_stds,
        "parameter_delta_groups": groups,
        "parameter_delta_vs_initial": deltas,
        "parameter_delta_vs_initial_final": final_delta,
        "beta_initial": _beta_summary(first),
        "beta_final": _beta_summary(last),
        "min_beta_alpha": min(value for update in updates for value in update["beta_alpha_min"]),
        "min_beta_beta": min(value for update in updates for value in update["beta_beta_min"]),
        "action_mean_initial": first["action_mean"],
        "action_mean_final": last["action_mean"],
        "action_std_initial": first["action_std"],
        "action_std_final": last["action_std"],
        "probe_guidance_rms_shift": probe_guidance_rms_shift(
            summary["probe_before"]["guidance_mean"],
            summary["probe_after"]["guidance_mean"],
        ),
        "probe_boundary_mass_max_after": max(
            value for row in summary["probe_after"]["boundary_mass"] for value in row
        ),
        "probe_boundary_mass_max_before": max(
            value for row in summary["probe_before"]["boundary_mass"] for value in row
        ),
        "losses": {
            "policy_loss_initial": first["mean_policy_loss"],
            "policy_loss_final": last["mean_policy_loss"],
            "value_loss_initial": first["mean_value_loss"],
            "value_loss_final": last["mean_value_loss"],
            "entropy_loss_initial": first["mean_entropy_loss"],
            "entropy_loss_final": last["mean_entropy_loss"],
            "entropy_initial": first["mean_entropy"],
            "entropy_final": last["mean_entropy"],
            "explained_variance_initial": first["mean_explained_variance"],
            "explained_variance_final": last["mean_explained_variance"],
        },
        "behavior": {
            "collision_count": sum(update["collision_count"] for update in updates),
            "out_of_road_count": sum(update["out_of_road_count"] for update in updates),
            "episode_length_first_median": _median(episode_lengths[: _tail_length(len(updates))]),
            "episode_length_last_median": _median(episode_lengths[-_tail_length(len(updates)) :]),
        },
    }


def policy_ratio_change(ratio_means: list[float], ratio_stds: list[float]) -> float:
    """Median over updates of ``max(|ratio_mean - 1|, ratio_std)``."""

    if len(ratio_means) != len(ratio_stds) or not ratio_means:
        raise ValueError("ratio statistics must share one non-empty update sequence")
    changes = [max(abs(mean - 1.0), std) for mean, std in zip(ratio_means, ratio_stds, strict=True)]
    return _median(changes)


def _beta_summary(update: dict[str, Any]) -> dict[str, Any]:
    statistics = {}
    for dimension, (alpha, beta) in enumerate(
        zip(update["beta_alpha_mean"], update["beta_beta_mean"], strict=True)
    ):
        mean, concentration, variance = beta_statistics(alpha, beta)
        statistics[f"dim{dimension}"] = {
            "alpha": alpha,
            "beta": beta,
            "mean": mean,
            "concentration": concentration,
            "variance": variance,
        }
    return statistics


def _tail_length(update_count: int) -> int:
    return min(10, update_count)


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])
