"""Measure saved training policies, updates and probe changes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from eco_planner.analysis.training import beta_statistics
from eco_planner.planning.policy import ExplorationPolicy
from eco_planner.planning.policy.config import parse_exploration_policy_config
from eco_planner.planning.policy.distribution import (
    AffineBeta,
    AffineBetaParameters,
    ExplicitGeneratorBetaSampler,
)
from eco_planner.planning.policy.model import POLICY_CONTEXT_KEYS

_PARAMETER_GROUPS = ("actor_head", "value_head", "shared_trunk")
_KL_BATCH_KEYS = (
    *POLICY_CONTEXT_KEYS,
    "guidance_action",
    "old_joint_guidance_log_prob",
    "beta_alpha",
    "beta_beta",
)


def post_clip_gradient_norm(pre_clip_norm: float, max_gradient_norm: float) -> float:
    """Exact post-clip total norm under global ``clip_grad_norm_`` semantics."""

    if max_gradient_norm <= 0.0:
        raise ValueError("max_gradient_norm must be positive")
    return min(pre_clip_norm, max_gradient_norm)


def probe_guidance_rms_shift(
    before: tuple[tuple[float, float], ...], after: tuple[tuple[float, float], ...]
) -> float:
    """Batch-level RMS shift of the deterministic Beta-mean probe output."""

    if len(before) != len(after) or not before:
        raise ValueError("probe summaries must share one non-empty scenario set")
    squared = 0.0
    for row_before, row_after in zip(before, after, strict=True):
        squared += sum(
            (value_after - value_before) ** 2
            for value_before, value_after in zip(row_before, row_after, strict=True)
        )
    return math.sqrt(squared / (len(before) * len(before[0])))


def policy_ratio_change(ratio_means: list[float], ratio_stds: list[float]) -> float:
    """Median over updates of ``max(|ratio_mean - 1|, ratio_std)``."""

    if len(ratio_means) != len(ratio_stds) or not ratio_means:
        raise ValueError("ratio statistics must share one non-empty update sequence")
    changes = [max(abs(mean - 1.0), std) for mean, std in zip(ratio_means, ratio_stds, strict=True)]
    return _median(changes)


def parameter_groups(keys: list[str]) -> dict[str, list[str]]:
    """Split policy state-dict keys into actor-head / value-head / shared-trunk."""

    groups: dict[str, list[str]] = {name: [] for name in _PARAMETER_GROUPS}
    for key in keys:
        if key.startswith("actor_head."):
            groups["actor_head"].append(key)
        elif key.startswith("value_head."):
            groups["value_head"].append(key)
        else:
            groups["shared_trunk"].append(key)
    if any(not members for members in groups.values()):
        raise ValueError(
            "policy state dict must contain actor_head, value_head, and shared-trunk keys"
        )
    return groups


def load_policy_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load one policy-only checkpoint written by ``save_exploration_policy_checkpoint``."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "format_version",
        "policy_state_dict",
    }:
        raise ValueError(f"invalid policy checkpoint structure: {path}")
    state_dict = checkpoint["policy_state_dict"]
    if not isinstance(state_dict, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise TypeError(f"policy_state_dict must map names to tensors: {path}")
    return state_dict


def parameter_delta_vs_reference(
    state: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    groups: dict[str, list[str]],
) -> dict[str, float]:
    """Per-group L2 parameter delta of one state against a reference state."""

    if set(state) != set(reference):
        raise ValueError("parameter delta requires matching state-dict keys")
    deltas = {}
    for group, keys in groups.items():
        squared = 0.0
        for key in keys:
            difference = state[key].to(dtype=torch.float64) - reference[key].to(dtype=torch.float64)
            squared += float(difference.square().sum())
        deltas[group] = math.sqrt(squared)
    return deltas


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
    initial = load_policy_state_dict(run_dir / "policy-initial.pt")
    groups = parameter_groups(list(initial))
    deltas = []
    for index in range(len(updates)):
        state = load_policy_state_dict(run_dir / f"policy-update-{index:03d}.pt")
        deltas.append(parameter_delta_vs_reference(state, initial, groups))
    final = load_policy_state_dict(run_dir / "policy-final.pt")
    final_delta = parameter_delta_vs_reference(final, initial, groups)
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
        "parameter_delta_groups": list(_PARAMETER_GROUPS),
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


def post_update_kl_series(
    run_dir: Path, *, update_count: int, mc_draws: int, mc_seed: int
) -> dict[str, Any]:
    """Recompute the post-update approximate KL for every persisted update.

    TorchRL's in-update ``kl_approx`` is evaluated during the loss forward,
    i.e. before the optimizer step. With one minibatch per epoch (this grid's
    batch layout) it therefore measures ``KL(old || policy-at-update-start)``
    and structurally reads ~0 regardless of update size.

    The post-update KL is estimated for the same estimand
    ``E_old[old_log_prob - new_log_prob]`` in two ways. The single persisted
    rollout action per sample gives the first-order k1 (and quadratic k3), but
    with 128 on-policy samples its batch noise (~1e-4) sits far above the gate
    floor (1e-6). The primary estimate therefore Monte-Carlo integrates the
    expectation over the persisted old Beta parameters with
    ``mc_draws`` seeded draws per context, which resolves ~1e-6 medians.
    """

    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    if not isinstance(config, DictConfig):
        raise TypeError(f"resolved training config must be a mapping: {run_dir}")
    policy = ExplorationPolicy(parse_exploration_policy_config(config["policy"]))
    policy.eval()
    kl_series: list[float] = []
    k1_series: list[float] = []
    k3_series: list[float] = []
    for index in range(update_count):
        batch = _load_update_batch(run_dir, index)
        state_dict = load_policy_state_dict(run_dir / f"policy-update-{index:03d}.pt")
        policy.load_state_dict(state_dict, strict=True)
        with torch.no_grad():
            new_alpha, new_beta, _ = policy.forward_tensors(
                *[batch[key] for key in POLICY_CONTEXT_KEYS]
            )
            new_alpha = new_alpha.to(dtype=torch.float32)
            new_beta = new_beta.to(dtype=torch.float32)
            old_alpha = batch["beta_alpha"].to(dtype=torch.float32)
            old_beta = batch["beta_beta"].to(dtype=torch.float32)
            batch_size = old_alpha.shape[0]
            generator = torch.Generator()
            generator.manual_seed(mc_seed + index)
            repeated = AffineBetaParameters(
                alpha=old_alpha.repeat(mc_draws, 1), beta=old_beta.repeat(mc_draws, 1)
            )
            base_action = ExplicitGeneratorBetaSampler.draw(
                repeated, generator, validate_args=False
            )
            guidance_action = 2.0 * base_action - 1.0
            old_distribution = AffineBeta(repeated.alpha, repeated.beta, validate_args=False)
            new_repeated = AffineBeta(
                new_alpha.repeat(mc_draws, 1),
                new_beta.repeat(mc_draws, 1),
                validate_args=False,
            )
            log_ratio_draws = new_repeated.log_prob(guidance_action) - old_distribution.log_prob(
                guidance_action
            )
            if not torch.isfinite(log_ratio_draws).all():
                raise FloatingPointError(
                    f"post-update MC log ratios must be finite (update {index})"
                )
            kl_series.append(float((-log_ratio_draws).reshape(mc_draws, batch_size).mean()))
            new_single = AffineBeta(new_alpha, new_beta, validate_args=False)
            new_log_prob = new_single.log_prob(batch["guidance_action"].to(dtype=torch.float32))
            log_ratio = new_log_prob.reshape(-1) - batch["old_joint_guidance_log_prob"].reshape(
                -1
            ).to(dtype=torch.float32)
            if not torch.isfinite(log_ratio).all():
                raise FloatingPointError(f"post-update log ratio must be finite (update {index})")
        k1_series.append(float(log_ratio.mul(-1.0).mean()))
        k3_series.append(float(log_ratio.square().mean() * 0.5))
    return {
        "post_update_kl": kl_series,
        "post_update_kl_median": _median(kl_series),
        "post_update_kl_max": max(kl_series),
        "post_update_kl_mc_draws_per_context": mc_draws,
        "post_update_kl_single_draw_k1": k1_series,
        "post_update_kl_single_draw_k1_median": _median(k1_series),
        "post_update_kl_single_draw_k3_median": _median(k3_series),
        "post_update_kl_single_draw_k3_max": max(k3_series),
    }


def _load_update_batch(run_dir: Path, update_index: int) -> dict[str, torch.Tensor]:
    update_dir = run_dir / "updates" / f"update-{update_index:03d}"
    parts: dict[str, list[torch.Tensor]] = {key: [] for key in _KL_BATCH_KEYS}
    for path in sorted(update_dir.glob("slot-*-episode-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for key in _KL_BATCH_KEYS:
                parts[key].append(torch.from_numpy(np.ascontiguousarray(data[key])))
    if any(not values for values in parts.values()):
        raise ValueError(f"update {update_index} has no persisted rollout episodes")
    return {key: torch.cat(values) for key, values in parts.items()}


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
