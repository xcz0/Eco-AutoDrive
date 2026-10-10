"""Measure saved training policies, updates and probe changes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Beta

from eco_planner.planning.policy import (
    POLICY_CONTEXT_KEYS,
    AffineBeta,
    AffineBetaParameters,
    ExplicitGeneratorBetaSampler,
    ExplorationPolicy,
    parse_exploration_policy_config,
)
from eco_planner.planning.policy.distribution import beta_kl

_PARAMETER_GROUPS = ("actor_head", "value_head", "shared_trunk")
# Per-group parameter-gradient norms persisted only when
# ``ppo.gradient_diagnostics`` is enabled (measurement-only diagnostic).
GRADIENT_GROUPS = (
    "actor_head_policy",
    "shared_trunk_policy",
    "value_head_critic",
    "shared_trunk_critic",
    "actor_head_entropy",
    "shared_trunk_entropy",
)
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


def cosine_learning_rate(
    step: int,
    *,
    base_learning_rate: float,
    minimum_learning_rate: float,
    total_optimizer_steps: int,
) -> float:
    """Closed-form ``CosineAnnealingLR`` learning rate after ``step`` steps."""

    if step < 0 or step > total_optimizer_steps:
        raise ValueError("cosine learning rate step is outside the scheduler horizon")
    return minimum_learning_rate + (
        (base_learning_rate - minimum_learning_rate)
        * (1.0 + math.cos(math.pi * step / total_optimizer_steps))
        / 2.0
    )


def learning_rate_history(run_dir: Path, updates: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the scheduler-matched per-update learning-rate history.

    The frozen cosine schedule is fully determined by the resolved PPO config,
    so the learning rate of every optimizer step is rebuilt in closed form from
    the persisted per-update ``optimizer_step_count`` (zero for KL-early-stopped
    updates). The reconstruction is cross-checked against each update's recorded
    ``final_learning_rate``; a mismatch means the run was not trained under the
    resolved scheduler horizon and is raised instead of silently reported.
    ``learning_rate_series[i]`` is the rate in effect at update ``i``'s first
    optimizer step; ``cumulative_learning_rate_exposure[i]`` is the running sum
    of the per-step learning rates through the update's last optimizer step.
    """

    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    if not isinstance(config, DictConfig):
        raise TypeError(f"resolved training config must be a mapping: {run_dir}")
    ppo = config["ppo"]
    base_learning_rate = float(ppo["learning_rate"])
    minimum_learning_rate = float(ppo["scheduler_minimum_learning_rate"])
    total_optimizer_steps = int(ppo["scheduler_total_optimizer_steps"])

    def rate_at(step: int) -> float:
        return cosine_learning_rate(
            step,
            base_learning_rate=base_learning_rate,
            minimum_learning_rate=minimum_learning_rate,
            total_optimizer_steps=total_optimizer_steps,
        )

    series: list[float] = []
    exposure: list[float] = []
    completed_steps: list[int] = []
    completed = 0
    cumulative_exposure = 0.0
    for index, update in enumerate(updates):
        steps = update["optimizer_step_count"]
        if type(steps) is not int or steps < 0:
            raise ValueError(f"optimizer step count is invalid (update {index})")
        series.append(rate_at(completed))
        for _ in range(steps):
            cumulative_exposure += rate_at(completed)
            completed += 1
        expected_final = rate_at(completed)
        recorded_final = float(update["final_learning_rate"])
        if not math.isclose(expected_final, recorded_final, rel_tol=1.0e-6, abs_tol=1.0e-12):
            raise ValueError(
                f"recorded final learning rate {recorded_final} disagrees with the "
                f"resolved cosine schedule {expected_final} (update {index})"
            )
        exposure.append(cumulative_exposure)
        completed_steps.append(completed)
    return {
        "learning_rate_initial": base_learning_rate,
        "learning_rate_series": series,
        "cumulative_learning_rate_exposure": exposure,
        "cumulative_optimizer_steps": completed_steps,
    }


def mc_log_ratio_draws(
    old_alpha: torch.Tensor,
    old_beta: torch.Tensor,
    new_alpha: torch.Tensor,
    new_beta: torch.Tensor,
    base_action: torch.Tensor,
) -> torch.Tensor:
    """Evaluate seeded-MC log ratios in base coordinates at float64 precision.

    Evaluating in guidance coordinates round-trips ``2u - 1`` then ``(g + 1) /
    2``; in float32 this collapses base draws below ``2**-25`` onto the exact
    boundary, turning mathematically finite interior ratios into ``-inf -
    (-inf)`` for near-floor concentrations.  The affine Jacobian cancels in the
    ratio, so the base-coordinate evaluation is exact; genuinely
    boundary-singular draws still evaluate non-finite.
    """

    old_distribution = Beta(old_alpha.double(), old_beta.double(), validate_args=False)
    new_distribution = Beta(new_alpha.double(), new_beta.double(), validate_args=False)
    draws = base_action.double()
    return (new_distribution.log_prob(draws) - old_distribution.log_prob(draws)).sum(dim=-1)


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

    The persisted old Beta parameters and the recomputed new parameters also
    give a deterministic, sampling-free analytic ``KL(old || new)`` (sum of the
    two per-dimension Beta KLs, the affine Jacobian cancelling), reported as
    ``post_update_kl_analytic`` alongside the seeded-MC and k1/k3 estimates.
    """

    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    if not isinstance(config, DictConfig):
        raise TypeError(f"resolved training config must be a mapping: {run_dir}")
    policy = ExplorationPolicy(parse_exploration_policy_config(config["policy"]))
    policy.eval()
    kl_series: list[float] = []
    analytic_series: list[float] = []
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
            analytic_series.append(float(beta_kl(old_alpha, old_beta, new_alpha, new_beta).mean()))
            batch_size = old_alpha.shape[0]
            generator = torch.Generator()
            generator.manual_seed(mc_seed + index)
            repeated = AffineBetaParameters(
                alpha=old_alpha.repeat(mc_draws, 1), beta=old_beta.repeat(mc_draws, 1)
            )
            base_action = ExplicitGeneratorBetaSampler.draw(
                repeated, generator, validate_args=False
            )
            log_ratio_draws = mc_log_ratio_draws(
                repeated.alpha,
                repeated.beta,
                new_alpha.repeat(mc_draws, 1),
                new_beta.repeat(mc_draws, 1),
                base_action,
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
        "post_update_kl_analytic": analytic_series,
        "post_update_kl_analytic_median": _median(analytic_series),
        "post_update_kl_analytic_max": max(analytic_series),
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


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def parameter_delta_series(
    run_dir: Path, *, update_count: int
) -> tuple[list[str], list[dict[str, float]], dict[str, float]]:
    """Measure saved update/final policy deltas against the saved initial policy."""
    initial = load_policy_state_dict(run_dir / "policy-initial.pt")
    groups = parameter_groups(list(initial))
    deltas = []
    for index in range(update_count):
        state = load_policy_state_dict(run_dir / f"policy-update-{index:03d}.pt")
        deltas.append(parameter_delta_vs_reference(state, initial, groups))
    final = load_policy_state_dict(run_dir / "policy-final.pt")
    final_delta = parameter_delta_vs_reference(final, initial, groups)
    return list(_PARAMETER_GROUPS), deltas, final_delta
