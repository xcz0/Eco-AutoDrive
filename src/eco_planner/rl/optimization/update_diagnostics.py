"""Measure saved training policies, updates and probe changes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from eco_planner.planning.policy import (
    POLICY_CONTEXT_KEYS,
    AffineBeta,
    AffineBetaParameters,
    ExplicitGeneratorBetaSampler,
    ExplorationPolicy,
    parse_exploration_policy_config,
)

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
