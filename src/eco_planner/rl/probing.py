"""Fixed-context policy diagnostics using independent sampling generators."""

from __future__ import annotations

import torch

from eco_planner.rl.artifacts.schema import PolicyProbeSummary
from eco_planner.rl.policy import ExplorationPolicyContext, policy_context_tensordict
from eco_planner.rl.policy.distribution import AffineBeta
from eco_planner.rl.rollout import FabricRolloutRuntime, RolloutEpisode


def capture_probe_contexts(
    slot_episodes: tuple[tuple[RolloutEpisode, ...], ...], scenario_count: int
) -> tuple[ExplorationPolicyContext, ...]:
    contexts = []
    for episodes in slot_episodes:
        if episodes:
            item = episodes[0].training[0]
            contexts.append(
                ExplorationPolicyContext(
                    scene_tokens=item["scene_tokens"].unsqueeze(0),
                    scene_padding_mask=item["scene_padding_mask"].unsqueeze(0),
                    navigation_tokens=item["navigation_tokens"].unsqueeze(0),
                    navigation_padding_mask=item["navigation_padding_mask"].unsqueeze(0),
                    reference_trajectory=item["reference_trajectory"].unsqueeze(0),
                )
            )
    if len(contexts) != scenario_count:
        raise RuntimeError("training did not capture one fixed probe context per scenario")
    return tuple(contexts)


def probe_policy(
    runtime: FabricRolloutRuntime,
    contexts: tuple[ExplorationPolicyContext, ...],
    sample_count: int,
    boundary_distance: float,
    diagnostic_seed: int,
) -> PolicyProbeSummary:
    alpha_values: list[tuple[float, float]] = []
    beta_values: list[tuple[float, float]] = []
    means: list[tuple[float, float]] = []
    masses: list[tuple[float, float]] = []
    for index, host_context in enumerate(contexts):
        context = _context_to_device(host_context, runtime.device)
        with torch.no_grad():
            outputs = runtime.policy.forward_tensordict(policy_context_tensordict(context))
            output = runtime.policy.output_from_tensordict(outputs)
        alpha = output.distribution.parameters.alpha
        beta = output.distribution.parameters.beta
        expanded = AffineBeta(alpha.expand(sample_count, -1), beta.expand(sample_count, -1))
        generator = runtime.new_policy_generator(diagnostic_seed + index)
        samples = expanded.sample(generator).base_action
        boundary = (samples <= boundary_distance) | (samples >= 1.0 - boundary_distance)
        alpha_values.append(_tensor_pair(alpha[0]))
        beta_values.append(_tensor_pair(beta[0]))
        means.append(_tensor_pair(expanded.mean[0]))
        masses.append(_tensor_pair(boundary.float().mean(dim=0)))
    return PolicyProbeSummary(
        alpha=tuple(alpha_values),
        beta=tuple(beta_values),
        guidance_mean=tuple(means),
        boundary_mass=tuple(masses),
    )


def _context_to_device(
    context: ExplorationPolicyContext, device: torch.device
) -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=context.scene_tokens.to(device),
        scene_padding_mask=context.scene_padding_mask.to(device),
        navigation_tokens=context.navigation_tokens.to(device),
        navigation_padding_mask=context.navigation_padding_mask.to(device),
        reference_trajectory=context.reference_trajectory.to(device),
    )


def _tensor_pair(value: torch.Tensor) -> tuple[float, float]:
    if tuple(value.shape) != (2,):
        raise ValueError("policy probe statistic must have shape [2]")
    host = value.detach().cpu()
    return float(host[0]), float(host[1])
