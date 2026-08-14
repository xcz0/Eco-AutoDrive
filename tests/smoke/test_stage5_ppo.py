from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from eco_planner.rl import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
    MetaDriveRolloutReward,
    MetaDriveTransitionAudit,
    PPOUpdater,
    RolloutBuffer,
    RolloutTransition,
    parse_ppo_optimization_config,
)


def test_stage5_smoke_profile_completes_one_torchrl_ppo_update() -> None:
    root = Path(__file__).parents[2]
    config = parse_ppo_optimization_config(
        OmegaConf.load(root / "configs" / "train" / "ppo_stage5_smoke.yaml")
    )
    policy = ExplorationPolicy(
        ExplorationPolicyConfig(
            name="exploration_beta",
            hidden_dim=12,
            reference_horizon=80,
            reference_state_dim=4,
            reference_mixer_depth=1,
            reference_token_mlp_hidden_dim=16,
            reference_channel_mlp_hidden_dim=24,
            cross_attention_heads=3,
            cross_attention_dropout=0.0,
            fusion_mlp_depth=1,
            fusion_hidden_dim=16,
            initial_concentration=2.0,
            minimum_concentration=1e-4,
        )
    )
    buffer = RolloutBuffer()
    for index, reward in enumerate((0.1, 0.2, 0.4, 0.8)):
        context = _context(float(index + 1))
        base_action = torch.tensor([[0.25 + index * 0.05, 0.7 - index * 0.05]])
        with torch.no_grad():
            output = policy(context)
            action = output.distribution.evaluate_base_action(base_action)
        buffer.append(
            RolloutTransition(
                policy_context=context,
                base_action=base_action,
                guidance_action=action.guidance_action,
                old_joint_guidance_log_prob=action.joint_guidance_log_prob.detach(),
                old_value=output.value.detach(),
                beta_alpha=output.parameters.alpha.detach(),
                beta_beta=output.parameters.beta.detach(),
                initial_noise=torch.zeros((1, 11, 80, 4)),
                diffusion_rng_state=torch.Generator().manual_seed(10).get_state(),
                policy_rng_state=torch.Generator().manual_seed(20).get_state(),
                reward=MetaDriveRolloutReward(
                    substep_scores=torch.tensor([reward]),
                    total_score=torch.tensor([reward]),
                    dense_step_scores=torch.tensor([reward]),
                    terminal_override_deltas=torch.tensor([0.0]),
                ),
                audit=MetaDriveTransitionAudit(
                    route_completion_delta=0.01,
                    distance_m=1.0,
                    speed_mps=10.0,
                    stopped=False,
                    position_error_m=0.0,
                    heading_error_rad=0.0,
                    arrive_dest=False,
                    out_of_road=False,
                    crash_vehicle=False,
                    crash_object=False,
                    crash_building=False,
                    crash_human=False,
                ),
                terminated=False,
                truncated=False,
                bootstrap_mask=True,
                scenario_name="synthetic",
                map_sequence="S",
                map_seed=1,
                noise_seed=2,
                policy_action_seed=3,
                planning_cycle_index=index,
                executed_substep_count=1,
            )
        )
    with torch.no_grad():
        tail_value = policy(_context(5.0)).value.detach()
    report = PPOUpdater(policy, config).update([buffer.finalize("rollout_limit", tail_value)])
    assert report.sample_count == 4
    assert report.optimizer_step_count == 4
    assert report.final_learning_rate == 0.0


def _context(scale: float) -> ExplorationPolicyContext:
    reference = torch.zeros((1, 80, 4))
    reference[..., 0] = torch.arange(1, 81) * scale * 0.1
    reference[..., 2] = 1.0
    return ExplorationPolicyContext(
        scene_tokens=torch.full((1, 4, 12), scale),
        scene_padding_mask=torch.tensor([[False, False, True, True]]),
        navigation_tokens=torch.full((1, 1, 12), scale),
        navigation_padding_mask=torch.tensor([[False]]),
        reference_trajectory=reference,
    )
