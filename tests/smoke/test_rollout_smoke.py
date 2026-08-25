from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from eco_planner.models import Ddim5SamplerConfig, OrthogonalPolicyGuidanceConfig
from eco_planner.rl.collector import collect_rollout_episode
from eco_planner.rl.config import ExplorationPolicyConfig, parse_rollout_config
from eco_planner.rl.runtime import create_fabric_rollout_runtime
from eco_planner.runtime.config import RuntimeConfig


@pytest.mark.slow
def test_real_checkpoint_policy_guided_decision_is_finite_and_replayable(
    baseline_checkpoint_dir, baseline_observation: dict[str, torch.Tensor]
) -> None:
    def runtime():
        return create_fabric_rollout_runtime(
            RuntimeConfig(accelerator="cpu", precision="32-true", seed=0),
            Ddim5SamplerConfig(
                name="ddim5",
                num_steps=5,
                timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
                initial_noise_scale=1.0,
                ddim_stochasticity=0.0,
                parity_label="plannerrft_paper_text",
            ),
            OrthogonalPolicyGuidanceConfig(
                name="orthogonal_policy",
                formula_label="centered_energy_gradient_delta_v1",
                lateral_max_offset_m=2.5,
                longitudinal_max_speed_fraction=0.25,
                trajectory_dt_s=0.1,
                gradient_step_coefficient=1.0,
                reference_refresh_cycles=1,
                share_scene_encoding=True,
                share_initial_noise=True,
                share_transition_noise=True,
                heading_norm_epsilon=1e-6,
                zero_speed_tolerance_mps=1e-6,
            ),
            ExplorationPolicyConfig(
                hidden_dim=192,
                reference_mixer_depth=2,
                reference_token_mlp_hidden_dim=128,
                reference_channel_mlp_hidden_dim=384,
                cross_attention_heads=6,
                cross_attention_dropout=0.0,
                fusion_mlp_depth=2,
                fusion_hidden_dim=256,
                initial_concentration=2.0,
                minimum_concentration=1e-4,
            ),
            baseline_checkpoint_dir / "args.json",
            baseline_checkpoint_dir / "model.pth",
            policy_action_seed=17,
        )

    first_runtime = runtime()
    global_state = torch.random.get_rng_state().clone()
    first = first_runtime.decide(
        baseline_observation,
        first_runtime.new_noise_generator(),
        first_runtime.new_policy_generator(),
    )
    replay_runtime = runtime()
    replay = replay_runtime.decide(
        baseline_observation,
        replay_runtime.new_noise_generator(),
        replay_runtime.new_policy_generator(),
    )
    first_audit = first.audit_result()
    replay_audit = replay.audit_result()

    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert first_audit.prediction.shape == (1, 11, 80, 4)
    assert first_audit.prediction.dtype.name == "float32"
    assert torch.equal(first_audit.initial_noise, replay_audit.initial_noise)
    assert torch.equal(first_audit.base_action, replay_audit.base_action)
    assert torch.equal(first_audit.guidance_action, replay_audit.guidance_action)
    assert torch.isfinite(first_audit.old_joint_guidance_log_prob).all()
    assert torch.isfinite(first_audit.old_value).all()


@pytest.mark.simulator
@pytest.mark.slow
def test_collects_one_real_10hz_transition_without_artifacts(baseline_checkpoint_dir) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="jobs/training/rollout_smoke")
    parsed = parse_rollout_config(config)
    runtime = create_fabric_rollout_runtime(
        parsed.runtime,
        parsed.sampler,
        parsed.guidance,
        parsed.policy,
        baseline_checkpoint_dir / "args.json",
        baseline_checkpoint_dir / "model.pth",
        parsed.rollout.policy_action_seed,
    )

    episode = collect_rollout_episode(
        parsed.scenario,
        runtime,
        parsed.env,
        mode=parsed.rollout.mode,
        map_query_radius_m=parsed.map_query_radius_m,
        history_warmup_steps=parsed.rollout.history_warmup_steps,
        max_transitions=1,
        stopped_speed_threshold_mps=parsed.rollout.stopped_speed_threshold_mps,
    )

    assert episode.transition_count == 1
    assert episode.training["guidance_action"].shape == (1, 2)
    assert episode.tail_kind == "rollout_limit"
    assert episode.training["reward"].shape == (1, 1)
