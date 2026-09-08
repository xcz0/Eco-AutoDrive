from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.policy.distribution import AffineBeta
from eco_planner.rl.probing import capture_probe_contexts, probe_policy
from tests.training.test_ppo import _context, _episode, _policy_config


def test_capture_uses_first_transition_of_first_episode_in_scenario_order():
    episodes = tuple(
        _episode(reward=1.0, terminated=True, truncated=False, bootstrap=0.0) for _ in range(3)
    )
    for index, episode in enumerate(episodes):
        episode.training["reference_trajectory"].fill_(index)
    contexts = capture_probe_contexts(((episodes[0], episodes[1]), (episodes[2],)), 2)
    assert len(contexts) == 2
    for context, index in zip(contexts, (0, 2), strict=True):
        assert context.reference_trajectory.shape == (1, 80, 4)
        assert torch.equal(
            context.reference_trajectory, episodes[index].training["reference_trajectory"]
        )
    with pytest.raises(RuntimeError, match="one fixed probe context per scenario"):
        capture_probe_contexts(((episodes[0],), ()), 2)


def test_probe_preserves_formula_scenario_order_and_training_random_streams():
    with torch.random.fork_rng():
        torch.manual_seed(11)
        policy = ExplorationPolicy(_policy_config()).eval()
        # Break symmetric initialization so different contexts exercise output ordering.
        torch.nn.init.normal_(policy.actor_head.weight, std=0.1)
        contexts = (_context(), replace(_context(), reference_trajectory=torch.ones(1, 80, 4)))
        noise_generator = torch.Generator().manual_seed(101)
        action_generator = torch.Generator().manual_seed(102)
        noise_before = noise_generator.get_state().clone()
        action_before = action_generator.get_state().clone()
        seeds = []

        def new_generator(seed):
            seeds.append(seed)
            return torch.Generator().manual_seed(seed)

        runtime = SimpleNamespace(
            policy=policy, device=torch.device("cpu"), new_policy_generator=new_generator
        )
        global_before = torch.random.get_rng_state().clone()
        actual = probe_policy(runtime, contexts, 128, 0.1, 40)
        assert seeds == [40, 41]
        assert probe_policy(runtime, contexts, 128, 0.1, 40) == actual
        assert torch.equal(torch.random.get_rng_state(), global_before)
        assert torch.equal(noise_generator.get_state(), noise_before)
        assert torch.equal(action_generator.get_state(), action_before)
        assert actual.alpha[0] != actual.alpha[1]
        for index, context in enumerate(contexts):
            with torch.no_grad():
                distribution = policy(context).distribution
            alpha = distribution.parameters.alpha
            beta = distribution.parameters.beta
            samples = (
                AffineBeta(alpha.expand(128, -1), beta.expand(128, -1))
                .sample(torch.Generator().manual_seed(40 + index))
                .base_action
            )
            boundary = ((samples <= 0.1) | (samples >= 0.9)).float().mean(dim=0)
            assert actual.alpha[index] == tuple(alpha[0].tolist())
            assert actual.beta[index] == tuple(beta[0].tolist())
            assert actual.guidance_mean[index] == tuple(distribution.mean[0].tolist())
            assert actual.boundary_mass[index] == tuple(boundary.tolist())
