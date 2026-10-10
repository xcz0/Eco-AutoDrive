"""Characterization tests for one learned-guidance planner decision.

These tests pin the observable semantics that the planning-owned learned-guidance
decision must preserve while planner execution moves out of the RL and evaluation
runtimes. Historical fixtures pin the schema and RNG boundaries; floating-point
comparisons use decisions produced under the same current execution conditions.
"""

from __future__ import annotations

import torch

from tests.characterization.harness import (
    AUDIT_FIELDS,
    RNG_FIELDS,
    decision_snapshot,
    load_reference_fixture,
    run_decision,
)


def test_sampled_decision_preserves_audit_schema_and_rng_boundaries() -> None:
    expected = load_reference_fixture()
    decision, rng = run_decision(sample=True)
    actual = decision_snapshot(decision, rng)

    assert set(actual) == set(expected) == {*AUDIT_FIELDS, "ego_trajectory", *RNG_FIELDS}
    for name, reference in expected.items():
        assert actual[name].shape == reference.shape
        assert actual[name].dtype == reference.dtype
        if actual[name].is_floating_point():
            assert torch.isfinite(actual[name]).all()
        else:
            assert torch.equal(actual[name], reference)


def test_sampled_decision_is_reproducible_for_fixed_generators() -> None:
    first, first_rng = run_decision(sample=True)
    second, second_rng = run_decision(sample=True)

    torch.testing.assert_close(
        decision_snapshot(first, first_rng),
        decision_snapshot(second, second_rng),
        rtol=0.0,
        atol=0.0,
    )


def test_sampled_decision_consumes_both_generator_streams() -> None:
    decision, rng = run_decision(sample=True)
    audit = decision.audit_result()

    assert not torch.equal(rng.diffusion_before, rng.diffusion_after)
    assert not torch.equal(rng.policy_before, rng.policy_after)
    assert not torch.equal(audit["reference_prediction"], audit["prediction"])
    assert torch.count_nonzero(audit["guidance_action"]).item() > 0


def test_deterministic_mean_decision_does_not_consume_policy_rng() -> None:
    decision, rng = run_decision(sample=False)

    assert torch.equal(rng.policy_before, rng.policy_after)
    assert not torch.equal(rng.diffusion_before, rng.diffusion_after)
    second, second_rng = run_decision(sample=False)
    torch.testing.assert_close(
        decision.ego_trajectories,
        second.ego_trajectories,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(rng.policy_after, second_rng.policy_after)


def test_mean_decision_shares_the_forward_path_and_only_changes_action_sampling() -> None:
    sampled, sampled_rng = run_decision(sample=True)
    expected = decision_snapshot(sampled, sampled_rng)
    decision, _ = run_decision(sample=False)
    audit = decision.audit_result()

    for name in (
        "reference_prediction",
        "initial_noise",
        "scene_tokens",
        "navigation_tokens",
        "beta_alpha",
        "beta_beta",
        "state_value",
    ):
        torch.testing.assert_close(audit[name], expected[name], rtol=0.0, atol=0.0)
    assert not torch.equal(audit["guidance_action"], expected["guidance_action"])
