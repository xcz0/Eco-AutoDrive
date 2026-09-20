"""Characterization tests for the planning-owned learned-guidance decision.

These pin that ``PlanningInference`` alone reproduces the same decision tensors and
RNG consumption as the golden snapshot recorded from the rollout runtime, and that
deterministic mean mode still leaves the policy generator untouched.
"""

from __future__ import annotations

import torch

from tests.characterization.harness import (
    AUDIT_FIELDS,
    RNG_FIELDS,
    load_reference_fixture,
    planning_decision_snapshot,
    run_planning_decision,
)


def test_planning_inference_matches_characterization_snapshot() -> None:
    expected = load_reference_fixture()
    result, rng = run_planning_decision(sample=True)
    actual = planning_decision_snapshot(result, rng)

    assert set(actual) == set(expected) == {*AUDIT_FIELDS, "ego_trajectory", *RNG_FIELDS}
    for name, reference in expected.items():
        torch.testing.assert_close(actual[name], reference, rtol=0.0, atol=0.0)


def test_planning_inference_is_reproducible_for_fixed_generators() -> None:
    first, first_rng = run_planning_decision(sample=True)
    second, second_rng = run_planning_decision(sample=True)

    torch.testing.assert_close(
        planning_decision_snapshot(first, first_rng),
        planning_decision_snapshot(second, second_rng),
        rtol=0.0,
        atol=0.0,
    )


def test_planning_inference_mean_decision_does_not_consume_policy_rng() -> None:
    result, rng = run_planning_decision(sample=False)

    assert torch.equal(rng.policy_before, rng.policy_after)
    assert not torch.equal(rng.diffusion_before, rng.diffusion_after)
    second, second_rng = run_planning_decision(sample=False)
    torch.testing.assert_close(
        planning_decision_snapshot(result, rng),
        planning_decision_snapshot(second, second_rng),
        rtol=0.0,
        atol=0.0,
    )
