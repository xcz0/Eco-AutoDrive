"""Characterization tests for the planning-owned learned-guidance decision.

These pin that ``PlanningInference`` alone reproduces the same decision tensors and
RNG consumption as the rollout runtime under the same execution conditions, and that
deterministic mean mode still leaves the policy generator untouched.
"""

from __future__ import annotations

import torch

from eco_planner.evaluation.inference.decision import prepare_learned_inference_decision
from eco_planner.runtime.host_transfer import HostTransfer
from tests.characterization.harness import (
    AUDIT_FIELDS,
    RNG_FIELDS,
    decision_snapshot,
    planning_decision_snapshot,
    run_decision,
    run_planning_decision,
)

_EXPECTED_AUDIT_KEYS = frozenset(
    {
        "initial_noise",
        "prediction",
        "reference_prediction",
        "guidance_action",
        "lateral_target_offset_m",
        "longitudinal_target_speed_fraction",
        "longitudinal_target_speed_delta_mps",
        "lateral_objective_delta",
        "longitudinal_objective_delta",
        "applied_gradient_l2",
        "applied_gradient_max_abs",
        "raw_neighbor_gradient_l2",
        "zero_speed_count",
    }
)


def test_planning_inference_matches_rollout_decision() -> None:
    decision, rollout_rng = run_decision(sample=True)
    expected = decision_snapshot(decision, rollout_rng)
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


def test_evaluation_adapter_reuses_planning_decision_tensors() -> None:
    result, _ = run_planning_decision(sample=True)
    decision = prepare_learned_inference_decision(result, HostTransfer(torch.device("cpu")))
    audit = decision.audit_result()

    assert set(audit.keys()) == _EXPECTED_AUDIT_KEYS
    torch.testing.assert_close(
        torch.from_numpy(decision.ego_trajectories), result.prediction[:, 0].detach().cpu()
    )
    torch.testing.assert_close(audit["prediction"], result.prediction.detach().cpu())
    torch.testing.assert_close(audit["initial_noise"], result.initial_noise.detach().cpu())
    assert result.reference_prediction is not None and result.policy is not None
    torch.testing.assert_close(
        audit["reference_prediction"], result.reference_prediction.detach().cpu()
    )
    torch.testing.assert_close(
        audit["guidance_action"], result.policy.action.guidance_action.detach().cpu()
    )
    assert result.guidance_diagnostics is not None
    torch.testing.assert_close(
        audit["applied_gradient_l2"],
        result.guidance_diagnostics.applied_gradient_l2.detach().cpu(),
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
