"""Policy-checkpoint evaluation agent mean/sample action-mode routing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensordict import TensorDict

import eco_planner.evaluation.inference.agent as agent_module
from eco_planner.evaluation.artifacts.models import PolicyCheckpointProvenance
from eco_planner.evaluation.inference.agent import PolicyCheckpointEvaluationAgent


@pytest.fixture(autouse=True)
def _identity_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_module,
        "prepare_learned_inference_decision",
        lambda result, host_transfer: result,
    )


def _provenance() -> PolicyCheckpointProvenance:
    return PolicyCheckpointProvenance(label="final", path="policy-final.pt", policy_hash="a" * 64)


def _runtime() -> tuple[SimpleNamespace, dict[str, Any]]:
    calls: dict[str, Any] = {}

    def decide_batch_mean(observation, generators):
        calls["mean"] = (observation, generators)
        return "mean-decision"

    def decide_batch(observation, generators, policy_generators):
        calls["sample"] = (observation, generators, policy_generators)
        return "sample-decision"

    def new_policy_generator(seed):
        return torch.Generator().manual_seed(seed)

    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        decide_batch_mean=decide_batch_mean,
        decide_batch=decide_batch,
        new_policy_generator=new_policy_generator,
    )
    return runtime, calls


def _observation() -> TensorDict:
    return TensorDict({"ego_current_state": torch.zeros(1, 7)}, batch_size=[1])


def test_mean_mode_routes_to_decide_batch_mean_without_policy_rng() -> None:
    runtime, calls = _runtime()
    agent = PolicyCheckpointEvaluationAgent(
        runtime=runtime,
        noise_seeds=(760025,),
        policy_checkpoint=_provenance(),
        action_mode="mean",
        policy_action_seeds=(),
    )
    generator = torch.Generator().manual_seed(0)

    assert agent.new_policy_generator(0) is None
    assert agent.decide_batch(_observation(), (generator,)) == "mean-decision"
    assert "sample" not in calls
    with pytest.raises(ValueError, match="policy generators"):
        agent.decide_batch(_observation(), (generator,), policy_generators=(generator,))


def test_sample_mode_seeds_one_policy_generator_per_scenario() -> None:
    runtime, calls = _runtime()
    agent = PolicyCheckpointEvaluationAgent(
        runtime=runtime,
        noise_seeds=(760025, 760025),
        policy_checkpoint=_provenance(),
        action_mode="sample",
        policy_action_seeds=(810001, 810001),
    )
    generator = torch.Generator().manual_seed(0)

    policy_generator = agent.new_policy_generator(1)
    assert isinstance(policy_generator, torch.Generator)
    assert (
        agent.decide_batch(_observation(), (generator,), policy_generators=(policy_generator,))
        == "sample-decision"
    )
    assert calls["sample"][2] == (policy_generator,)
    with pytest.raises(ValueError, match="requires policy generators"):
        agent.decide_batch(_observation(), (generator,))


def test_agent_rejects_mode_seed_mismatches() -> None:
    runtime, _ = _runtime()
    with pytest.raises(ValueError, match="one policy action seed per scenario"):
        PolicyCheckpointEvaluationAgent(
            runtime=runtime,
            noise_seeds=(760025,),
            policy_checkpoint=_provenance(),
            action_mode="sample",
            policy_action_seeds=(),
        )
    with pytest.raises(ValueError, match="must not configure policy action seeds"):
        PolicyCheckpointEvaluationAgent(
            runtime=runtime,
            noise_seeds=(760025,),
            policy_checkpoint=_provenance(),
            action_mode="mean",
            policy_action_seeds=(810001,),
        )
    with pytest.raises(ValueError, match="align with the noise seed"):
        PolicyCheckpointEvaluationAgent(
            runtime=runtime,
            noise_seeds=(760025, 760025),
            policy_checkpoint=_provenance(),
            action_mode="sample",
            policy_action_seeds=(810001,),
        )
