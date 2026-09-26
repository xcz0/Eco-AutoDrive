"""Planning execution and evaluation transfer ownership at the public boundary."""

import pytest
import torch

import eco_planner.evaluation.inference.agent as agent_module
from eco_planner.evaluation.inference import DiffusionEvaluationAgent
from eco_planner.runtime.host_transfer import HostTransfer
from tests.characterization.test_diffusion_inference import build_diffusion_case


def test_execution_is_ready_before_deferred_audit_and_keeps_decision_identity(monkeypatch):
    runtime, observation, generators, _ = build_diffusion_case("base_ddim")
    events = []

    class TrackingTransfer(HostTransfer):
        def defer(self, tensors, *, profile=False):
            events.append("defer")
            result = super().defer(tensors, profile=profile)
            resolve = result.resolve

            def recorded_resolve():
                events.append("resolve")
                return resolve()

            monkeypatch.setattr(result, "resolve", recorded_resolve)
            return result

        def execution_trajectories(self, prediction):
            events.append("execution")
            return super().execution_trajectories(prediction)

    monkeypatch.setattr(agent_module, "HostTransfer", TrackingTransfer)
    agent = DiffusionEvaluationAgent(runtime)
    first_noise = runtime.sample_noise(generators)
    first = agent.infer_batch(observation, first_noise, generators)
    first_execution = first.ego_trajectories.copy()
    assert events == ["defer", "execution"]
    # Simulate advancing a workflow before resolving the previous cycle's audit.
    second_noise = runtime.sample_noise(generators)
    second = agent.infer_batch(observation, second_noise, generators)
    assert events == ["defer", "execution", "defer", "execution"]
    second_audit = second.audit_result()
    first_audit = first.audit_result()
    assert first.audit_result() is first_audit
    assert events.count("resolve") == 2
    assert torch.equal(first_audit["initial_noise"], first_noise)
    assert torch.equal(second_audit["initial_noise"], second_noise)
    assert not torch.equal(first_noise, second_noise)
    torch.testing.assert_close(torch.from_numpy(first_execution), first_audit["prediction"][:, 0])


def test_raw_artifact_fields_are_validated_by_evaluation_not_planning():
    runtime, observation, generators, _ = build_diffusion_case("base_ddim")
    observation.del_("route_lanes_speed_limit")
    noise = runtime.sample_noise(generators)
    runtime.decide_batch(observation, noise, generators)
    with pytest.raises(ValueError, match="route_lanes_speed_limit"):
        DiffusionEvaluationAgent(runtime).infer_batch(observation, noise, generators)


@pytest.mark.parametrize("case", ["base_ddim", "fixed"])
def test_agent_regular_decision_matches_explicit_noise_path(case):
    runtime, observation, generators, _ = build_diffusion_case(case)
    agent = DiffusionEvaluationAgent(runtime)
    states = [generator.get_state() for generator in generators]
    explicit = agent.infer_batch(observation, runtime.sample_noise(generators), generators)
    for generator, state in zip(generators, states, strict=True):
        generator.set_state(state)
    regular = agent.decide_batch(observation, generators)
    torch.testing.assert_close(dict(explicit.audit_result()), dict(regular.audit_result()))
