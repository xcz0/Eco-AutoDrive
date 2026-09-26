"""Base/fixed/manual decision semantics over the real sampler and guidance code."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from lightning.fabric import Fabric

from eco_planner.evaluation.inference import DiffusionEvaluationAgent
from eco_planner.planning import DiffusionRuntime
from eco_planner.planning.diffusion import (
    Dpm10SamplerConfig,
    NoGuidanceConfig,
    OrthogonalReferenceGuidanceConfig,
    sampler_report,
)
from tests.characterization.harness import build_observation, build_planner, sampler_config

CASES = ("base_dpm", "base_ddim", "base_stochastic", "fixed_zero", "fixed", "manual")


def build_diffusion_case(case):
    planner = build_planner()
    sampler = (
        Dpm10SamplerConfig()
        if case == "base_dpm"
        else replace(
            sampler_config(),
            initial_noise_scale=1.0,
            parity_label="plannerrft_paper_text",
            ddim_stochasticity=0.5 if case in ("base_stochastic", "fixed", "manual") else 0.0,
        )
    )
    if case.startswith("base"):
        guidance = NoGuidanceConfig()
    elif case == "manual":
        guidance = planner.guidance_config
    else:
        fields = asdict(planner.guidance_config)
        fields["name"] = "orthogonal_reference"
        guidance = OrthogonalReferenceGuidanceConfig(
            **fields,
            lateral_scale=0.0 if case == "fixed_zero" else 0.2,
            longitudinal_scale=0.0 if case == "fixed_zero" else -0.3,
        )
    planner = type(planner)(planner.config, planner.model, sampler, guidance)
    planner.config.route_num = 25
    runtime = DiffusionRuntime(
        Fabric(accelerator="cpu", devices=1, precision="32-true"),
        planner,
        planner.config,
        None,
        SimpleNamespace(seed=101),
        sampler_report(sampler),
        guidance,
    )
    observation = build_observation().expand(2).clone()
    observation["route_lanes_speed_limit"] = torch.full((2, 25, 1), 13.0)
    observation["route_lanes_has_speed_limit"] = torch.ones((2, 25, 1), dtype=torch.bool)
    generators = tuple(torch.Generator().manual_seed(seed) for seed in (101, 303))
    action = torch.tensor([[-1.0, 1.0], [1.0, -1.0]]) if case == "manual" else None
    return runtime, observation, generators, action


def run_diffusion_case(case, *, profile=False):
    runtime, observation, generators, action = build_diffusion_case(case)
    snapshot = {"rng_before": torch.stack([g.get_state() for g in generators])}
    global_before = torch.get_rng_state().clone()
    noise = runtime.sample_noise(generators)
    snapshot["rng_after_initial"] = torch.stack([g.get_state() for g in generators])
    decision = DiffusionEvaluationAgent(runtime).infer_batch(
        observation,
        noise,
        generators,
        profile=profile,
        guidance_action=action,
    )
    snapshot.update(dict(decision.audit_result()))
    snapshot["execution"] = torch.from_numpy(decision.ego_trajectories)
    snapshot["rng_after"] = torch.stack([g.get_state() for g in generators])
    assert torch.equal(global_before, torch.get_rng_state())
    assert (decision.timing is not None) == profile
    if runtime.sampler_report.ddim_stochasticity == 0:
        assert torch.equal(snapshot["rng_after_initial"], snapshot["rng_after"])
    else:
        assert not torch.equal(snapshot["rng_after_initial"], snapshot["rng_after"])
    if case == "fixed_zero":
        assert torch.equal(snapshot["prediction"], snapshot["reference_prediction"])
    return snapshot


@pytest.mark.parametrize("case", CASES)
def test_profiling_preserves_decision_and_rng(case):
    expected = run_diffusion_case(case)
    actual = run_diffusion_case(case, profile=True)
    assert actual.keys() == expected.keys()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("case", ["fixed_zero", "fixed", "manual"])
def test_guidance_shares_base_reference_and_transition_draws(case):
    base = run_diffusion_case("base_ddim" if case == "fixed_zero" else "base_stochastic")
    guided = run_diffusion_case(case)
    torch.testing.assert_close(guided["reference_prediction"], base["prediction"])
    assert torch.equal(guided["initial_noise"], base["initial_noise"])
    assert torch.equal(guided["rng_after"], base["rng_after"])
    assert "reference_prediction" not in base
    if case != "fixed_zero":
        assert not torch.equal(guided["prediction"], guided["reference_prediction"])
        assert torch.any(guided["applied_gradient_l2"] > 0)
    if case == "manual":
        assert torch.equal(guided["guidance_action"], torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))


@pytest.mark.parametrize("case", CASES)
def test_mode_grad_context_and_pass_count(case, monkeypatch):
    runtime, observation, generators, action = build_diffusion_case(case)
    planner = runtime._planner
    calls = []
    denoise = planner.model.denoise

    def record(*args, **kwargs):
        calls.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return denoise(*args, **kwargs)

    monkeypatch.setattr(planner.model, "denoise", record)
    result = runtime.decide_batch(
        observation, runtime.sample_noise(generators), generators, guidance_action=action
    )
    if case.startswith("base"):
        assert all(inference and not grad for grad, inference in calls)
        assert len(calls) == (11 if case == "base_dpm" else 5)
        assert result.planner.reference_prediction is None
    elif case == "fixed_zero":
        assert len(calls) == 5
        assert result.planner.prediction is result.planner.reference_prediction
    else:
        assert any(grad and not inference for grad, inference in calls)
        assert len(calls) == 10
    assert all(not p.requires_grad and p.grad is None for p in planner.parameters())


def test_explicit_noise_zero_stochasticity_needs_no_generator():
    runtime, observation, generators, _ = build_diffusion_case("base_ddim")
    noise = runtime.sample_noise(generators)
    before = torch.get_rng_state().clone()
    result = runtime.decide_batch(observation, noise, (None, None))
    assert result.initial_noise is noise
    assert torch.equal(before, torch.get_rng_state())


def test_disabled_profiling_does_not_sync_or_create_events(monkeypatch):
    from eco_planner.planning.diffusion_inference import synchronize_if_cuda

    def unexpected(*args, **kwargs):
        pytest.fail("disabled profiling must not synchronize or create CUDA events")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    monkeypatch.setattr(torch.cuda, "Event", unexpected)
    synchronize_if_cuda(torch.device("cuda"), False)
    run_diffusion_case("base_stochastic")
