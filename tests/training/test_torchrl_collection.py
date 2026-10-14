"""Compare the synchronous Collector with explicit domain-owned episode semantics."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict, cat

from eco_planner.configuration import ScenarioConfig
from eco_planner.envs.parallel import (
    VectorEnvScenario,
    VectorEnvTiming,
    WorkerResetResult,
    WorkerStepResult,
    operation_results,
)
from eco_planner.envs.parallel.torchrl import planner_observation_spec
from eco_planner.rl.rollout.collector import VectorRolloutCollector, _EpisodeLifecycle
from eco_planner.rl.rollout.decision import RolloutDecision
from eco_planner.rl.rollout.profiling import RolloutPlannerTiming
from eco_planner.runtime.contracts import HostTrajectories
from eco_planner.runtime.profiling import PhaseTiming
from tests.training.test_rollout import (
    _execution_result,
    _reward_result,
    _substep_metrics,
    _transition,
)


class _Environment:
    def __init__(self, capacity):
        self.num_envs = capacity
        self.scenarios = [None] * capacity
        self.counts = [0] * capacity
        self.events = []

    def _observation(self, slots):
        observation = planner_observation_spec().expand(len(slots)).zero()["observation"]
        for index, slot in enumerate(slots):
            observation["ego_current_state"][index, 0] = (
                self.scenarios[slot].seed * 10 + self.counts[slot]
            )
        return observation

    def reset(self, scenarios, *, slots=None):
        slots = tuple(range(self.num_envs)) if slots is None else tuple(slots)
        self.events.append(("reset", slots, tuple(scenario.seed for scenario in scenarios)))
        results = []
        for slot, scenario in zip(slots, scenarios, strict=True):
            self.scenarios[slot] = scenario
            self.counts[slot] = 0
            results.append(
                WorkerResetResult(
                    scenario,
                    0.0,
                    100.0,
                    np.zeros(7),
                    np.zeros(7),
                    (),
                    None,
                    {},
                    VectorEnvTiming(0.0, 0.0),
                )
            )
        return TensorDict(
            {"observation": self._observation(slots)}, batch_size=[len(slots)]
        ).set_non_tensor("operation_results", tuple(results))

    def step(self, trajectories, *, slots=None):
        slots = tuple(range(self.num_envs)) if slots is None else tuple(slots)
        self.events.append(("step", slots, tuple(trajectories[:, 0, 0].tolist())))
        results, terminated, truncated = [], [], []
        for slot in slots:
            self.counts[slot] += 1
            role = self.scenarios[slot].seed
            terminal = (role == 0 and self.counts[slot] == 1) or (
                role == 2 and self.counts[slot] == 2
            )
            truncation = (role == 0 and self.counts[slot] == 1) or (
                role == 1 and self.counts[slot] == 2
            )
            step = _execution_result(
                tuple(_substep_metrics(i) for i in range(role + 1)),
                route_completion=0.1 * self.counts[slot],
            )
            results.append(
                WorkerStepResult(
                    replace(step, terminated=terminal, truncated=truncation),
                    None,
                    VectorEnvTiming(0.0, 0.0),
                )
            )
            terminated.append(terminal)
            truncated.append(truncation)
        terminal = torch.tensor(terminated).unsqueeze(-1)
        truncation = torch.tensor(truncated).unsqueeze(-1)
        return TensorDict(
            {
                "observation": self._observation(slots),
                "reward": torch.zeros(len(slots), 1),
                "terminated": terminal,
                "truncated": truncation,
                "done": terminal | truncation,
            },
            batch_size=[len(slots)],
        ).set_non_tensor("operation_results", tuple(results))


class _Runtime:
    def __init__(self, device, events):
        self.device = torch.device(device)
        self.events = events

    def decide_batch(self, observation, diffusion, policy=None, **kwargs):
        self._timing("decision", kwargs.get("timings"))
        markers = observation["ego_current_state"][:, 0].tolist()
        self.events.append(("plan", tuple(markers)))
        fields = []
        for index, generator in enumerate(diffusion):
            training, audit, _, _ = _transition()
            noise = torch.rand((), generator=generator, device=self.device)
            action = (
                torch.rand(2, generator=policy[index], device=self.device) * 1.5 - 0.75
                if policy is not None
                else torch.zeros(2, device=self.device)
            )
            training = training.to(self.device)
            training["state_value"].fill_(markers[index] + float(noise))
            training["guidance_action"][0].copy_(action)
            audit["state_value"].copy_(training["state_value"].cpu())
            audit["guidance_action"][0].copy_(action.cpu())
            audit["base_action"][0].copy_((action.cpu() + 1.0) / 2.0)
            audit["initial_noise"].fill_(float(noise))
            fields.append((training, audit))
        return SimpleNamespace(
            training_decision=cat([training for training, _ in fields]),
            audit_transfer_timing=None,
            audit_result=lambda: cat([audit for _, audit in fields]),
            ego_trajectories=np.broadcast_to(
                np.asarray(markers, dtype=np.float32)[:, None, None], (len(markers), 80, 4)
            ).copy(),
        )

    def decide_batch_mean(self, observation, diffusion, **kwargs):
        return self.decide_batch(observation, diffusion, **kwargs)

    def bootstrap_value_batch(self, observation, diffusion, **kwargs):
        self._timing("bootstrap", kwargs.get("timings"))
        markers = observation["ego_current_state"][:, 0].tolist()
        self.events.append(("bootstrap", tuple(markers)))
        return torch.stack(
            [
                torch.tensor(marker, device=self.device)
                + torch.rand((), generator=generator, device=self.device)
                for marker, generator in zip(markers, diffusion, strict=True)
            ]
        )

    @staticmethod
    def _timing(phase, timings):
        if timings is not None:
            zero = PhaseTiming(0.0, 0.0)
            timings.append(
                RolloutPlannerTiming(
                    phase,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero if phase == "decision" else None,
                    zero if phase == "decision" else None,
                    zero if phase == "decision" else None,
                    0.0,
                )
            )


def _reference(collector, diffusion, policy, *, sampling, quota):
    """Independent explicit loop: bootstrap the final state before partial reset."""
    groups = []
    for start in range(0, len(collector._specs), collector._physical_slot_count):
        specs = collector._specs[start : start + collector._physical_slot_count]
        scenarios = collector._scenarios[start : start + len(specs)]
        slots = tuple(range(len(specs)))
        resets = collector._envs.reset(scenarios, slots=slots)
        observation = resets["observation"]
        lifecycles = [_EpisodeLifecycle(0.0) for _ in specs]
        episodes = [[] for _ in specs]
        noises = diffusion[start : start + len(specs)]
        actions = policy[start : start + len(specs)]
        for round_index in range(quota):
            decision = collector._runtime.decide_batch(
                observation, noises, actions if sampling == "sample" else None
            )
            steps = collector._envs.step(decision.ego_trajectories, slots=slots)
            tails = []
            for slot, step in enumerate(operation_results(steps, WorkerStepResult)):
                kind = lifecycles[slot].append(
                    RolloutDecision(
                        HostTrajectories(decision.ego_trajectories[slot : slot + 1]),
                        lambda slot=slot, decision=decision: decision.audit_result()[
                            slot : slot + 1
                        ],
                        decision.training_decision[slot : slot + 1],
                    ),
                    step.step,
                    map_seed=specs[slot].seed,
                    noise_seed=10 + start + slot,
                    policy_action_seed=20 + start + slot,
                    reward_evaluator=collector._reward_evaluator,
                    terminated=bool(steps["terminated"][slot]),
                    truncated=bool(steps["truncated"][slot]),
                    collection_limit=round_index + 1 == quota,
                )
                if kind is not None:
                    tails.append((slot, kind))
            bootstrap_slots = [slot for slot, kind in tails if kind != "terminated"]
            values = (
                collector._runtime.bootstrap_value_batch(
                    steps["observation"][bootstrap_slots],
                    tuple(noises[slot] for slot in bootstrap_slots),
                )
                if bootstrap_slots
                else []
            )
            bootstraps = {
                slot: values[index : index + 1] for index, slot in enumerate(bootstrap_slots)
            }
            for slot, kind in tails:
                episodes[slot].append(
                    lifecycles[slot].finish(
                        kind, torch.zeros(1) if kind == "terminated" else bootstraps[slot]
                    )
                )
            reset_slots = [slot for slot, _ in tails] if round_index + 1 < quota else []
            observation = steps["observation"].clone()
            if reset_slots:
                resets = collector._envs.reset(
                    tuple(scenarios[slot] for slot in reset_slots), slots=reset_slots
                )
                observation[reset_slots] = resets["observation"]
                for slot in reset_slots:
                    lifecycles[slot].reset(0.0)
        groups.extend(tuple(group) for group in episodes)
    return tuple(groups)


def _collector(capacity, device):
    collector = object.__new__(VectorRolloutCollector)
    collector._specs = tuple(
        ScenarioConfig(name=f"s{seed}", map="S", seed=seed) for seed in range(4)
    )
    collector._scenarios = tuple(
        VectorEnvScenario(spec.name, spec.map, spec.seed) for spec in collector._specs
    )
    collector._physical_slot_count = capacity
    collector._envs = _Environment(capacity)
    collector._runtime = _Runtime(device, collector._envs.events)
    collector._reward_evaluator = lambda metric: _reward_result(0.25 + metric.speed_mps * 0.01)
    return collector


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("capacity", [1, 3, 4])
@pytest.mark.parametrize("sampling", ["sample", "mean"])
@pytest.mark.parametrize("shared_storage", [False, True])
def test_synchronous_collection_matches_domain_reference(
    device, capacity, sampling, shared_storage
):
    actual_collector, reference_collector = (_collector(capacity, device) for _ in range(2))
    generators = [
        tuple(torch.Generator(device=device).manual_seed(seed + slot) for slot in range(4))
        for seed in (10, 20, 10, 20)
    ]
    diffusion, policy, reference_diffusion, reference_policy = generators
    global_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if device == "cuda" else []
    snapshots = []
    for _update in range(2):
        storage = TensorDict({}, batch_size=[12]) if shared_storage else None
        actual = actual_collector.collect(
            transitions_per_slot=3,
            diffusion_generators=diffusion,
            policy_generators=policy,
            noise_seeds=(10, 11, 12, 13),
            policy_action_seeds=(20, 21, 22, 23),
            policy_sampling=sampling,
            training_storage=storage,
        )
        expected = _reference(
            reference_collector, reference_diffusion, reference_policy, sampling=sampling, quota=3
        )
        assert actual_collector._envs.events == reference_collector._envs.events
        for group, reference_group in zip(actual, expected, strict=True):
            assert sum(episode.transition_count for episode in group) == 3
            for episode, reference in zip(group, reference_group, strict=True):
                assert episode.tail_kind == reference.tail_kind
                assert torch.equal(episode.tail_bootstrap_value, reference.tail_bootstrap_value)
                assert (episode.training == reference.training).all()
                assert (episode.audit == reference.audit).all()
                assert all(
                    value.device.type == device for value in episode.training.values(True, True)
                )
                if storage is not None:
                    for key, field in episode.training.items(True, True):
                        assert (
                            field.untyped_storage().data_ptr()
                            == storage[key].untyped_storage().data_ptr()
                        ), key
                snapshots.append((episode.training, episode.training.clone()))
        for actual_generator, reference_generator in zip(
            (*diffusion, *policy), (*reference_diffusion, *reference_policy), strict=True
        ):
            assert torch.equal(actual_generator.get_state(), reference_generator.get_state())
        assert all((training == snapshot).all() for training, snapshot in snapshots)
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert all(
        torch.equal(actual, before)
        for actual, before in zip(
            torch.cuda.get_rng_state_all() if cuda_before else [], cuda_before, strict=True
        )
    )


def test_profiling_preserves_collection_and_explicit_rng():
    outputs, collectors, generators = [], [], []
    timings = []
    for profile in (None, timings):
        collector = _collector(3, "cpu")
        diffusion = tuple(torch.Generator().manual_seed(10 + slot) for slot in range(4))
        policy = tuple(torch.Generator().manual_seed(20 + slot) for slot in range(4))
        outputs.append(
            collector.collect(
                transitions_per_slot=3,
                diffusion_generators=diffusion,
                policy_generators=policy,
                noise_seeds=(10, 11, 12, 13),
                policy_action_seeds=(20, 21, 22, 23),
                timings=profile,
            )
        )
        collectors.append(collector)
        generators.append((*diffusion, *policy))
    assert collectors[0]._envs.events == collectors[1]._envs.events
    for groups in zip(*outputs, strict=True):
        for first, second in zip(*groups, strict=True):
            assert (first.training == second.training).all()
            assert (first.audit == second.audit).all()
    assert all(torch.equal(a.get_state(), b.get_state()) for a, b in zip(*generators, strict=True))
    assert sum(timing.phase == "decision" for timing in timings) == 6
    assert sum(timing.phase == "bootstrap" for timing in timings) == 3


def test_failed_collection_propagates_without_publishing_partial_training(monkeypatch):
    collector = _collector(3, "cpu")
    original = collector._envs.step
    error = RuntimeError("original worker failure")
    count = 0

    def step(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(collector._envs, "step", step)
    storage = TensorDict({}, batch_size=[12])
    with pytest.raises(RuntimeError) as failure:
        collector.collect(
            transitions_per_slot=3,
            diffusion_generators=tuple(
                torch.Generator().manual_seed(10 + slot) for slot in range(4)
            ),
            policy_generators=tuple(torch.Generator().manual_seed(20 + slot) for slot in range(4)),
            noise_seeds=(10, 11, 12, 13),
            policy_action_seeds=(20, 21, 22, 23),
            training_storage=storage,
        )
    assert failure.value is error
    assert not list(storage.keys())
