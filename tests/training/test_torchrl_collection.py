"""Check synchronous collection against hand-specified trajectories and RNG draws."""

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
)
from eco_planner.envs.parallel.torchrl import planner_observation_spec
from eco_planner.rl.rollout.collector import VectorRolloutCollector
from eco_planner.rl.rollout.profiling import RolloutPlannerTiming
from eco_planner.runtime.profiling import PhaseTiming
from tests.training.rollout_helpers import (
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


# Each episode specifies: tail, observation markers, decision draw indices,
# optional (final-state marker, bootstrap draw index), and final environment flags.
# Bootstrap draws share the diffusion stream with decisions. In s1, draw 2 belongs
# to the truncated final state (12), before reset returns the marker to 10.
_EXPECTED_EPISODES = (
    (
        ("terminated", (0,), (0,), None, True, True),
        ("terminated", (0,), (1,), None, True, True),
        ("terminated", (0,), (2,), None, True, True),
    ),
    (
        ("truncated", (10, 11), (0, 1), (12, 2), False, True),
        ("rollout_limit", (10,), (3,), (11, 4), False, False),
    ),
    (
        ("terminated", (20, 21), (0, 1), None, True, False),
        ("rollout_limit", (20,), (2,), (21, 3), False, False),
    ),
    (("rollout_limit", (30, 31, 32), (0, 1, 2), (33, 3), False, False),),
)

# Capacity 3 exercises partial resets and a final wave with one active slot.
_EXPECTED_EVENTS = [
    ("reset", (0, 1, 2), (0, 1, 2)),
    ("plan", (0, 10, 20)),
    ("step", (0, 1, 2), (0, 10, 20)),
    ("reset", (0,), (0,)),
    ("plan", (0, 11, 21)),
    ("step", (0, 1, 2), (0, 11, 21)),
    ("bootstrap", (12,)),
    ("reset", (0, 1, 2), (0, 1, 2)),
    ("plan", (0, 10, 20)),
    ("step", (0, 1, 2), (0, 10, 20)),
    ("bootstrap", (11, 21)),
    ("reset", (0,), (3,)),
    ("plan", (30,)),
    ("step", (0,), (30,)),
    ("plan", (31,)),
    ("step", (0,), (31,)),
    ("plan", (32,)),
    ("step", (0,), (32,)),
    ("bootstrap", (33,)),
]


def _assert_expected_group(group, scenario, noise, actions):
    expected = _EXPECTED_EPISODES[scenario]
    assert len(group) == len(expected)
    action_offset = 0
    for episode, (tail, markers, draw_indices, bootstrap, terminal, truncated) in zip(
        group, expected, strict=True
    ):
        length = len(markers)
        values = torch.tensor(markers, device=noise.device) + noise[list(draw_indices)]
        tail_value = (
            torch.zeros(1, device=noise.device)
            if bootstrap is None
            else (bootstrap[0] + noise[bootstrap[1]]).reshape(1)
        )
        assert episode.tail_kind == tail
        assert episode.transition_count == length
        torch.testing.assert_close(episode.tail_bootstrap_value, tail_value)
        training, audit = episode.training, episode.audit
        torch.testing.assert_close(training["state_value"].flatten(), values)
        torch.testing.assert_close(
            training["next", "state_value"].flatten(), torch.cat((values[1:], tail_value))
        )
        for key, last in (("done", True), ("terminated", terminal), ("truncated", truncated)):
            assert training["next", key].flatten().tolist() == [False] * (length - 1) + [last]
        expected_actions = actions[action_offset : action_offset + length]
        torch.testing.assert_close(training["guidance_action"], expected_actions)
        torch.testing.assert_close(audit["guidance_action"], expected_actions.cpu())
        torch.testing.assert_close(audit["base_action"], (expected_actions.cpu() + 1) / 2)
        expected_noise = noise[list(draw_indices)].cpu().reshape(length, 1, 1, 1)
        torch.testing.assert_close(
            audit["initial_noise"], expected_noise.expand_as(audit["initial_noise"])
        )
        torch.testing.assert_close(audit["state_value"].flatten(), values.cpu())
        # Each substep has reward .25 + .01 * speed, with speeds 2, 3, 4, 5.
        expected_reward = torch.full((length, 1), (0.27, 0.55, 0.84, 1.14)[scenario])
        torch.testing.assert_close(training["next", "reward"].cpu(), expected_reward)
        torch.testing.assert_close(audit["reward_total"], expected_reward)
        for key, value in (
            ("map_seed", scenario),
            ("noise_seed", 10 + scenario),
            ("policy_action_seed", 20 + scenario),
            ("reward_substep_count", scenario + 1),
        ):
            assert audit[key].flatten().tolist() == [value] * length
        assert audit["planning_cycle_index"].flatten().tolist() == list(range(length))
        for key, last in (("terminated", terminal), ("truncated", truncated)):
            assert audit[key].flatten().tolist() == [False] * (length - 1) + [last]
        torch.testing.assert_close(audit["route_completion_delta"], torch.full((length, 1), 0.1))
        assert all(value.device == noise.device for value in training.values(True, True))
        action_offset += length
    assert action_offset == 3


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


@pytest.mark.parametrize(
    "device,capacity,sampling,shared_storage",
    [
        ("cpu", capacity, sampling, shared)
        for capacity in (1, 3, 4)
        for sampling in ("sample", "mean")
        for shared in (False, True)
    ]
    + [
        pytest.param("cuda", 3, "sample", True, marks=pytest.mark.gpu),
        pytest.param("cuda", 3, "mean", False, marks=pytest.mark.gpu),
    ],
)
def test_synchronous_collection_matches_explicit_trajectories(
    device, capacity, sampling, shared_storage
):
    collector = _collector(capacity, device)
    diffusion, policy, expected_diffusion, expected_policy = [
        tuple(torch.Generator(device=device).manual_seed(seed + slot) for slot in range(4))
        for seed in (10, 20, 10, 20)
    ]
    global_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if device == "cuda" else []
    snapshots = []
    for _update in range(2):
        # These draw counts and positions come from the explicit episode table,
        # not from collector events, runtime outputs, or lifecycle assembly.
        noises = [
            torch.stack([torch.rand((), generator=g, device=device) for _ in range(count)])
            for g, count in zip(expected_diffusion, (3, 5, 4, 4), strict=True)
        ]
        actions = [
            torch.stack([torch.rand(2, generator=g, device=device) * 1.5 - 0.75 for _ in range(3)])
            if sampling == "sample"
            else torch.zeros(3, 2, device=device)
            for g in expected_policy
        ]
        event_offset = len(collector._envs.events)
        storage = TensorDict({}, batch_size=[12]) if shared_storage else None
        actual = collector.collect(
            transitions_per_slot=3,
            diffusion_generators=diffusion,
            policy_generators=policy,
            noise_seeds=(10, 11, 12, 13),
            policy_action_seeds=(20, 21, 22, 23),
            policy_sampling=sampling,
            training_storage=storage,
        )
        assert len(actual) == 4
        for scenario, group in enumerate(actual):
            _assert_expected_group(group, scenario, noises[scenario], actions[scenario])
            for episode in group:
                snapshots.extend((data, data.clone()) for data in (episode.training, episode.audit))
        if capacity == 3:
            assert collector._envs.events[event_offset:] == _EXPECTED_EVENTS
        if storage is not None:
            ordered = cat([episode.training for group in actual for episode in group])
            assert (storage == ordered).all()
        for actual_generator, expected_generator in zip(
            (*diffusion, *policy), (*expected_diffusion, *expected_policy), strict=True
        ):
            assert torch.equal(actual_generator.get_state(), expected_generator.get_state())
        assert all((data == snapshot).all() for data, snapshot in snapshots)
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
    assert {timing.phase for timing in timings} == {"decision", "bootstrap"}


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
