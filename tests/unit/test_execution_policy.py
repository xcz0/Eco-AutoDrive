from __future__ import annotations

from types import SimpleNamespace

import pytest

from eco_planner.envs.metadrive.policy import KinematicTrajectoryPolicy


def test_reset_step_with_empty_external_action_does_not_require_a_cached_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = object.__new__(KinematicTrajectoryPolicy)
    policy._trajectory = None
    policy._cache_last_update = None
    engine = SimpleNamespace(external_actions={"ego": None}, episode_step=0)
    monkeypatch.setattr(KinematicTrajectoryPolicy, "engine", property(lambda self: engine))

    assert policy.act("ego") is None


def test_noninitial_step_without_a_cached_trajectory_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = object.__new__(KinematicTrajectoryPolicy)
    policy._trajectory = None
    policy._cache_last_update = None
    engine = SimpleNamespace(external_actions={"ego": None}, episode_step=1)
    monkeypatch.setattr(KinematicTrajectoryPolicy, "engine", property(lambda self: engine))

    with pytest.raises(RuntimeError, match="trajectory continuation"):
        policy.act("ego")
