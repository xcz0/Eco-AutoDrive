"""Strict persisted field sets for rollout artifacts."""

from __future__ import annotations

from eco_planner.rl.rollout.contracts import RewardProfileName, rollout_audit_keys

_ROLLOUT_METADATA_FIELDS = ("reward_profile", "tail_kind", "tail_bootstrap_value")


def rollout_artifact_fields(reward_profile: RewardProfileName) -> tuple[str, ...]:
    """Return the exact NPZ fields for one explicit reward profile."""

    return (*rollout_audit_keys(reward_profile), *_ROLLOUT_METADATA_FIELDS)


BUILTIN_ROLLOUT_ARTIFACT_FIELDS = rollout_artifact_fields("metadrive_builtin_v1")
ENERGY_ROLLOUT_ARTIFACT_FIELDS = rollout_artifact_fields("plannerrft_energy_v1")
