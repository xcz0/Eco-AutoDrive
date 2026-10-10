"""Planning-owned exploration policy architecture and checkpoint API."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .checkpoint import (
        PolicyCheckpointReport,
        load_exploration_policy_checkpoint,
        policy_state_hash,
        save_exploration_policy_checkpoint,
    )
    from .config import (
        ExplorationPolicyConfig,
        parse_exploration_policy_config,
    )
    from .distribution import (
        AffineBeta,
        AffineBetaParameters,
        ExplicitGeneratorBetaSampler,
    )
    from .inputs import (
        POLICY_CONTEXT_KEYS,
        ExplorationPolicyContext,
        build_policy_inputs,
        policy_context_tensordict,
        validate_exploration_policy_context,
    )
    from .model import (
        ExplorationPolicy,
        ExplorationPolicyOutput,
    )

_EXPORTS = {
    "PolicyCheckpointReport": (".checkpoint", "PolicyCheckpointReport"),
    "load_exploration_policy_checkpoint": (".checkpoint", "load_exploration_policy_checkpoint"),
    "policy_state_hash": (".checkpoint", "policy_state_hash"),
    "save_exploration_policy_checkpoint": (".checkpoint", "save_exploration_policy_checkpoint"),
    "ExplorationPolicyConfig": (".config", "ExplorationPolicyConfig"),
    "parse_exploration_policy_config": (".config", "parse_exploration_policy_config"),
    "AffineBeta": (".distribution", "AffineBeta"),
    "AffineBetaParameters": (".distribution", "AffineBetaParameters"),
    "ExplicitGeneratorBetaSampler": (".distribution", "ExplicitGeneratorBetaSampler"),
    "POLICY_CONTEXT_KEYS": (".inputs", "POLICY_CONTEXT_KEYS"),
    "ExplorationPolicyContext": (".inputs", "ExplorationPolicyContext"),
    "build_policy_inputs": (".inputs", "build_policy_inputs"),
    "policy_context_tensordict": (".inputs", "policy_context_tensordict"),
    "validate_exploration_policy_context": (".inputs", "validate_exploration_policy_context"),
    "ExplorationPolicy": (".model", "ExplorationPolicy"),
    "ExplorationPolicyOutput": (".model", "ExplorationPolicyOutput"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "AffineBeta",
    "AffineBetaParameters",
    "ExplorationPolicy",
    "ExplorationPolicyConfig",
    "ExplorationPolicyContext",
    "ExplorationPolicyOutput",
    "ExplicitGeneratorBetaSampler",
    "POLICY_CONTEXT_KEYS",
    "PolicyCheckpointReport",
    "build_policy_inputs",
    "load_exploration_policy_checkpoint",
    "parse_exploration_policy_config",
    "policy_context_tensordict",
    "policy_state_hash",
    "save_exploration_policy_checkpoint",
    "validate_exploration_policy_context",
]
