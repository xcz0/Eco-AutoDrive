# Track PPO through Fabric and MLflow

## Context

The custom PPO loop already exposes typed update summaries and persists research artifacts.
Experiment comparison needs a searchable run index and live curves without coupling reward,
rollout, or TorchRL optimization to a tracking backend.

## Decision

Use Lightning Fabric `log_dict` with Lightning's `MLFlowLogger`. A training-level adapter owns
metric names, parameters, run lifecycle and artifact uploads. Keep `update_observer` for existing
study monitors. Use TorchMetrics for detached, update-scoped generic reductions; TorchRL and
domain computations retain their existing ownership.

Enable local SQLite tracking by default. MLflow is an index and visualization layer; resolved
configs, strict summaries, rollout NPZ and checkpoints remain the research artifacts. Do not use
Lightning Trainer, MLflow autologging, model registry, or a custom logger backend framework.

A new training job creates a run. Resume continues the checkpoint's run and absolute update
indices, with fixed scientific parameters. Invocation artifacts preserve changes to execution
controls. Old checkpoints without tracking identity start a run and backfill their stored update
summaries; historical parameters must come from the original resolved configuration.

## Consequences

Training checkpoints gain optional tracking identity in loop state, while policy export and
summary schemas remain unchanged. Explicitly disabled tracking preserves an inherited identity.
Logging errors propagate; they do not silently turn tracking off. The current metric denominators,
recovery rules and configuration live in the system contract rather than this ADR.
