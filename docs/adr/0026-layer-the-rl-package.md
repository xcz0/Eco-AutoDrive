# Layer the RL package

## Context

The RL implementation had grown as one flat package. Policy math, rollout execution, PPO
optimization, persisted artifacts, and training orchestration were individually implemented, but
their Python import surface did not express their dependency direction. Serial and vector
collectors also assembled the same training and audit transition in separate loops.

## Decision

Organize `eco_planner.rl` into four explicit layers:

1. `policy` owns the Exploration Policy, affine-Beta distribution, context, and architecture
   configuration.
2. `rollout` depends on `policy` and owns inference, collection, and episode contracts.
3. `optimization` depends on `policy` and `rollout` and owns GAE/PPO plus checkpoints.
4. `artifacts` depends on the preceding contracts and owns persisted schemas, I/O, and analysis.

`config` composes complete Hydra job models, while `trainer` is the top-level orchestrator allowed
to depend on every layer. Serial and vector collection share one episode builder for next-value
linking, reward-profile consistency, audit assembly, and tail finalization.

The migration is a direct cutover: old flat-module imports and broad `eco_planner.rl` re-exports
are removed. The Hydra PPO subtree is named `ppo`, not `rl`. Episode artifacts carry an explicit
reward profile and summaries require all current fields; no compatibility aliases or schema
migration branch is added. Policy and resumable training checkpoint contents remain unchanged.

## Consequences

Imports now communicate ownership and dependency direction, and rollout boundary logic has one
implementation. Existing Python callers and Hydra overrides must migrate atomically. Historical
resolved configs that use `rl` are records of their original run and are not accepted as current
training input. PPO mathematics, seed namespaces, 10 Hz transition semantics, reward definitions,
and frozen-planner behavior are unchanged.
