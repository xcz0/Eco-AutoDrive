# Consolidate runtime ownership and evaluation topology

**Status:** Accepted and implemented
**Date:** 2026-09-01

## Context

Evaluation and policy rollout already shared Fabric setup and CUDA-to-host behavior, but runtime
resources still lived in a top-level module, host copies were exposed as free functions, and each
caller implemented the same per-generator standard-normal loop. Evaluation configuration also mixed
semantic execution selection with machine capacity: `mode`, vector slot count, and worker thread
count were stored in one job subtree. The rollout runtime additionally owned decision adaptation and
benchmark profiling contracts, while the parent vector façade contained the TorchRL worker class.

These boundaries made internal ownership harder to read and allowed a semantic job to change merely
because a different resource profile supplied a vector slot count.

## Decision

`eco_planner.runtime` owns resource profiles, `HostTrajectories`, `HostTransfer`, and the shared
per-generator batched standard-normal sampler. Evaluation and rollout each construct one
`HostTransfer`; the synchronous ego-trajectory copy remains independent of the deferred full audit
copy. Per-slot generators retain their previous draw order and state transitions.

Evaluation jobs select exactly one topology: `serial`, `vector`, or `job_parallel`, plus the
deterministic flag. Resource profiles supply only numeric capacity. The runner resolves topology and
capacity into the unchanged formal runtime metadata fields (`mode`, launcher, worker count, vector
slots, worker threads), configures the process, and chooses serial or vector execution. The
inference runtime owns only model assembly, device placement, inference, and host-transfer behavior.

Rollout decisions and profiling live in separate modules from the Fabric inference runtime. The
permanently empty `guidance_action_check` benchmark field is removed. All environment-runtime
infrastructure lives under `runtime.envs`: the TorchRL adapter, worker and remote-method result
contracts, and the parent-process `VectorMetaDriveEnv` façade. `eco_planner.envs` owns only domain,
observation and simulator-facing services and does not re-export runtime types. The unused generic
slot abstraction and the old `envs.runtime` / `envs.torchrl` packages are deleted. This is a direct
internal cutover without forwarding modules.

A runtime-persistent CUDA audit stream was measured as a candidate. It did not show a consistent
improvement across the fixed rollout shapes under the observed machine-state variation, so the
final `HostTransfer` creates an independent audit stream per deferred transfer. The structural owner
is retained; stream lifetime remains an internal performance choice.

## Consequences

Semantic jobs no longer inherit a topology from machine resources, and the same job keeps its
execution meaning across hosts. Formal evaluation artifacts remain compatible, while resolved
execution config and benchmark-only schemas use the new topology and profiling fields directly.

Evaluation and rollout share one random-sampling and host-transfer implementation without changing
planner outputs, random streams, trajectory execution, rewards, termination, checkpoints, or
training artifact schemas. Internal imports move atomically to their owning layers.
