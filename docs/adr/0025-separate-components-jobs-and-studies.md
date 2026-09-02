# Separate reusable components, runnable jobs, and research studies

**Status:** Accepted and implemented; terminology and entrypoint layout amended by ADR 0030
**Date:** 2026-08-27

Repository entrypoints previously mixed thin CLI wrappers, benchmark implementations, fixed research
experiments, and directly importable test helpers. Hydra configuration likewise mixed reusable groups,
runnable jobs, and experiment manifests under paths such as `configs/matrices/`. This made dependency
direction and the stability of each interface unclear.

This ADR's configuration layering remains accepted. Its decision to keep repository-only
`benchmarking`, `studies`, and `analysis` logic under `scripts` is superseded by
[ADR 0027](0027-use-internal-application-modules-and-resource-overlays.md).

Hydra configuration has three layers:

- `configs/components/` contains reusable composition units;
- `configs/jobs/` contains directly runnable evaluation, training, and benchmark profiles;
- `configs/experiments/` contains fixed research manifests and their experiment-specific jobs.

A component with one implementation remains a single file. A config-group directory is introduced
only after multiple selectable implementations exist. Component files use explicit Hydra packages so
the resolved root schema consumed by the existing strict configuration models does not depend on the
physical directory layout.

Old script paths, config names, and aliases are removed rather than maintained through compatibility
forwarders. Historical experiment records retain their original commands and paths as provenance.
Runtime behavior, random streams, experiment parameters, and stable evaluation/training artifact
schemas are outside this organizational change.
