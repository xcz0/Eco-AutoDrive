# Separate reusable components, runnable jobs, and research studies

**Status:** Accepted and implemented
**Date:** 2026-08-27

Repository entrypoints previously mixed thin CLI wrappers, benchmark implementations, fixed research
studies, and directly importable test helpers. Hydra configuration likewise mixed reusable groups,
runnable jobs, and study manifests under paths such as `configs/matrices/`. This made dependency
direction and the stability of each interface unclear.

Therefore, `scripts` is a non-installed Python package invoked with `python -m`. Its top-level modules
are Hydra entrypoints, while `benchmarking`, `studies`, and `analysis` retain repository-only logic.
Only helpers shared across independent workflows belong in `src/eco_planner`; moving code there does
not turn repository studies into runtime APIs.

Hydra configuration has three layers:

- `configs/components/` contains reusable composition units;
- `configs/jobs/` contains directly runnable evaluation, training, and benchmark profiles;
- `configs/studies/` contains fixed research manifests and their study-specific jobs.

A component with one implementation remains a single file. A config-group directory is introduced
only after multiple selectable implementations exist. Component files use explicit Hydra packages so
the resolved root schema consumed by the existing strict configuration models does not depend on the
physical directory layout.

Old script paths, config names, and aliases are removed rather than maintained through compatibility
forwarders. Historical experiment records retain their original commands and paths as provenance.
Runtime behavior, random streams, experiment parameters, and stable evaluation/training artifact
schemas are outside this organizational change.
