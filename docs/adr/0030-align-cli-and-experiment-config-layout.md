# Align CLI and experiment configuration layout

**Status:** Accepted and implemented
**Date:** 2026-09-02

The evaluation-stack refactor moved repository application logic to `eco_planner.experiments`, but
the repository-facing CLI and configuration layout still retained `studies`, `analysis`, and
non-runnable job-base paths. That exposed a different ownership model to users than the code uses.

The repository uses the following layout:

- `scripts/` contains the thin generic evaluation, training, benchmark, and evaluation-report CLIs.
- `scripts/experiments/` contains one thin CLI per concrete experiment; complex experiments expose
  their own explicit subcommands.
- `configs/components/` contains reusable composition bases, including evaluation and PPO training.
- `configs/jobs/` contains only directly runnable semantic profiles.
- `configs/experiments/` contains fixed experiment manifests and experiment-specific job overlays.
- `justfile` exposes one recipe per domain, with an explicit first action that dispatches only to the
  owning CLI and forwards the remaining arguments unchanged.

Old script modules, Hydra config paths, and Just recipes are removed without compatibility aliases.
Historical experiment records retain their original commands and paths as provenance. This is an
organizational change only: configuration values, random streams, output directories, and persisted
artifact schemas remain unchanged.

This supersedes ADR 0025's `studies` terminology and the application-ownership portion of ADR 0027.
ADR 0027's internal-application and resource-overlay decisions remain in force.
