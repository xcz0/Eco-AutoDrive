# Use internal application modules and execution resource overlays

**Status:** Partially superseded by ADR 0030; the resource-overlay decision remains accepted.
**Date:** 2026-08-31

As benchmark and study workflows grew, `scripts/` accumulated typed configuration, composition,
ranking, artifact interpretation, and orchestration logic. Tests imported those modules directly,
while the project's static type-check boundary covered only `src/eco_planner/`. Study workflows also
reused training and evaluation through inconsistent Hydra and subprocess paths.

The following application-ownership terminology is historical and superseded by ADR 0030. Stable
repository application logic belongs in internal `eco_planner.benchmarking`,
`eco_planner.studies`, and `eco_planner.analysis` modules. `scripts/` contains only CLI parsing,
bootstrap, presentation, and exit-code mapping. Moving these modules into the installed package makes
them tested and type-checked project code; it does not create a stable third-party public API.

ADR 0030 retains the `components / jobs / experiments` configuration layering. A job owns its
complete experiment semantics, while an experiment manifest selects jobs and declares pairing,
search, ranking, or explicit overrides. Shared composition and typed execution boundaries are used
by both CLIs and experiment workflows instead of treating another CLI as an internal RPC endpoint.

Versioned `components/resources` profiles are execution overlays. Semantic jobs contain a null Hydra
group placeholder, so they compose and validate without local machine state. At CLI or study
bootstrap, an explicit Hydra resource override wins over `MACHINE_NAME`; an existing process value
wins over the optional repository `.env`. Resource-dependent execution fails when no profile is
selected rather than inventing a worker budget or hardware fallback.

The repository `justfile` is a Windows PowerShell task alias layer. It forwards workflow arguments to
their owning CLI, keeps formatting separate from read-only validation, and uses pytest markers as the
sole smoke-suite membership source.
