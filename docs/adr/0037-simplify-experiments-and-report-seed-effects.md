# Simplify experiment modules and report conditional seed effects

**Status:** Accepted and implemented
**Date:** 2026-09-11

Small experiment packages separated configuration, diagnostics and orchestration even when one
file could express the workflow clearly. Keep function-level computation/I/O boundaries and retain
packages only where internal modules or length justify them. Single-YAML studies use a file;
energy-sweep retains its evaluation configuration subtree. Reward sanity belongs in one repository
application module. This replaces ADR 0036's mandatory small-module layout and reward-validation
ownership, while preserving its research domains, explicit CLI operations and lazy analysis boundary.

Final scalar comparisons report completion/availability, safety, then conditional matched energy.
A2 minus A1 isolates the energy-reward contrast; both trained arms retain comparisons with A0.
Display each training seed's mean paired effect with a scenario bootstrap interval and report sign
counts across seeds. These intervals condition on each trained policy and its available scenarios;
they do not infer uncertainty over the training-seed population. Initial checkpoints remain
diagnostic. No failure energy penalty, composite score or cross-experiment registry is introduced.

Git/runtime metadata and clean commits identify formal results. Source copies and tracked-diff
artifacts are removed, including training, evaluation and benchmarking writers/readers and MLflow
uploads. This replaces earlier requirements to persist tracked diffs; checkpoint identity and
replay checks remain because they validate experiment inputs and execution, rather than duplicate
source provenance. Historical artifacts and experiment records are not rewritten.

The current numerical, input and artifact contracts live in
[experiment analysis contract](../agents/contracts/experiments.md#实验离线分析与报告).
