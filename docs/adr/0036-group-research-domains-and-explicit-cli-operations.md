# Group research domains and expose explicit CLI operations

**Status:** Accepted and implemented
**Date:** 2026-09-10

Experiment packages mixed research protocols with reward correctness checks and execution
benchmark reports. Flat CLI names also exposed historical stage names as actions. Group research
code and configuration by reward, guidance and training, while retaining separate diagnostic
configurations, computations and orchestration.

Reward owns the existing shared fixed-batch infrastructure because its current consumers are
reward studies. Reward correctness belongs to RL reward validation, with artifact publication
in a repository application module. Execution comparisons belong to benchmarking. Do not add a
diagnostic engine, plugin framework or cross-layer study registry.

Use `just experiment <domain> <study> <action>`. Scalar and stability execution require an explicit
`--operation`; they do not automatically run a full study. The CLI owns a static command table,
argument validation, bootstrap and lazy dispatch. Analysis consumes resolved inputs and saved
evidence without importing experiment modules. Scalar protocol validation remains in its study;
guidance decisions are saved by its study and only read during offline presentation. Stability
analysis regenerates search summaries from read-only study state into a separate output directory.

This supersedes ADR 0035's module locations and flat CLI choices. Its independent diagnostics,
shared numerical primitives, explicit references and lazy imports remain in force, as do ADR
0034's execution, analysis and presentation boundaries. Existing algorithms, budgets, seeds,
promotion rules and numerical tolerances are preserved.

There are no old import or command aliases and no artifact migration. Internal stage keys,
artifact identifiers and filenames remain unchanged except for the explicit guidance decision
artifact. Historical experiment records retain their original commands and paths. SciPy is now
a direct dependency at the existing locked resolution.

Current commands are maintained in the README; artifact and numerical contracts are maintained
in [experiment analysis contract](../agents/contracts/experiments.md#实验离线分析与报告). Verification covers command
routing and bootstrap, numerical and artifact regression, dependency boundaries and the
simulator intervention-to-report path; it is not new research evidence.
