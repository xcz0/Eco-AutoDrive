# Separate experiment execution, descriptive analysis, and static reporting

**Status:** Accepted and implemented
**Date:** 2026-09-09

Experiment modules combined fixed-batch backward passes, reusable NumPy statistics, and Markdown
presentation. Reusing statistics or changing figures consequently pulled training dependencies into
artifact inspection. Several report paths also lacked an independent way to regenerate their output.

Keep experiment execution and protocol decisions in `experiments`, including calibration, GAE,
backward-only diagnostics, gate attribution, exact replay validation, and candidate promotion.
Introduce `analysis` for descriptive computations on saved arrays, typed summaries, and Optuna state;
place presentation in `analysis.reporting`. Experiment writers and an explicit offline `analyze` CLI
reuse these computations and figure helpers. Directory moves follow responsibilities, rather than
moving every function whose name contains “analyze” or “report”.

The interface is the existing artifact collection. Original artifact names, array meanings, and
experimental acceptance results remain authoritative. Derived results are separate JSON, Markdown,
SVG, and PNG artifacts. Training summary models load independently of rollout code. Statistics that
already had multiple consumers move to the analysis layer, retaining their numerical definitions.
Typed evaluation readers remain owned by evaluation, consistent with ADR 0031.

Use Matplotlib and Optuna's native Matplotlib visualizations. No dashboard, plugin registry, schema
migration framework, or statistical inference platform is introduced. Existing user-added analysis
dependencies remain available without creating new research tasks merely to use them.

Scalar reward comparisons require an explicit list of artifacts because existing job summaries do
not completely encode arm/training-seed grouping. Study inspection opens existing SQLite state in
read-only mode. Experimental decisions remain recorded evidence, never silently re-evaluated by
plotting. Insufficient data and undefined quantities remain visible in reports.

The implemented input/output and numerical contracts have one authoritative home in
[system-contract.md](../agents/system-contract.md#实验离线分析与报告). This decision adds the analysis
boundary without replacing evaluation's artifact ownership or changing training and simulator semantics.
