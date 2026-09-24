---
name: diagnose-hard-bug
description: "Diagnose hard bugs and performance regressions using observable failure or measurement signals, including stochastic, numerical, and GPU problems. Use for diagnosis/debugging requests; implement fixes only when authorized."
---

# Diagnose Hard Bug

Choose a diagnosis path proportional to the observed problem, available evidence, and authorized scope. The workflow below guides decisions; it is not a mandatory sequence of phases for every bug.

## Establish scope and a useful signal

Distinguish a request to diagnose or explain from a request to fix. Diagnosis may inspect code and run appropriate non-destructive probes, but does not authorize persistent implementation changes. When repair is already authorized, proceed within that scope without seeking repeated confirmation.

Identify the reported symptom and expected behavior from the request and relevant contracts. Follow `AGENTS.md` to select domain, system-contract, ADR, or experiment sources as needed; avoid reading unrelated background by default. Use repository entrypoints and the prepared environment for checks.

Establish the cheapest reliable failure or measurement signal available under current conditions. Prefer an existing failing test, benchmark, trace, log, or saved artifact over building a new harness. Check that it captures the reported symptom and note the conditions that affect its interpretation, such as input/configuration, code revision, seed, device, or workload.

A signal need not be deterministic, finish in seconds, or come from a command already run in this session. For stochastic, numerical, GPU, or performance problems, use appropriate statistical evidence: failure rates, distributions, numerical error, or timing variation. Choose sample counts, tolerances, and comparisons according to the question and measurement noise, making uncertainty explicit. Account for effects such as warmup and asynchronous GPU work when they affect the measurement. Do not choose seeds or thresholds after observing results merely to obtain a favorable verdict.

Reduce a reproduction when doing so improves diagnosis without losing the failure mechanism. Stop reducing when the signal is useful; a realistic workload may be essential and need not be minimal in every element.

## Form and discriminate hypotheses

Use observations and relevant code paths to form falsifiable hypotheses. For each hypothesis worth testing, identify a prediction and an observation that would weaken or reject it. Select the next probe for its ability to distinguish plausible causes, relative to cost. No fixed number of hypotheses or prior executable repro is required.

When feasible, change one causal variable at a time. If variables are coupled or a realistic workload prevents isolation, explain what remains confounded and limit the conclusion accordingly. Target instrumentation at the boundary that distinguishes hypotheses; use inspection, profiling, replay, or bisection when they serve that purpose.

Keep diagnostic interventions separate from the research protocol. Changing sampling, reward, metrics, timing, termination, RNG behavior, baseline, or evaluation conditions can change the phenomenon itself. Make such interventions explicit and do not carry them into a fix or claim protocol-equivalent results without supporting evidence and authorization for any intended protocol change.

If execution is unavailable or the symptom cannot be reproduced, code inspection and existing artifacts may still support hypotheses. Label them as provisional, report what the evidence does and does not establish, and identify the smallest missing observation. Ask for access or information only when it materially blocks progress; do not claim a cause or successful fix from insufficient evidence.

## Establish regression evidence and repair when authorized

Prefer regression evidence that exercises the actual failure mechanism. When a stable test seam exists, add or update the regression test for an authorized fix and, where feasible, demonstrate failure before repair and success afterward. Reuse an adequate existing failing test rather than duplicating it.

When no stable seam exists, explain why and choose an appropriate alternative, such as artifact replay, an integration check, a numerical comparison, or a benchmark with reported variability. A missing seam is a limitation of the available evidence, not automatic proof that the architecture requires redesign. Do not create a shallow test that bypasses the real mechanism merely to report a passing test.

For an authorized repair, make the smallest change supported by the diagnosis and validate it against the relevant signal and acceptance criteria. Recheck the original scenario when a reduced reproduction or surrogate leaves a material gap. Broaden validation only for newly observed failures or unresolved affected behavior. Preserve undefined and failure states rather than hiding them through defaults or fallback behavior.

## Finish within scope

Remove temporary instrumentation and disposable harness changes introduced for this diagnosis; retain useful regression evidence in the appropriate repository location. Do not remove the user's existing work.

Report the observed symptom, supported cause or remaining hypotheses, key evidence, any authorized repair, and validation limits. Distinguish checks actually run from proposed checks, and software correctness from research conclusions. If evidence is insufficient, say so directly. Stop when the requested diagnosis or repair acceptance conditions are met; do not expand into unrelated refactoring or experiments.
