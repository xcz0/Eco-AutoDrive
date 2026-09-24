---
name: research-code-review
description: "Review research code changes for spec compliance, scientific semantic integrity, and implementation correctness. Use for commit, branch, PR, or uncommitted-change reviews, with or without an Issue."
---

# Research Code Review

Review the requested changes along all three axes below. Review authorization does not authorize implementing fixes or writing to GitHub.

## Establish the review scope

Use the user's request to identify the comparison and state it in the report. Inspect `git status --short` so local work is not mistaken for committed changes.

- For a commit, compare its parent with that commit; for explicit endpoints, use `git diff <base> <target>`. For merge commits, establish which parent is intended.
- For branch or PR changes since divergence, resolve the base and target, then use `git diff <base>...<target>` (merge-base to target). Use direct endpoint comparison instead when the user requests it. Record resolved revisions so the review is stable.
- For uncommitted work, `git diff` covers unstaged tracked changes, `git diff --cached` covers staged changes, and `git diff HEAD` covers their combined tracked result. Inspect in-scope untracked files separately; none of these diffs includes them.
- For a combined review, include both the committed range and the requested local work, and distinguish their sources. Do not silently substitute a committed diff for WIP.

Resolve refs before reviewing. If the selected scope is empty, report that rather than inventing findings. Clarify only when existing context cannot resolve a material scope ambiguity; an uncommitted review does not require a base branch.

## Establish intent and relevant contracts

Start with the current user request and any supplied Issue or spec. Read only the Issue details needed for goals, scope, non-goals, and acceptance criteria, using available GitHub integration or the repository's documented fallback. No Issue is required: the user request can supply the spec. If intended behavior is incomplete, state which requirements cannot be assessed and continue the semantic and correctness review.

Follow `AGENTS.md` for authority and source selection. Read `docs/research/semantics.md` for concepts and only the relevant active Protocol and Contract sections routed there for research methods and software guarantees. Read ADRs when design rationale matters. Runtime observations, code, tests, and config establish actual behavior; a conflict with active Protocol/Contract is a conformance mismatch to investigate, not automatic evidence that the specification is stale. An Issue or hypothesis does not replace active requirements or prove that a behavior exists. Surface conflicting sources instead of synthesizing a new contract; the current request controls this task's scope.

## Review axes

### Spec compliance

Check whether the change satisfies the request and applicable Issue acceptance criteria, including missing or partial requirements and scope creep. Tie findings to specific requirements. Do not treat absent Issue metadata as a reason to skip review, or invent requirements from personal design preferences.

### Scientific semantic integrity

Trace affected research definitions through configuration, data flow, and execution.
Check for silent changes to baseline, algorithm, sampling, reward, metric, timing, termination, randomness, and evaluation protocol. Relevant contracts may include units, coordinates, episode boundaries, seed ownership, and train/evaluation data separation; inspect those reached by this change rather than auditing the whole repo.

Distinguish explicitly requested protocol changes from accidental ones, and assess whether their implementation and evidence match the declared intent. Flag defaults, fallbacks, filtering, clipping, or seed selection that conceal failures or alter the research question. Software tests alone do not establish safety, energy savings, parity, or other scientific conclusions.

### Implementation correctness

Trace changed behavior through relevant callers and consumers, beyond diff hunks when necessary. Examine affected boundary conditions, shapes/types/devices, data flow, resource lifecycle, exception handling, RNG state, concurrency, and resume behavior. These are investigation directions, not a checklist to apply to every file.

Assess whether existing or executed validation reaches the changed behavior and supports the claimed software conclusion. Identify concrete gaps and consequences; do not demand broad tests or repeated experiments without a relevant risk. Apply the repository's verification rules if running checks, and distinguish checks actually run from evidence inspected or verification still needed.

Style, formatting, and lint belong to repository tools. Do not maintain a separate style rubric or fixed code-smell baseline in this skill.

## Scale the work and report findings

Review locally for small, coupled changes. When useful and available, delegate independent axes or subsystems in parallel based on scope and risk; no fixed agent count is required. Give delegates the exact review scope, intent, and relevant contracts, and ensure the combined review covers all three axes.

Consolidate duplicate findings and order actionable findings by severity. Retain all applicable axis labels on each finding, with the file/line, triggering condition, consequence, and concrete requirement, contract, or code evidence. Separate established defects from unresolved questions; plausibility alone is not a confirmed bug.

State the comparison reviewed, evidence used, and verification limits. Account for each axis, including when no actionable issue was found or requirements were unavailable. An empty finding list is not proof of correctness or scientific validity.
