# Domain vocabulary

This file defines how agent skills should consume stable domain and repository vocabulary. It is intentionally narrower than `AGENTS.md` and does not repeat general repository, implementation, validation, or Git workflow rules.

## Authoritative vocabulary

- `CONTEXT.md` is the canonical glossary for project domain concepts.
- `docs/agents/system-contract.md` defines stable data-flow, execution, timing, shape, unit, and runtime-semantics contracts.
- `docs/adr/` records long-lived design decisions and their rationale.

Use the exact terms defined by these sources when naming domain concepts in issues, implementation plans, tests, experiments, or documentation.

## Vocabulary rules

Do not silently replace a defined term with a near-synonym that changes its meaning.

If a concept needed by the work is not defined in `CONTEXT.md`, do not invent a permanent repository definition here. Note the vocabulary gap so it can be resolved in the authoritative glossary when necessary.

Experiment IDs, issue-specific labels, temporary stage names, and historical artifact-format labels are not stable domain vocabulary unless an authoritative source explicitly defines them as such.

If proposed terminology or behavior conflicts with an existing ADR, surface the conflict explicitly rather than silently redefining the concept.

## Boundaries

This file is only a vocabulary-routing document.

- General repository rules belong in `AGENTS.md`.
- GitHub issue workflow belongs in `docs/agents/issue-tracker.md`.
- Active research questions belong in `docs/research/`.
- Experiment provenance and run-specific terminology belong in `docs/experiments/`.
