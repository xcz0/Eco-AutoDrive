# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Prefer the connected GitHub app
when working in Codex. Use the `gh` CLI when it is authenticated and the connector
does not cover the operation; an unauthenticated local `gh` must not block read-only
issue triage.

## Conventions

- **Create an issue**: use the connected GitHub app, or `gh issue create --title "..." --body "..."` when authenticated. Use a heredoc for multi-line bodies.
- **Read an issue**: use the connected GitHub app, or `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: use the connected GitHub app, or `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: use the connected GitHub app, or `gh issue comment <number> --body "..."` when authenticated.
- **Apply / remove labels**: use the connected GitHub app, or `gh issue edit <number> --add-label "..."` / `--remove-label "..."` when authenticated.
- **Close**: use the connected GitHub app, or `gh issue close <number> --comment "..."` when authenticated.

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone;
the connected GitHub app requires the explicit `owner/name` repository identifier.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Read the relevant issue and its comments through the connected GitHub app, or run
`gh issue view <number> --comments` when `gh auth status` reports a valid login.
