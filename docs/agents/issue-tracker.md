# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the authenticated local `gh` CLI for issue operations.

## Codex sandbox access

Before running any `gh` command, Codex must send the user a sandbox approval request and run the command outside the sandbox. The sandbox identity cannot read the user's Windows credential manager and can otherwise report a misleading HTTP 401.

Never copy a token into repository files or diagnostic output, and do not use `gh auth status --show-token`. Only request re-authentication if an outside-sandbox `gh auth status --hostname github.com` also fails.

## Conventions

- **Create**: `gh issue create --title "..." --body-file <path>`.
- **Read**: `gh issue view <number> --comments` and include labels in structured reads.
- **List**: `gh issue list --state open --json number,title,body,labels,comments` with the required `--label` and `--state` filters.
- **Comment**: `gh issue comment <number> --body-file <path>`.
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- **Close**: `gh issue close <number> --comment "..."`.

Infer the repository from `git remote -v`; `gh` does this automatically inside the clone.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments` and include its labels.
