# Issue tracker: GitHub

GitHub Issues are the canonical tracker for active work and repository specifications. This file defines issue-tracker workflow only; general repository rules belong in `AGENTS.md`.

## Access

Use the authenticated local `gh` CLI for issue operations.

When Codex runs `gh`, request sandbox approval and execute the command outside the sandbox. The sandbox identity cannot read the user's Windows credential manager and may otherwise report a misleading HTTP 401.

Never copy a token into repository files or diagnostic output, and do not use `gh auth status --show-token`. Only request re-authentication if an outside-sandbox

```text
gh auth status --hostname github.com
```

also fails.

Infer the repository from `git remote -v`; `gh` resolves it automatically when run inside the clone.

## Issue operations

- **Create**: `gh issue create --title "..." --body-file <path>`.
- **Read**: `gh issue view <number> --comments`; include labels when a structured read is needed.
- **List**: `gh issue list --state open --json number,title,body,labels,comments` with the   required `--label` and `--state` filters.
- **Comment**: `gh issue comment <number> --body-file <path>`.
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` or   `--remove-label "..."`.
- **Close**: `gh issue close <number> --comment "..."`.

## Workflow conventions

When a skill says **"publish to the issue tracker"**, create a GitHub issue.

When a skill says **"fetch the relevant ticket"**, read the corresponding GitHub issue, including comments and labels.

Pull requests are not the repository's request or specification surface. A pull request may reference an issue, but it does not replace the issue as the tracker entry for active work.
