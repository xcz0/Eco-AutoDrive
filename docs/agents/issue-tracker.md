# GitHub Issue workflow for agents

GitHub Issues 是仓库中**跨会话 active implementation work 与持久化验收标准**的入口。Issue 描述“要变成什么”，不替代代码、测试或 `system-contract.md` 对“当前已经是什么”的描述。

本文件只定义 Issue 工作流。一般实现、验证、权限和文档写回规则见 [`AGENTS.md`](../../AGENTS.md)。

## 什么时候使用 Issue

适合进入 Issue：

- 已接受、可执行且需要跨会话持续跟踪的开发工作；
- 有明确目标和验收标准的 bug、重构、性能或实验基础设施任务；
- 需要 parent/child work items 的较大改动；
- research proposal 已经被接受为 implementation work。

不要用 Issue 取代其他权威位置：

- 当前已实现系统契约 → `docs/agents/system-contract.md`；
- 尚未接受的研究假设 → `docs/research/`；
- 已运行实验的 provenance 和结果 → `docs/experiments/`；
- 长期设计理由 → `docs/adr/`。

Pull Request 可以引用 Issue，但不替代任务规范。

## GitHub 访问 gotcha

本地编码智能体使用已认证的 `gh` CLI。在 Codex Windows 沙箱中调用 `gh` 时，请求沙箱外执行；沙箱身份无法读取用户的 Windows credential manager，可能得到误导性的 HTTP 401。

只有下面的沙箱外检查也失败时才要求重新认证：

```powershell
gh auth status --hostname github.com
```

不要输出 token，也不要使用 `gh auth status --show-token`。在仓库 clone 内从 Git remote 推断 repository，不硬编码 owner/repo。

## 读取 Issue：先提取任务契约

用户给出 Issue 编号，或工作流要求读取相关 ticket 时，先读取 body、comments 和 labels：

```powershell
gh issue view <number> --comments
```

动手前只提取会影响当前执行的内容：

- **Goal / Problem**：为什么要改；
- **Scope / Tasks**：本次允许修改什么；
- **Non-goals**：明确不做什么；
- **Acceptance criteria**：完成的可验证条件；
- **Dependencies / parent-child links**：仅在影响顺序或范围时保留。

Issue 中的目标不能当作当前实现事实。判断“现在代码做什么”时仍读取代码、测试、配置和 system contract。

当前用户请求与 Issue 冲突时，以当前用户请求为本次任务边界并指出差异；不要未经请求重写 Issue 来消除冲突。

## 创建 Issue

只有当前任务或工作流明确要求把工作发布到 issue tracker 时才创建 Issue；不要因为发现可选改进就自动外部写入。

优先使用最小、可执行结构：

```markdown
## Goal

[希望达到的可观察结果]

## Context

[理解任务真正需要的现状]

## Scope

- [要完成的工作]

## Non-goals

- [容易误扩展且本次明确不做的内容]

## Acceptance criteria

- [ ] [可验证的完成条件]
```

小型 Issue 可省略 `Context` 或 `Non-goals`；只有复杂 umbrella Issue 才增加 child issues、依赖或推荐顺序。不要为了模板完整而制造空章节。

使用 body file，避免 shell quoting 破坏 Markdown：

```powershell
gh issue create --title "..." --body-file <path>
```

## 更新与关闭

Issue 只记录**任务状态、阻塞信息或新的验收信息**。实现细节、系统契约和实验 provenance 写回各自权威位置后，在 Issue 中引用即可，不复制全文。

常用操作：

```powershell
gh issue comment <number> --body-file <path>
gh issue edit <number> --add-label "..."
gh issue close <number> --comment "..."
```

只有验收标准已经满足，或用户明确决定取消/不实施时才关闭。若验收依赖真实实验，先在 `docs/experiments/` 登记运行证据，再从 Issue 引用对应记录；Issue comment 本身不是实验 provenance。
