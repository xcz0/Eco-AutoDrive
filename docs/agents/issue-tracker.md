# Issue tracker: GitHub

GitHub Issues 是仓库中 **active implementation work 和持久化验收标准** 的规范入口。Issue 描述“要变成什么”，不替代代码、测试或 `system-contract.md` 对“当前已经是什么”的描述。

本文件只定义 Issue 工作流；一般实现、验证和权限规则属于 `AGENTS.md`。

## 什么时候使用 Issue

适合进入 Issue 的内容：

- 已接受、可执行，但需要跨会话持续跟踪的开发工作；
- 有明确目标和验收标准的 bug、重构、性能优化或实验基础设施任务；
- 需要拆成 parent/child work items 的较大改动；
- research proposal 已经被接受为 implementation work。

不要把以下内容塞进 Issue 代替其权威位置：

- 当前已实现系统契约 → `docs/agents/system-contract.md`；
- 尚未接受的研究假设 → `docs/research/`；
- 已运行实验的 provenance 和结果 → `docs/experiments/`；
- 长期设计理由 → `docs/adr/`。

Pull Request 可以引用 Issue，但不替代 Issue 作为任务/规格入口。

## GitHub 访问

本地编码智能体使用已认证的 `gh` CLI 进行 Issue 操作。

在 Codex Windows 沙箱中调用 `gh` 时，请求 sandbox approval 并在沙箱外执行。沙箱身份不能读取用户的 Windows credential manager，否则可能得到误导性的 HTTP 401。

不要把 token 写入仓库、命令输出或诊断信息，也不要使用 `gh auth status --show-token`。只有下面的沙箱外检查也失败时才要求重新认证：

```text
gh auth status --hostname github.com
```

在仓库 clone 内运行时，从 `git remote -v` 推断仓库；不要硬编码 owner/repo。

## 读取 Issue 后先提取任务契约

当用户给出 Issue 编号，或工作流要求“fetch the relevant ticket”时，先读取 body、comments 和 labels：

```powershell
gh issue view <number> --comments
```

在动手前提取当前任务真正需要的部分：

- **Goal / Problem**：为什么要改；
- **Scope / Tasks**：本次允许修改什么；
- **Non-goals**：明确不做什么；
- **Acceptance criteria**：完成的可验证条件；
- **Dependencies / parent-child links**：只有确实影响执行顺序时才读取。

Issue 中的目标描述不能被当作当前实现事实。需要判断“现在代码做什么”时，仍以代码、测试、机器可读配置和 system contract 为准。

如果当前用户的明确要求与 Issue 冲突，按当前用户要求执行并指出差异；不要未经请求静默重写 Issue 来消除冲突。

## 创建 Issue

当任务或 skill 明确要求“publish to the issue tracker”时才创建 Issue。不要因为发现一个可选改进就自动外部写入。

推荐的最小结构：

```markdown
## Goal

[希望达到的可观察结果]

## Context

[只有理解任务所必需的现状/问题]

## Scope

- [要完成的工作]

## Non-goals

- [容易误扩展、且本次明确不做的内容]

## Acceptance criteria

- [ ] [可验证的完成条件]
```

不需要为了形式完整而强行增加所有章节。小型 Issue 可以省略 `Context` 或 `Non-goals`；复杂 umbrella Issue 才增加 child issues、推荐顺序或 dependency 说明。

使用 body file，避免 shell quoting 破坏 Markdown：

```powershell
gh issue create --title "..." --body-file <path>
```

## 更新与关闭

常用操作保持简单：

```powershell
# 查看开放工作
gh issue list --state open

# 添加实施结果或阻塞证据
gh issue comment <number> --body-file <path>

# 标签调整
gh issue edit <number> --add-label "..."
gh issue edit <number> --remove-label "..."

# 验收完成后关闭
gh issue close <number> --comment "..."
```

更新 Issue 时记录**任务状态或新的验收信息**，不要复制已经写入代码、ADR、system contract 或实验记录的完整内容。

只有验收标准已经满足，或用户明确决定取消/不实施时才关闭。若某项验收依赖实际实验，先在 `docs/experiments/` 登记运行证据，再在 Issue 中引用对应实验记录；Issue comment 本身不是实验 provenance。
