# Use internal application modules and execution resource overlays

**Status:** Partially superseded by ADR 0030; the resource-overlay decision remains accepted.
**Date:** 2026-08-31

Scripts 曾积累配置组合、ranking、产物解释与编排，测试直接导入这些文件，但类型检查只
覆盖安装包。稳定 application logic 移到内部 package，是为让它接受同样验证并通过共享
typed boundary 复用；这不把科研仓库变成第三方 SDK，也不把 CLI 当内部 RPC。

当时 benchmarking/studies/analysis 的命名已由
[ADR 0030](0030-align-cli-and-experiment-config-layout.md) 修订；
scripts 只承担入口、bootstrap、展示与退出码的设计理由继续有效。

Resources 被选为独立 overlay，使 job 的科学定义不受机器容量影响。显式选择优先于
机器自动选择，缺少预算时失败而不猜测，这是可追溯性与使用便利之间的明确取舍。
具体 precedence、composition 和执行保证由 execution contract 拥有。

Justfile 被定位为 Windows PowerShell 下的薄 alias，避免工作流语义在第二处出现。
历史入口和配置命名不作为当前操作说明。

规范归属：[Execution contract](../contracts/execution.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
