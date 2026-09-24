# Separate experiment execution, descriptive analysis, and static reporting

**Status:** Accepted and implemented
**Date:** 2026-09-09

> [ADR 0038](0038-consolidate-scientific-workflows.md) 已取代通用机制留在 experiments、
> 旧 CLI、强制参考链和历史工作流保留的部分；数值、配对与证据边界保留。

Experiment 曾把 backward、可复用 NumPy 统计和 Markdown 展示放在一起，导致检查 artifact
也引入训练依赖，且一些报告无法独立再生成。因此选择分离执行／研究裁定、描述统计和展示，
通过持久化证据连接它们，而不按函数名机械移动代码。

当时采用 Matplotlib 与 Optuna 原生图，并保留 read-only study inspection，避免引入 dashboard
或新框架。实验裁定作为已记录证据供报告读取，不由画图重裁；undefined 与不足数据保持可见。
这些理由不要求恢复后来移除的旧数据库分析入口。

[ADR 0035](0035-separate-fixed-batch-collection-and-experiment-workflows.md) 已取代
本篇独立 analyze CLI 和 artifact-compatibility 要求，但保留执行／分析／展示的职责分离。
后续 ADR 0038 进一步收口机制归属；当前命令、字段与工作流不在本篇重复维护。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)、[Diagnostic protocol](../research/protocols/diagnostic-studies.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。

