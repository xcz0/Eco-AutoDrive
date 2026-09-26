# Group research domains and expose explicit CLI operations

**Status:** Accepted and implemented
**Date:** 2026-09-10

> [ADR 0038](0038-consolidate-scientific-workflows.md) 已取代通用机制留在 experiments、
> 旧 CLI、强制参考链和历史工作流保留的部分；数值、配对与证据边界保留。

实验包曾混合研究 protocol、reward 正确性检查与 benchmark，flat CLI 还暴露历史 stage 名称。
因此当时按 reward/guidance/training 研究域分组，以显式 operation 避免一次调用隐式跑完整 study，
同时让只读 analysis 不依赖执行。

这取代了 [ADR 0035](0035-separate-fixed-batch-collection-and-experiment-workflows.md)
的模块位置与 flat CLI，但保留独立诊断、共享数值机制、配对与延迟导入的理由。
[ADR 0037](0037-simplify-experiments-and-report-seed-effects.md) 随后取代强制小模块布局
及 reward-validation ownership；ADR 0038 又收口机制与旧工作流。

当时保留算法、预算、seed 和 promotion 规则，是该次组织变更的范围限定，
不是要求复活随后删除的 stage/promotion/stability 框架。依赖版本与当前命令归配置和入口，
历史记录不因内部命名变化而改写。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)、[Diagnostic protocol](../research/protocols/diagnostic-studies.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。

