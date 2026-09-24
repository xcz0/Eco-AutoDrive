# Separate fixed-batch collection and unify experiment workflows

**Status:** Accepted and implemented
**Date:** 2026-09-10

> [ADR 0038](0038-consolidate-scientific-workflows.md) 已取代通用机制留在 experiments、
> 旧 CLI、强制参考链和历史工作流保留的部分；数值、配对与证据边界保留。

Runner 之间曾互相借用 batch 恢复与校准，通用 reward/gradient 操作归属个别诊断，
collection 又与 lambda analysis 绑定。选择将采集、共享数值机制和各诊断编排分开，
让依赖方向表达职责而非实验出现顺序。

当时把共享 fixed-batch 放在 experiments，使用显式 batch 与诊断 references，并统一命令；
这是历史选择。它取代了 [ADR 0030](0030-align-cli-and-experiment-config-layout.md) 的
experiment-specific aliases，以及 [ADR 0034](0034-separate-experiment-analysis-and-reporting.md)
的独立分析 CLI／旧 artifact 兼容要求，保留了后者的职责分离。

[ADR 0036](0036-group-research-domains-and-explicit-cli-operations.md) 后续调整模块位置
与 flat CLI，[ADR 0038](0038-consolidate-scientific-workflows.md) 再移除强制历史参考链。
样本／策略身份对齐和共享数值 primitive 的理由继续成立，但不能据此恢复 reference-dir 守卫。

当时直接切换内部路径和新 artifact，未建设兼容框架；软件验证不等于新研究证据，
历史命令、路径与实验结果保持其原条件。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)、[Diagnostic protocol](../research/protocols/diagnostic-studies.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。

