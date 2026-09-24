# Align CLI and experiment configuration layout

**Status:** Accepted and implemented
**Date:** 2026-09-02

Evaluation 重构已把 application logic 移到 experiments，但 CLI/config 仍暴露 studies、
analysis 和不可直接运行的 job-base 路径，使用户面对另一套所有权模型。

当时选择让薄入口、可复用 components、可运行 jobs 与 experiment manifests 的命名一致，
并由各 CLI 拥有动作语义；Just 只转发。旧路径直接移除，历史命令仍作为 provenance。
本次组织改动没有改变科学参数、随机流或 artifact 含义。

本篇取代 [ADR 0025](0025-separate-components-jobs-and-studies.md) 的 studies 术语及
[ADR 0027](0027-use-internal-application-modules-and-resource-overlays.md) 的具体
application 命名；后者的 internal-application 与 resource-overlay 决定保留。
本篇的 experiment-specific CLI aliases 随后被
[ADR 0035](0035-separate-fixed-batch-collection-and-experiment-workflows.md) 取代，
后续工作流收口见 [ADR 0038](0038-consolidate-scientific-workflows.md)。
这些历史布局不作为当前命令清单。

规范归属：[Execution contract](../contracts/execution.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
