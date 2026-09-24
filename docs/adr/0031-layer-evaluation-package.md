# Layer the evaluation package by execution phase

**Status:** Accepted and implemented
**Date:** 2026-09-02

[ADR 0029](0029-consolidate-evaluation-api-and-current-artifacts.md) 统一了入口，
但扁平目录同时容纳推理、episode 执行、recorder 和离线产物，部分文件又混合多类控制流。
因此选择按 inference、episodes、artifacts 三个执行阶段组织内部实现，保留 job 级入口，
而不继续按文件长度机械拆分。

这只取代 ADR 0029 的物理目录布局，保留仓库级延迟入口、轻量离线 reader/report、
typed models 与单一 recorder 的设计理由。直接切换内部路径避免长期 forwarder 成本。

该次分层没有改变配置、随机流、执行或 JSON/NPZ/report 契约。
当前文件组织从代码读取，离线依赖与 artifact 要求统一由 contracts 定义。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
