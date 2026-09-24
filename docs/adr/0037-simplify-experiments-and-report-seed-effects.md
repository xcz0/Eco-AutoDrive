# Simplify experiment modules and report conditional seed effects

**Status:** Accepted and implemented
**Date:** 2026-09-11

> [ADR 0038](0038-consolidate-scientific-workflows.md) 已取代通用机制留在 experiments、
> 旧 CLI、强制参考链和历史工作流保留的部分；数值、配对与证据边界保留。

小型实验包的配置、计算与编排被过度拆散，因此选择按实际复杂度保留文件／包，
取代 [ADR 0036](0036-group-research-domains-and-explicit-cli-operations.md) 的强制
小模块布局及 reward-validation ownership，保留职责边界而不维护目录对称。

比较结果选择先交代完成性与安全，再解释条件 matched energy；逐 training seed
报告配对效应与 scenario bootstrap，避免把固定策略下的场景不确定性误当训练 seed 总体区间。
Initial checkpoint 保留诊断用途，不引入失败能耗 penalty 或混合分数来掩盖可用性差异。
具体 estimator、分母及缺项处理统一归 planning/evaluation protocol。

Provenance 选择 Git/runtime metadata 与正式 clean commit，移除 source copies/tracked diffs，
避免重复保存源码；checkpoint identity 与 replay 检查仍有验证输入的独立作用。
这一决定替代旧的 diff 保存要求，不重写历史 artifacts 或 records。

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。

