# Preserve raw planner failures

Execute and retain the model's raw trajectory without smoothing, clipping, centerline projection, seed selection or fallback control. A failed episode is experimental evidence and must remain attributable to the evaluated planner and stated execution condition.

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
