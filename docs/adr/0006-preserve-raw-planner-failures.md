# Preserve raw planner failures

Execute and retain the model's raw trajectory without smoothing, clipping, centerline projection, seed selection or fallback control. A failed episode is experimental evidence and must remain attributable to the evaluated planner and stated execution condition.

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
