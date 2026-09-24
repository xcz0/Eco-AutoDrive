# Use kinematic receding-horizon execution for the baseline

Use direct trajectory-point execution with frequent replanning to isolate and validate the model, coordinates, observations, map adaptation and closed-loop interface before introducing low-level control. Conclusions from this baseline are explicitly limited to the kinematic execution condition.

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Execution contract](../contracts/execution.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
