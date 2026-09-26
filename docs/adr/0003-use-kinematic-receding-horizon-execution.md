# Use kinematic receding-horizon execution for the baseline

Use direct trajectory-point execution with frequent replanning to isolate and validate the model, coordinates, observations, map adaptation and closed-loop interface before introducing low-level control. Conclusions from this baseline are explicitly limited to the kinematic execution condition.

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Execution contract](../contracts/execution.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
