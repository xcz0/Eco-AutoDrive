# Prewarm background traffic without ego interaction

**Status:** Accepted, not implemented

The target traffic initialization lets background traffic evolve while ego is absent from collision, blocking and traffic decisions, then introduces ego at the fixed evaluation anchor with continuous history and stable object IDs. This preserves real history while avoiding the initialization disturbance caused by a stationary ego; implementation is tracked in GitHub Issue #2.

该接受目标不等于既有 stationary-ego warmup 已满足要求；本次迁移不改变未实现状态。

规范归属：[Execution contract](../contracts/execution.md)、[Data/model contract](../contracts/data-and-model.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
