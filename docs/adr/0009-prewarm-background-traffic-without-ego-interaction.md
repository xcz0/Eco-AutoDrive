# Prewarm background traffic without ego interaction

**Status:** Accepted, not implemented

The target traffic initialization lets background traffic evolve while ego is absent from collision, blocking and traffic decisions, then introduces ego at the fixed evaluation anchor with continuous history and stable object IDs. This preserves real history while avoiding the initialization disturbance caused by a stationary ego; implementation is tracked in GitHub Issue #2.

该接受目标不等于既有 stationary-ego warmup 已满足要求；本次迁移不改变未实现状态。

规范归属：[Execution contract](../contracts/execution.md)、[Data/model contract](../contracts/data-and-model.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
