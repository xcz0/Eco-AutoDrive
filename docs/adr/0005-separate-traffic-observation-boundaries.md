# Separate traffic and no-traffic observation boundaries

Use a strict no-traffic adapter only for provably empty scenes and a history-aware adapter for scenes with traffic. The separation makes accidental omission of existing participants a boundary error instead of an apparently valid observation.

规范归属：[Data/model contract](../contracts/data-and-model.md)、[Execution contract](../contracts/execution.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
