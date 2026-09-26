# Separate traffic and no-traffic observation boundaries

Use a strict no-traffic adapter only for provably empty scenes and a history-aware adapter for scenes with traffic. The separation makes accidental omission of existing participants a boundary error instead of an apparently valid observation.

规范归属：[Data/model contract](../contracts/data-and-model.md)、[Execution contract](../contracts/execution.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
