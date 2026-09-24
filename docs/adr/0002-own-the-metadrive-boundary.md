# Own the MetaDrive integration boundary

Port the checkpoint-compatible model, but implement ego, traffic, map, route and planner lifecycle adaptation against MetaDrive APIs inside this project. Upstream snapshots are read-only references for semantics and numerical comparison, never runtime dependencies.

规范归属：[Data/model contract](../contracts/data-and-model.md)、[Execution contract](../contracts/execution.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
