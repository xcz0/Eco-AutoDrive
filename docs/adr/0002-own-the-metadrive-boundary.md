# Own the MetaDrive integration boundary

Port the checkpoint-compatible model, but implement ego, traffic, map, route and planner lifecycle adaptation against MetaDrive APIs inside this project. Upstream snapshots are read-only references for semantics and numerical comparison, never runtime dependencies.

规范归属：[Data/model contract](../contracts/data-and-model.md)、[Execution contract](../contracts/execution.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
