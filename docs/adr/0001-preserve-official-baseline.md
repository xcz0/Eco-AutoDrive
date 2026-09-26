# Preserve the official Diffusion Planner baseline

Keep the official model structure, parameter hierarchy, normalization, checkpoint and baseline inference semantics so the pretrained EMA remains a controlled reference. Add MetaDrive, preview and future training capabilities at explicit boundaries rather than mixing network rewrites with research variables.

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Data/model contract](../contracts/data-and-model.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
