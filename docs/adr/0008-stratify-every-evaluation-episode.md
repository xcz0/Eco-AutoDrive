# Stratify every evaluation episode

Retain every evaluated episode and interpret metrics by scenario features, traffic condition, run phase and termination type. This prevents successful-run filtering and avoids merging incomparable completed, truncated, collision, off-road or runtime-error outcomes.

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
