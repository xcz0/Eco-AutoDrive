# Stratify every evaluation episode

Retain every evaluated episode and interpret metrics by scenario features, traffic condition, run phase and termination type. This prevents successful-run filtering and avoids merging incomparable completed, truncated, collision, off-road or runtime-error outcomes.

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
