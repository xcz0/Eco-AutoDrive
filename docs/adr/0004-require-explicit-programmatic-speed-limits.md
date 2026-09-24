# Require explicit programmatic speed limits

Treat a programmatic lane speed as an explicit experiment condition and replace only MetaDrive's exact unset sentinel while preserving legitimate lane limits. This avoids silently changing map semantics or letting an unset value enter the model as an extreme valid speed.

规范归属：[Data/model contract](../contracts/data-and-model.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
