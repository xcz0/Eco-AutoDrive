# Layer the RL package

原先平面的 RL package 没有表达 policy、rollout、optimization 与 artifact 的依赖方向，
serial/vector 还重复构造训练与审计 transition。因此当时选择按四类职责分层，
由上层 trainer 编排，而让 collection 共用 episode boundary 机制。

关键理由是让参数所有权和数据边界清晰，避免 next-value、tail、reward audit 在两条路径
分叉；不是永久维护一份模块地图。具体目录与导出由代码拥有，共同 boundary/RNG 保证
由 training contract 定义。

当时以直接 cutover 移除 flat imports 和广泛 re-exports，并把 Hydra 子树从 rl 改称 ppo，
没有新增迁移层。历史 resolved config 仍是原运行记录，不自动成为新输入。
该次分层没有改变 checkpoint、PPO、seed、reward 或冻结 planner；当时“10 Hz 不变”
是历史上下文，正式 cadence 后由 [ADR 0039](0039-unify-closed-loop-cadence.md) 改变。

规范归属：[Execution contract](../contracts/execution.md)、[Training contract](../contracts/training.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
