# Add an explicit five-step DDIM sampler

选择把五步 DDIM 作为显式研究变量加入，同时保留官方十步 DPM-Solver++ 受控基线，
以免将 sampler 替换混入模型适配。独立配置使初始噪声尺度、时间序列和 stochasticity
能够分别比较和追溯。

PlannerRFT 论文没有公开 DDIM timestep subsequence，因此本项目选择均匀连续时间 schedule，
不能将它描述为作者实现。标准高斯 profile 用于论文文本条件的复现；另设半尺度噪声 profile
只用于隔离它与官方 baseline 的初始分布差异，不构成 PlannerRFT parity。
具体 schedule 与随机流消费规则分别归 Protocol 和 Contract，不在 ADR 维护另一套采样定义。

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Execution contract](../contracts/execution.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
