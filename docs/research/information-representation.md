# Information and Representation Study

## 研究问题

本专题研究：

> **为了利用长程闭环反馈降低能耗，策略究竟需要哪些环境信息，以及这些信息应如何被表示和适配？**

当前不把“道路预瞄”预设为唯一核心信息。道路几何、导航、限速变化、动态交通和当前局部场景都属于候选信息来源，需要通过消融判断其增量价值。

## 当前起点

现阶段暂借用 Diffusion Planner 预训练得到的感知/scene representation。

这样做的目的不是假设预训练 representation 已经最优，而是先把两个问题解耦：

1. **information value**：某类信息本身是否对 energy-oriented policy 有用；
2. **representation learning**：已有 encoder 是否能够把这些信息表示得足够适合 RL。

第一阶段固定 encoder，只比较信息；第二阶段再在相同信息条件下比较 frozen / fine-tuned representation。

## 第一阶段：信息价值消融

信息消融应从弱到强分层，而不是一开始加入所有可用特征。

### I0 — Ego + reference trajectory

最小条件集：

```text
ego state
+
frozen planner reference trajectory
```

这一 baseline 用于回答：

> 只观察当前运动状态和原 planner 的计划意图，Exploration Policy 能否已经学到一部分能耗优化行为？

如果该条件已经能够获得大部分收益，则后续环境信息的边际贡献可能有限。

### I1 — Local scene

在 I0 基础上加入当前局部场景 representation，例如：

- local lanes；
- local route / navigation；
- nearby agents；
- static context。

比较 I0 与 I1 可以回答：

> 当前局部环境感知是否对 long-horizon energy optimization 提供额外价值？

### I2 — Long-range road / navigation information

在 I1 基础上增加更远未来道路与导航结构。候选包括：

- future curvature / curvature changes；
- speed-limit changes 及变化位置；
- merge / split / turn 等拓扑事件；
- 更长程 route geometry；
- 其他能够提前影响速度规划的道路约束。

这部分包含原先“road preview”的研究内容，但其角色变为 information ablation 中的一组候选变量。

需要区分：

- 仅扩大已有 `route_lanes` 的空间范围；
- 增加新的语义事件或连续预瞄特征；
- 模型实际利用信息，而不是 observation 中“存在”信息。

### I3 — Dynamic traffic information

在道路信息之外研究动态交通对节能策略的作用，例如：

- lead vehicle / following context；
- traffic density；
- interaction state；
- 可合理获得时的更长程动态约束。

研究问题是：

> 能耗收益主要由道路几何提前规划产生，还是动态交通信息同样是必要条件？

I2 与 I3 不要求固定先后；实际实验可根据场景准备情况独立开展。

## 信息消融原则

第一阶段尽量保持：

```text
reward       fixed
PPO config   fixed
encoder      frozen
scenarios    matched
seeds        matched
```

只改变 policy 可用的信息集合。

主要比较形式为：

| Information set | 研究作用 |
| --- | --- |
| I0 ego + reference | 最小 baseline |
| I1 + local scene | 当前环境信息的增量价值 |
| I2 + road/navigation preview | 长程静态结构的增量价值 |
| I3 + traffic information | 动态环境信息的增量价值 |

最终不一定需要保留所有信息。若某类信息在 matched evaluation 中没有稳定增益，应将其视为可删减变量，而不是因为理论上合理就默认加入最终模型。

## 如何证明策略实际利用信息

仅观察到“full observation 模型更好”不足以说明某类信息被利用。

至少需要一种严格对照：

- remove / mask 某类信息；
- 改变可见范围；
- 对目标信息做 matched perturbation；
- 保持当前状态和其他条件尽量一致，只改变被研究的信息。

核心是建立：

```text
information change
      ↓
policy / guidance change
      ↓
closed-loop behavior change
      ↓
energy / driving metric change
```

而不是只比较不同网络配置的最终分数。

## 第二阶段：representation adaptation

只有在第一阶段确认某类信息具有稳定增量价值后，才研究 encoder 是否需要 RL 适配。

推荐的研究顺序是：

```text
pretrained encoder, frozen
          ↓
extract encoder as explicit RL perception module
          ↓
partial fine-tuning
          ↓
optional joint fine-tuning
```

其中“提取 encoder”主要用于明确模块边界和实验控制，不等于必须立即改写感知架构。

### Frozen vs fine-tuned

最关键的表示学习消融应保持 information set、reward、PPO 和 evaluation 一致，只改变 encoder update policy：

```text
same information
same objective
same RL algorithm

frozen encoder
vs
partially fine-tuned encoder
```

由此回答：

> 预训练 Diffusion Planner 的 representation 是否已经足够支撑 energy-oriented RL，还是 long-horizon feedback 需要重新组织这些环境特征？

如果 fine-tuning 没有独立增益，则没有必要为了“端到端 RL”增加训练复杂度。

## 与多头 critic 的关系

Information study 与 multi-head critic 原则上正交。

先在固定 scalar objective 下判断信息增量价值；再在 reward study 确认 multi-head critic 有研究价值后，使用少量代表 information set 做联合验证。

不建议同时改变：

- observation；
- encoder trainability；
- reward structure；
- critic architecture。

否则难以判断收益来源。

## 本专题暂不回答

- 最终 road preview encoder 的具体网络结构；
- 具体使用多少米或多少秒的 preview；
- 是否必须加入坡度/高程等当前环境不稳定提供的信息；
- 感知 backbone 最终是否与 planner 完全共享；
- 哪种 reward 最适合能耗优化。

Reward 问题由 [`reward-objective.md`](reward-objective.md) 单独研究；联合实验见 [`ablation-plan.md`](ablation-plan.md)。
