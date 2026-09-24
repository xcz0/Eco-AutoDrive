# Reward and Objective Study

> 类型：Hypothesis。以下是待验证的候选方法和消融设想，不是 active protocol 或实施清单。
> 已有结论见 [Findings](../findings.md)；已接受的训练方法见
> [training protocol](../protocols/training.md)，其他规范按 [AGENTS](../../../AGENTS.md) 路由。

## 研究问题

本专题研究：

> **如何构造长程优化 objective，使策略获得可重复的能耗收益，同时避免安全、有效进度、平均速度和旅行效率出现不可接受的退化？**

当前阶段先研究单一标量 reward。多头 critic、约束式强化学习和其他 multi-objective 方法属于后续扩展，只有在 scalar reward 已经产生可解释行为信号后再进入。

## 证据入口与待回答问题

[Findings](../findings.md) 分开整理 objective 可辨识性、有效更新、行为方向和执行时间语义。
这里继续研究：这些条件满足后，scalar objective 能否带来跨 seed、场景可重复的节能收益？
已接受的 canonical-cadence transfer 与 λ 实验由
[Issue #83](https://github.com/xcz0/Eco-AutoDrive/issues/83) 拥有，不在本篇维护执行状态。

## 研究变量

### 1. Reward structure

第一阶段不预设唯一正确的 aggregate score，而比较少量结构清晰的标量 reward。

基本候选包括：

1. **加权和**

   ```text
   safety + progress + speed + comfort - energy cost
   ```

   用作最简单、最容易解释的 baseline。

2. **safety-gated quality reward**

   ```text
   safety / legality gates
             ×
   progress + speed + comfort + energy quality
   ```

   项目已接受 reward 的定义见 [training protocol](../protocols/training.md)。
   本候选结构的研究动机是避免“先牺牲基本驾驶合法性，再交换 energy score”，但仍需要实验验证其 trade-off。

3. **reference-relative objective**

   将当前策略的能耗或行为与 frozen reference planner 的同场景表现比较，而不是只依赖绝对 reward scale。

   这一形式目前只是候选研究变量，不预设为最终实现。

研究重点不是穷举 reward 公式，而是识别哪些结构能够产生稳定、可解释的优化方向。

### 2. Energy signal definition

能耗本身的归一化方式可能决定策略学到的是“节能”还是“少走”。需要比较的主要类别是：

- total energy；
- distance-normalized energy intensity；
- progress-normalized energy；
- relative-to-reference energy change。

这些指标必须与 completion、distance、progress、speed 和 termination type 联合解释。

Native energy 与执行 trace proxy 的已观测差异见 [Findings](../findings.md#能耗读数能否代表实际执行)。
候选归一化方案不改变该证据的仿真与 proxy 解释边界。

### 3. Trade-off weight

对于标量 reward，需要研究 energy 权重增加时其他指标如何变化，而不是只寻找 training return 最大的配置。

主要关心：

```text
energy weight ↑
      |
      +--> energy
      +--> progress
      +--> mean speed / travel efficiency
      +--> safety
      +--> comfort
```

目标是得到一个可解释的 trade-off 区域，而不是预先规定唯一权重。

### 4. Temporal credit assignment

本项目关注长程反馈，因此需要区分：

- 当前/短程 reward；
- 多步 discounted return；
- GAE 所形成的更长程 advantage。

研究问题是：

> **能耗优化是否真的依赖更长的 temporal credit assignment？**

若较长 horizon 不能提供独立收益，则“长程反馈”不应仅凭 PPO 的存在被宣称有效。

具体 horizon 数值和训练预算应在对应实验 Issue 中确定，本研究文档不提前固定。

## 主要消融逻辑

Reward study 应尽量保持 observation、optimizer 和 evaluation 不变，只改变 objective。

推荐按以下顺序收敛：

```text
R0  非能耗 baseline objective
 |
R1  当前 scalar energy objective
 |
R2  改变 energy normalization / composition
 |
R3  energy weight trade-off
 |
R4  temporal credit horizon ablation
```

这里的 `R0...R4` 是研究角色，不是固定配置名。

每一阶段只有在前一阶段已经出现 measurable behavioral effect 后才需要扩大训练或增加公式复杂度。

## 评价与判据

已接受的 matched comparison、指标与失败解释统一引用
[planning/evaluation protocol](../protocols/planning-and-evaluation.md)。
本专题尚待确定的 trade-off / non-inferiority 判据见 [候选消融设计](ablation-plan.md#候选结果判据)。

对 reward 的评价优先关注方向一致性与 trade-off，而不是过早定义单一“通过阈值”。

## 从 scalar reward 到 multi-head critic

多头 critic 是本专题的自然第二阶段，而不是当前 baseline 的前置条件。

概念上可从：

```text
single scalar reward
        ↓
single V(s)
```

扩展为：

```text
energy return      -> V_energy(s)
safety return      -> V_safety(s)
progress return    -> V_progress(s)
speed return       -> V_speed(s)
comfort return     -> V_comfort(s)
```

随后再研究如何将多个 advantage 用于 policy update，例如：

- weighted multi-objective advantage；
- safety / speed constraints + energy objective；
- Lagrangian 或其他 constrained formulation。

是否需要这些方法，应由 scalar reward 的实验结果决定：

- 如果 scalar reward 已能形成稳定且可接受的 trade-off，多头 critic 主要用于提高可解释性或鲁棒性；
- 如果 scalar reward 持续出现目标冲突或 reward-weight sensitivity，多头 critic 才成为更强的研究动机。

若进入该研究，还需比较多个 value heads 是否共享 trunk、actor/critic 是否共享 representation、
各 objective 的 target scale 与梯度竞争，以及 advantage aggregation / constraint bookkeeping。
这些承接旧 PPO support plan D 与 G-P4，仍是未接受的设计问题，不预设网络结构。

为验证候选 reward / credit horizon，可能需要扩展明确的 reward profiles、component/raw metric
审计与 return ablation 配置。是否存在具体支持缺口应在实验设计确定后判断，不据旧 support plan A
重复实现已有机制，也不在这里维护工具能力清单。

## 本专题暂不回答

- 哪一个多头 critic 结构最终最好；
- 最终真实车辆 energy model；
- GRPO 或直接微调 DiT 的 reward 设计；
- 哪些 road / traffic information 应加入 observation。

后一个问题由 [`information-representation.md`](information-representation.md) 独立研究。
