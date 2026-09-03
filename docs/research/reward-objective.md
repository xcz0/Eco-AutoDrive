# Reward and Objective Study

## 研究问题

本专题研究：

> **如何构造长程优化 objective，使策略获得可重复的能耗收益，同时避免安全、有效进度、平均速度和旅行效率出现不可接受的退化？**

当前阶段先研究单一标量 reward。多头 critic、约束式强化学习和其他 multi-objective 方法属于后续扩展，只有在 scalar reward 已经产生可解释行为信号后再进入。

## 当前起点

项目已经具备：

- closed-loop rollout + GAE/PPO；
- `plannerrft_energy_v1` 单一已实现 reward profile，reward 由父进程 collector 单次计算；
- execution-trace proxy energy；
- 已完成的历史 matched reward A/B 实验记录；
- 受限 no-traffic S/SC 条件下的 PPO 稳定候选配置。

现有证据只说明 reward 可以正确进入 PPO 数据流，并且短趋势 A/B 没有观察到明显 reward hacking；尚没有证据证明当前 energy reward 能够产生 learned-policy energy improvement。

相关实验：

- [`E-019`](../experiments/records/e-019-metadrive-native-energy-proxy-comparison.md)
- [`E-026`](../experiments/records/e-026-issue59-stage-b-ppo-reward-ab-short-trend.md)
- [`E-028`](../experiments/records/e-028-issue76-ppo-stability-search.md)

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

   当前 `plannerrft_energy_v1` 属于这一类扩展。该结构适合避免“先牺牲基本驾驶合法性，再交换 energy score”的情况，但仍需要实验验证其 trade-off。

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

当前 MetaDrive native energy 在 kinematic waypoint execution 下不可直接使用；研究仍以 execution boundary 重算的 proxy energy 为主。

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

Reward study 的主要结果来自独立闭环 evaluation，不以 training return 排名。

至少报告：

- energy / energy intensity；
- route progress / completion；
- mean speed / travel efficiency；
- collision / out-of-road / wrong-direction；
- comfort（若该实验涉及）。

以下情况不能被解释为 energy improvement：

- energy 降低但有效距离或 route progress 明显降低；
- energy 降低主要来自停车或显著降速；
- energy 降低同时 collision / out-of-road 增加；
- 只有单个 seed 或少量特定场景出现改善。

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

## 本专题暂不回答

- 哪一个多头 critic 结构最终最好；
- 最终真实车辆 energy model；
- GRPO 或直接微调 DiT 的 reward 设计；
- 哪些 road / traffic information 应加入 observation。

后一个问题由 [`information-representation.md`](information-representation.md) 独立研究。
