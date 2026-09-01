# 研究空间

本目录只保存尚未成为系统事实、正式决定或 active implementation work 的研究内容。

文档职责：

- 当前已实现行为 → [`docs/agents/system-contract.md`](../agents/system-contract.md)
- 已接受的技术决定及理由 → [`docs/adr/`](../adr/)
- 可执行开发工作 → GitHub Issues
- 实验运行证据 → [`docs/experiments/`](../experiments/)
- 外部论文和代码事实 → 对应 primary-source notes
- 尚未解决的研究问题 → 本目录

研究文档不复制当前实现契约，也不使用路线图阶段描述代码是否已经实现。

## 核心研究目标

本项目研究：

> 在尽量保留预训练 Diffusion Planner 原有驾驶能力的前提下，引入当前局部 route conditioning 之外的道路预瞄信息，使规划器能够利用更长程的道路、导航和交通结构改善闭环能耗表现，同时维持安全、有效进度和旅行效率。

因此，项目最终需要回答的不是“能否把 PPO 跑起来”或“能否定义一个 energy reward”，而是以下三个连续问题：

1. **现有信息是否不足**：当前 Diffusion Planner 已有的 lane / route conditioning 对更长程道路结构到底利用到什么程度；
2. **新增预瞄是否有增量价值**：加入明确的 preview information 后，模型是否真的利用了这些信息，并产生可重复的节能行为变化；
3. **如何优化这种能力**：PPO learned guidance、固定 guidance、监督式条件学习或其他优化方式中，哪种方法最适合把预瞄信息转化为长程能耗收益。

当前研究不把某一种强化学习算法预设为最终方案。

## 当前研究基线

截至 2026-09-01，仓库已经具备开展上述研究所需的主要闭环实验基础，但尚未得到“道路预瞄能够降低能耗”的研究结论。

已经建立的基础包括：

- MetaDrive 闭环执行、固定场景/seed 对照和可追溯 evaluation artifact；
- 预训练 Diffusion Planner 的冻结推理、reference trajectory 和 orthogonal guidance；
- Exploration Policy、closed-loop rollout、GAE/PPO update 和训练产物；
- 稳定能耗场景矩阵与代理能耗指标的基本比较；
- `plannerrft_energy_v1` energy-oriented PPO reward 及严格匹配的短程 reward A/B 基础设施；
- PPO 稳定性搜索工作流及一个在当前受限实验条件下通过多 seed 长程验证的稳定候选配置。

重要实验边界：

- E-026 中观察到的 PPO out-of-road 退化后来被确认主要来自 MetaDrive reset 语义 bug，而不能作为“PPO update 过强”的证据；修复后的结论见 E-028。
- E-028 仅证明在固定 `metadrive_builtin_v1`、no-traffic S/SC 场景和当前训练规模下存在可用的 PPO 稳定基线；它不证明 energy reward 有效，也不能直接外推到有交通、更长训练或其他 reward profile。
- Issue #59 的 builtin / energy 短趋势 A/B 没有给出可解释的节能差异，因此当前仍没有 learned-policy energy improvement 的证据。
- 当前还没有形成经过实验验证的 preview-conditioned planner，因此“预瞄是否有效”仍是项目的核心未决问题。

相关证据：

- [`E-019 MetaDrive native energy proxy comparison`](../experiments/records/e-019-metadrive-native-energy-proxy-comparison.md)
- [`E-026 PPO reward A/B short trend`](../experiments/records/e-026-issue59-stage-b-ppo-reward-ab-short-trend.md)
- [`E-028 PPO stability search`](../experiments/records/e-028-issue76-ppo-stability-search.md)

## 当前研究主线

现阶段研究应围绕一条统一主线组织：

```text
现有 Diffusion Planner
        |
        v
确认当前 route conditioning 的有效信息范围
        |
        v
定义额外 road preview information
        |
        v
构造 preview-conditioned planning / guidance
        |
        v
在稳定闭环训练与评测基线上优化
        |
        v
比较 no-preview / preview 的能耗、安全、进度与速度
        |
        v
判断 preview 是否带来独立的长程能耗收益
```

这里的重点是验证 **preview 的增量贡献**，而不是单独证明某个 reward、某个 PPO 超参数或某个模型模块能够运行。

## 道路预瞄

道路预瞄应定义为：**当前 planner 原始局部 lane / route observation 之外，能够明确描述更远未来道路状态或事件的条件信息**。

候选信息包括：

- 前方道路曲率及其变化；
- 限速及限速变化位置；
- 变道、合流、分流和道路拓扑；
- 导航路线中的长程结构；
- 前方交通状态或交通约束；
- 数据确实存在时的坡度或高程。

需要严格区分：

- 当前 Diffusion Planner 已经接收的 `lanes` 与 `route_lanes`；
- 仅仅扩大现有 route information 的空间范围；
- 新增的、更长程或语义更明确的 preview representation。

因此，在设计新的 preview encoder 之前，需要先回答现有 route conditioning 是否已经包含、以及模型是否已经实际利用足够的长程信息。否则新增特征可能只是重复输入，而不是新的研究变量。

主要开放问题包括：

- 现有 route conditioning 的有效利用距离有多长；
- preview 应使用距离范围还是时间范围定义；
- 连续几何与离散道路事件应如何共同表示；
- 曲率、限速、拓扑和交通信息中哪些对能耗具有真正的增量贡献；
- preview 应作为独立条件、route representation 扩展，还是 guidance / policy 的输入；
- 如何证明模型确实利用了 preview，而不是由当前状态或已有 route information 产生相同行为。

## 策略优化

PlannerRFT PPO-only 目前应视为一个**已经具备稳定实验基线的候选优化方法**，而不再只是实现可行性问题。

其研究入口见：

- [PlannerRFT PPO-only 研究](plannerrft-ppo/README.md)

当前证据支持：

- Exploration Policy + PPO 的完整闭环训练链路能够运行；
- 在受限 no-traffic 场景中可以找到跨 seed、100 updates 稳定的 PPO 配置；
- 过多 epochs 与较高 learning rate 可能推动 guidance Beta distribution 向边界塌缩。

当前证据仍不支持：

- PPO 能够改善能耗；
- `plannerrft_energy_v1` 优于 `metadrive_builtin_v1`；
- PPO 比 fixed guidance、training-free guidance 或非 RL 方法更适合利用 preview；
- 当前 PPO 配置能够直接迁移到 preview-conditioned training。

因此后续策略优化研究的重点应从“训练是否能跑通”转向：

1. 在稳定训练条件下，energy-oriented objective 是否真的改变 learned policy 的闭环行为；
2. 在同样的优化方法下，加入 preview 是否比 no-preview 带来额外收益；
3. 若存在收益，再比较 PPO learned guidance 与更简单的 fixed / non-RL baseline，判断强化学习是否是必要因素。

GRPO、直接微调 DiT、PPO-Lagrangian、多 critic 等仍属于可能的扩展方向，但当前没有必要把它们预设为主线。

## 能耗评价

当前能耗研究的首要目标是得到**可信的相对比较**，而不是把 MetaDrive proxy 解释为真实车辆能耗。

当前研究结构为：

```text
closed-loop driving
        |
        +--> proxy energy metric
        |
        +--> executed trajectory / speed trace
                    |
                    +--> optional high-fidelity energy model
```

已经明确：

- MetaDrive native `step_energy / episode_energy` 与当前运动学 waypoint execution 的 phase boundary 不匹配，不能直接作为主要 reward / metric；
- 当前 energy-oriented reward 使用 execution boundary 上重算的 MetaDrive fuel proxy；
- proxy metric 只用于固定仿真条件下的相对比较，不代表真实燃油或电池能耗；
- 低能耗不能通过停车、明显降速、减少有效进度或增加失败率来解释为优化成功。

仍待研究：

- 哪个 energy intensity / progress-normalized 指标最适合作为主要比较量；
- energy、平均速度、有效进度和旅行时间之间的 trade-off 应如何报告；
- 不同 termination type 下哪些场景可以进行能耗比较；
- proxy metric 与 FASTSim 等精细模型的排序是否一致；
- 在加入 preview 后，节能变化是否来自真正的提前速度规划，而不是短期行为偏移。

## 研究证据层级

为避免把工程可运行性误写成研究结论，本项目区分以下证据层级：

1. **mechanical validity**：训练、rollout、artifact、reward 和梯度链路能够正确运行；
2. **optimization stability**：策略在足够训练长度和多 seed 下不出现明显数值或行为崩溃；
3. **behavioral effect**：learned policy 相比初始策略产生稳定、可解释的闭环行为变化；
4. **energy effect**：在 reward-independent、matched evaluation 中能耗指标改善，同时安全、进度和速度不出现不可接受退化；
5. **preview contribution**：preview 与 no-preview 的严格对照表明收益确实来自新增预瞄信息；
6. **energy-model robustness**：proxy 下的主要结论在更精细能耗模型中仍保持方向一致。

当前项目已经较好覆盖第 1 层，并在受限 PPO 配置上获得第 2 层证据；第 3–6 层仍是后续研究对象。

## 低层动力学

当前主要研究仍基于运动学轨迹执行。

后续若研究 steering / throttle / brake、车辆动力学或轨迹可跟踪性，应作为独立研究问题处理。运动学闭环中的能耗或安全结论不能自动外推为低层动力学结论。

## 当前开放问题

当前主要开放问题收敛为：

- 当前 Diffusion Planner 实际利用现有 `route_lanes` 的距离和语义范围有多大？
- 哪些 road preview 信息对长程能耗最有可能提供现有输入之外的增量信息？
- preview 应如何表示，才能与现有 lane / route conditioning 明确区分？
- 在已经稳定的 PPO 基线上，energy-oriented reward 是否能够产生可重复的 learned-policy 行为变化？
- 在相同 reward 与优化配置下，preview 是否优于 no-preview？
- 若 preview 有效，PPO learned guidance 是否优于 fixed guidance 或更简单的非 RL 方法？
- 如何在降低能耗时同时保持安全、有效进度、平均速度和旅行效率？
- 如何证明节能来自提前利用道路信息，而不是停车、降速或其他 reward shortcut？
- MetaDrive proxy energy 与更精细车辆能耗模型之间是否保持一致的策略排序？
- 运动学研究结果在多交通场景和后续车辆动力学条件下能保留多少？

可执行工作一旦被接受，不继续在本文件中维护任务状态，而转入 GitHub Issue。