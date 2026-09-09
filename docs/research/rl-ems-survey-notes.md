# RL-based Energy Management Survey — 对 Eco-AutoDrive 的可借鉴点

> 文档性质：primary-source research note。用于记录外部综述对当前研究问题的支持、启发与边界；不代表已接受的技术决定或 active implementation work。

## 来源

- Xiaolin Tang, Jiaxin Chen, Yechen Qin, et al., **Reinforcement Learning-Based Energy Management for Hybrid Power Systems: State-of-the-Art Survey, Review, and Perspectives**, *Chinese Journal of Mechanical Engineering*, 37, 43 (2024).
- DOI / Springer: <https://link.springer.com/article/10.1186/s10033-024-01026-4>
- Published: 2024-05-17.
- 综述统计截至 2023-07-21，共收集 266 篇 RL-based energy management 文献，并按 **algorithm innovation / powertrain innovation / environmental innovation** 三条主线整理。

本文研究对象主要是 hybrid power system 的 energy management strategy（EMS），而本项目研究的是自动驾驶扩散规划器的闭环能耗优化，因此下面区分：

- **综述直接支持的事实**：论文对 RL-EMS 文献的总结；
- **对本项目的研究推论**：这些事实如何映射到 `docs/research/README.md` 中的 reward / information / representation 研究问题。

## 与当前研究框架的总体关系

当前项目核心问题是：

> 如何利用闭环长程反馈优化自动驾驶规划器的能耗表现，同时避免安全、有效进度、平均速度和旅行效率出现不可接受的退化。

这篇综述不能直接给出 Diffusion Planner + PPO 的具体解法，但它对当前研究拆分提供了较强的外部依据：

1. **objective 不能只看能耗**：RL-based EMS 已经从单一 fuel/SOC 目标扩展到效率、寿命、温度、安全、舒适等多目标，并有工作专门研究 reward 组成与权重。
2. **未来信息具有独立研究价值**：GPS、V2V/V2I、未来速度预测、route、交通灯、traffic flow、道路识别等被用于 predictive / eco-driving EMS。
3. **速度规划与能量管理应联合理解**：部分工作已经把 reference-speed planning / ACC 与 EMS 联合优化，而不是把能耗视为与驾驶行为无关的纯动力系统问题。
4. **环境与模型保真度会改变结论**：综述强调真实工况、traffic-in-the-loop、道路属性、高保真动力学以及 sim-to-real / long-tail 问题。

因此，这篇综述更适合作为当前 Phase B–E 的**研究动机与变量来源**，而不是作为某个具体算法或 reward 公式的直接依据。

---

## 1. 对 Reward / Objective Study 的借鉴

### 1.1 多目标冲突是 EMS 中已经存在的核心问题

综述指出，RL-based EMS 的研究目标已经从传统的 fuel economy / SOC 扩展到效率、温度、寿命等多目标；同时列举了 multi-objective RL、reward-function comparison、IRL-based reward-weight determination 等工作。

这与 [`reward-objective.md`](reward-objective.md) 当前的问题高度一致：

```text
energy
vs
progress / speed / safety / comfort
```

**可借鉴点：**

- 当前先研究 scalar reward 是合理的 baseline，而不是因为 EMS 文献已经证明 scalar reward 足够；
- reward composition 与权重本身应作为独立实验变量，因为综述中已有文献表明它们会显著影响 RL-EMS 的行为；
- 如果后续发现明显 weight sensitivity 或目标冲突，再进入 multi-head critic / constrained objective，有充分的领域背景动机。

**不能从综述直接推出：**

- safety-gated reward 一定优于 weighted sum；
- 某个 energy 权重或某种归一化最优；
- multi-head critic、PPO-Lagrangian 等一定优于 scalar critic。

这些仍需要本项目 matched closed-loop experiments 自行验证。

### 1.2 安全优先不能只依赖“最终平均 reward”

综述总结的 RL-EMS safety 手段包括：

- 对不合理 action 增加 penalty；
- coach / heuristic rule；
- rule-based restriction；
- action masking。

其背景是 RL exploration 可能产生动力系统不可接受的突变或不可行动作。

对本项目而言，这可以作为当前 `safety-gated quality reward` 思路的**领域动机**：节能收益不能通过破坏基本安全或合法性获得。

但这里仅是原则层面的对应。EMS 的 action constraint 与自动驾驶中的 collision / drivable-area / wrong-direction 不是同一种约束，因此不能把 EMS 的具体安全机制直接移植为本项目实现。

### 1.3 长程反馈的动机来自 predictive energy management，但 GAE 仍需单独证明

综述中的 predictive EMS 使用：

- cumulative trip information；
- GPS/global SOC planning；
- short-term speed prediction；
- connected information；
- MPC + RL predictive control。

这些结果支持一个较弱但重要的结论：**能量管理确实可能受未来工况影响，单步局部状态未必足够。**

这与当前 `reward-objective.md` 中的 temporal credit assignment 研究方向一致。

但综述没有证明：

```text
longer GAE horizon
    ⇒
better energy optimization
```

因此当前项目仍应把“长程 GAE / return 是否提供独立增益”保留为待验证研究问题，而不能仅凭 predictive EMS 文献宣称 PPO 已经解决长程信用分配。

---

## 2. 对 Information / Representation Study 的借鉴

这是该综述对本项目最直接有价值的部分。

### 2.1 静态道路 / 导航预瞄

综述的 environmental innovation 中明确涉及：

- GPS / route information；
- lane-level map；
- road recognition；
- slope、curvature、road signs 等道路属性；
- global / local trip information。

这为 [`information-representation.md`](information-representation.md) 中的 I2 提供了较直接的领域依据：

```text
I2 — Long-range road / navigation information
```

当前 I2 候选变量：

- future curvature / curvature changes；
- speed-limit changes；
- merge / split / turn 等拓扑事件；
- longer-range route geometry；

与综述所归纳的 predictive / connected EMS 方向是一致的。

不过，综述中的很多工作优化的是 SOC / power split，而不是 diffusion trajectory planner。因此它只能说明“这些信息在节能控制中有可能有价值”，不能证明 Diffusion Planner 的 Exploration Policy 一定会利用这些信息。

### 2.2 动态交通信息

综述还总结了：

- V2V / V2I；
- surrounding vehicles；
- lead vehicle / following distance；
- traffic flow；
- traffic signal；
- car-following；
- ACC + EMS co-optimization。

这与当前 I3：

```text
I3 — Dynamic traffic information
```

直接对应。

尤其值得保留的研究区分是：

```text
road/navigation preview
vs
traffic-dependent preview
```

因为综述表明 eco-driving 不仅依赖道路几何，也依赖前车、信号灯、traffic flow 等动态约束。也就是说，“预瞄信息”不应在研究设计中被缩减为单纯的道路曲率或 route geometry。

### 2.3 “信息存在”与“策略使用信息”必须区分

综述本身主要统计“哪些输入被加入 EMS”，并没有统一证明各类输入的独立 causal contribution。

因此，本项目当前的信息消融原则反而需要比多数 EMS 文献更严格：

```text
information change
      ↓
policy / guidance change
      ↓
closed-loop behavior change
      ↓
energy / driving metric change
```

这支持继续采用：

- remove / mask；
- visibility-range ablation；
- matched perturbation；
- matched scenario / seed；

来证明信息贡献，而不是因为综述列出某类信息就默认加入最终 observation。

---

## 3. 对“速度规划 × 能耗”的借鉴

综述中一条与本项目非常相关的脉络是：**eco-driving 与 energy management 开始从分离控制走向协同优化。**

代表性总结包括：

- connected traffic 环境下，用 DDPG 规划 reference speed，再由 adaptive ECMS 做 energy management；
- 将 ACC 与 EMS 结合，通过 RL 联合优化 velocity 与 power distribution；
- car-following 中同时考虑安全距离与 comfort；
- 通过交通灯或环境信息减少频繁起停。

对本项目的意义不是要增加一个独立 EMS，而是确认 README 中的以下评价原则是必要的：

```text
energy improvement
必须与
mean speed / progress / travel efficiency
联合解释
```

因为速度本身既是能耗的主要决定因素之一，也是任务效率指标。若只优化 total energy，很容易得到“慢下来 / 少走 / 停下来”的伪收益。

因此，当前把 energy、progress、mean speed、travel efficiency 分开报告，而不是压成单一训练分数，是与 eco-driving / EMS 文献问题结构一致的。

---

## 4. 对 Hierarchical / Guidance 架构的借鉴

综述总结了一类较成熟的混合方案：RL 不直接取代所有控制模块，而是用于调节传统优化器或控制器中的关键参数，例如：

- PMP co-state；
- ECMS equivalent factor；
- MPC 中的预测 / adaptive component。

还存在 hierarchy：

```text
future / connected information
        ↓
reference speed / high-level decision
        ↓
energy management / low-level control
```

这与当前项目使用 Exploration Policy 调节 diffusion guidance、而不是从零训练完整驾驶策略，存在**架构层面的类比**：

```text
learned long-horizon policy
        ↓
modulate a pretrained / stable base controller or planner
```

这个类比可以作为“为什么研究 guidance policy 而不是立即端到端重训 Diffusion Planner”的旁证。

但需要保持结论边界：综述没有研究 diffusion guidance，也没有证明这种架构在 autonomous planning 上优于直接 fine-tuning。这里仅是控制分层思想的迁移。

---

## 5. 对 Evaluation / Scenario Design 的借鉴

### 5.1 不能只在单一标准工况评价

综述指出，标准 driving cycle 无法覆盖真实环境中的：

- traffic signals；
- traffic flow；
- pavement properties；
- weather；
- ramp / merge；
- car-following；
- driving style。

因此相关工作开始使用 featured driving cycles、real data clustering、SUMO traffic environment 和 traffic-in-the-loop simulator。

这支持本项目 README 中已有的 matched closed-loop evaluation 原则，并进一步说明后续结论应按**场景类型分层解释**，而不是只看总平均值。

与当前研究最相关的场景维度包括：

- steady / straight driving；
- curvature / turn；
- speed-limit change；
- stop-and-go / traffic signal；
- car-following；
- merge / split；
- 不同 traffic density。

这些更适合作为 energy opportunity / information-value 的实验维度，而不是直接变成训练 reward 项。

### 5.2 generalization / robustness 是独立证据层级

综述明确把 generalization、safety、robustness、sim-to-real gap 和 long-tail scenarios 视为 RL-EMS 的核心难题。

这与 README 的研究证据分层高度一致：

```text
mechanical validity
→ optimization stability
→ behavioral effect
→ energy effect
→ information / representation contribution
→ energy-model robustness
```

综述提供的启示是：即使训练 return 或单一场景 energy 下降，也不能直接解释为可泛化的节能能力。

---

## 6. 对 Energy Model Fidelity 的借鉴

综述认为当前 EMS 研究仍缺少足够高保真的动态模型，并特别提出应逐步考虑：

- engine / motor / battery / gearbox 等 component dynamics；
- vehicle longitudinal + lateral dynamics；
- road slope；
- curvature；
- road surface / friction；
- temperature / aging 等因素。

对本项目而言，这与 README 当前的 phase boundary 是一致的：

1. 先在固定 MetaDrive execution boundary 下，用 proxy energy 做 matched relative comparison；
2. 只有在出现稳定 behavioral / energy signal 后，再进入更精细 energy model 的 robustness validation。

这篇综述支持“最终需要更高 fidelity”，但并不意味着当前阶段必须立即引入完整动力系统模型。相反，如果 research question 当前是 reward / information contribution，先控制环境和能耗模型复杂度有利于因果归因。

---

## 7. 哪些内容不应直接借鉴

### 7.1 Powertrain-specific state / action

综述主体针对 HEV / PHEV / FCV 等混合动力系统，常见状态与动作包括：

- SOC；
- engine / motor power split；
- fuel-cell power；
- equivalent factor；
- battery health / temperature。

这些变量不能直接迁移到当前 trajectory-planning policy。

### 7.2 不能从综述得出“PPO 是最佳算法”

综述同时覆盖 Q-learning、DQN、DDPG、TD3、SAC、PPO、MARL 等算法，只说明不同算法在 EMS 中被采用，并未在统一设置下证明 PPO 优于其他方法。

因此它最多为当前 PPO 路线提供“领域中已有 on-policy application”的背景，不构成算法选择证据。当前 PPO 的主要依据仍应来自 PlannerRFT / DPPO 和项目自身训练证据。

### 7.3 “Alpha HEV”更接近长期愿景

文章最后提出将 Autopilot 与 energy-saving control 统一到 “Alpha HEV” 的愿景，其中包含更高保真车辆建模、多模态感知和 integrated control。

这与本项目的长期目标方向一致，但抽象层次过高，不能直接转化为当前 Phase B / C 的实验设计。

---

## 8. 对当前 research roadmap 的具体支持关系

| 当前阶段 | 综述可提供的支持 | 结论边界 |
| --- | --- | --- |
| Phase B — Scalar reward | reward composition、multi-objective conflict、reward-weight sensitivity 在 EMS 中均是实际问题 | 不提供最终 reward 公式或权重 |
| Phase B — Temporal credit | predictive EMS、trip/GPS/speed prediction 表明未来工况可能影响节能决策 | 不证明 GAE horizon 越长越好 |
| Phase C — Information | route/GPS、road recognition、traffic、V2X、lead vehicle、signal 等都是已研究信息源 | 不证明 Diffusion Planner 会实际使用 |
| Phase D — Representation | 综述强调多源信息融合与 context adaptation | 不回答 frozen pretrained encoder 是否足够 |
| Phase D — Multi-objective | 多目标 RL、IRL reward weighting、安全约束提供进一步研究动机 | 不证明 multi-head critic 是最优实现 |
| Phase E — Joint ablations | real/featured cycles 与多环境测试支持 matched scenario-stratified evaluation | 仍需项目内严格 factorial / matched ablation |
| Energy-model robustness | 高保真动力学、道路属性被认为是实际部署的重要缺口 | 适合后续 robustness stage，不必提前阻塞当前研究 |

## 结论

这篇综述对 Eco-AutoDrive 最值得借鉴的不是某一种 RL 算法，而是三个研究判断：

1. **节能 objective 天然是多目标 trade-off 问题**，必须独立观察 energy、speed/progress、safety、comfort，而不能只看 aggregate return；
2. **未来道路、route、traffic 与速度信息是有根据的候选节能信息源**，但其价值必须通过严格 information ablation 验证；
3. **速度规划、环境信息和能耗优化具有强耦合**，因此 long-horizon closed-loop evaluation 比静态单步 energy score 更符合问题本身。

它总体上强化了当前 README 已经采用的两条主线：

```text
RQ1: optimization objective
RQ2: useful information
```

并支持“先得到稳定 behavioral / energy signal，再增加 multi-head critic、representation fine-tuning 和更高 fidelity energy model”的分阶段研究逻辑，而不是要求立即扩大系统复杂度。
