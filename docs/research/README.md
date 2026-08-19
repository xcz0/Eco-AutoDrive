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

## 研究目标

研究如何在保留预训练 Diffusion Planner 驾驶能力的前提下，引入道路预瞄信息，使规划器能够利用更长程的道路和导航信息改善能耗表现，同时保持安全、有效进度和旅行效率。

当前研究重点包括：

1. 道路预瞄信息应包含什么，以及如何表示和注入 Diffusion Planner；
2. 如何评价预瞄信息是否真正改善能耗，而不是通过停车、降低有效进度或增加失败率取得较低能耗；
3. 是否需要强化学习，以及如果需要，应采用何种策略优化方式；
4. MetaDrive 中的代理能耗指标与更精细车辆能耗模型之间如何建立可信的比较关系。

这些问题仍属于研究问题，不构成当前系统契约。

## 专题研究

| 专题 | 当前研究状态 | 入口 |
| --- | --- | --- |
| PlannerRFT PPO-only | PPO、policy-guided rollout 和 MetaDrive closed-loop smoke training 基础设施已经实现；是否最终使用 PPO 作为能耗优化方法仍未决定 | [PlannerRFT PPO-only 研究](plannerrft-ppo/README.md) |

PlannerRFT/PPO 基础设施的存在只说明该路线已经具备实验条件，不表示 PPO 已被选为本项目最终的能耗优化方法。

## 道路预瞄

候选预瞄信息包括：

- 前方道路曲率及其变化；
- 限速及限速变化位置；
- 变道、合流、分流和道路拓扑；
- 导航路线中的长程结构；
- 交通状态；
- 数据确实存在时的坡度或高程。

需要区分：

- 当前 Diffusion Planner 已经接收的局部 `route_lanes`；
- 新增的、更长程或语义更明确的 preview information。

仍待研究的问题包括：

- 预瞄距离；
- 空间或时间采样方法；
- 连续几何与离散道路事件如何共同表示；
- 是否使用独立 encoder、route token 扩展、条件残差或其他条件注入方式；
- 模型是否已经能够从现有 route information 中提取足够的长程信息。

## 策略优化

是否需要强化学习仍是开放问题。

当前可研究的路线包括：

- PlannerRFT 风格的 Exploration Policy + PPO；
- 其他基于环境 MDP 的 diffusion-policy optimization；
- 只训练新增的 preview module；
- 参数高效微调；
- 监督式辅助目标；
- 推理期 guidance 或代价引导。

仓库已经实现 PlannerRFT PPO-only 所需的主要实验基础设施，包括 Exploration Policy、closed-loop rollout、GAE/PPO update 和 MetaDrive smoke training。

这些能力属于当前实现事实，具体契约见
[`system-contract.md`](../agents/system-contract.md)。

它们不能被解释为以下研究结论：

- PPO 已被证明优于非 RL 方法；
- 当前 MetaDrive reward 是最终能耗优化目标；
- PlannerRFT 的论文 reward 已在 MetaDrive 中实现 parity；
- 当前 PPO 路线已经证明能够改善能耗。

这些问题仍需要实验回答。

## 能耗评价

MetaDrive 中可以使用代理能耗指标进行快速闭环比较，但代理指标与真实车辆能耗并不等价。

候选的研究结构是：

```text
closed-loop driving
        |
        +--> proxy energy metric
        |
        +--> executed speed / acceleration trace
                    |
                    +--> FASTSim or another detailed model
````

仍待确定：

* 研究对象是燃油车、电动车还是一般化车辆；
* 车辆参数；
* 主能耗指标；
* 总能耗、单位距离能耗和单位有效进度能耗之间的取舍；
* 不同场景长度和 termination type 的可比性；
* proxy metric 与 detailed model 排序不一致时的解释方式。

## 低层动力学

当前主要研究仍基于运动学轨迹执行。

后续若研究 steering/throttle/brake、车辆动力学或轨迹可跟踪性，应作为独立研究阶段处理。运动学闭环中的能耗或安全结论不能自动外推为低层动力学结论。

## 开放问题

当前主要开放问题为：

* 道路预瞄应包含哪些信息？
* 预瞄距离应多长？
* 预瞄信息应如何编码并注入 Diffusion Planner？
* 现有 route conditioning 是否已经包含足够长程信息？
* 如何证明模型确实利用了新增 preview，而不是由其他输入产生相同行为？
* 是否需要强化学习来利用预瞄信息？
* 如果使用 PPO，优化对象应继续是 Exploration Policy，还是未来考虑其他参数子集？
* 当前 smoke reward 与最终 energy-oriented objective 应如何区分？
* 如何同时约束能耗、安全、有效进度、旅行时间和舒适性？
* 什么能耗指标适合作为主要比较指标？
* MetaDrive proxy energy 与精细车辆模型之间应如何交叉验证？
* 运动学研究结果如何与后续低层控制和动力学研究衔接？

可执行工作一旦被接受，不继续在本文件中维护任务状态，而转入 GitHub Issue。
