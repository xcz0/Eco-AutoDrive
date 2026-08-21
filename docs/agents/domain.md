# Domain guidance for agents

本文件只保留**名称本身不足以避免误解、且容易导致实现或实验解释错误的领域语义 gotchas**。稳定术语的精确定义见 [`CONTEXT.md`](../../CONTEXT.md)；shape、时间频率、算法和运行时行为见 [`system-contract.md`](system-contract.md)。

涉及 evaluation/rollout 时间语义、guidance/reference、随机种子、能耗、道路预瞄或研究结论时优先读取本文件。不要把它扩展成第二份 glossary 或 system contract。

## 高风险语义区分

### Scenario、episode 与 evaluation job 是不同层级

- **scenario**：地图、交通条件和场景随机状态的定义。
- **episode**：一个 scenario 从 reset 到终止、截断或运行错误的单次闭环运行。
- **evaluation job**：一份 resolved config、一个运行时和一个独立产物目录；可顺序包含多个 episode。

不要把 job 级配置、失败状态或实验结论下沉成 scenario 定义，也不要用“场景结果”含糊代替 episode 结果。

### Evaluation planning cycle 与 rollout transition 不可互换

现有 evaluation 每次规划后执行预测轨迹前 `0.5 s`；policy-guided rollout 的一个 MDP transition 只执行第一个 `0.1 s` 点。

reward、done、bootstrap、GAE 和 trace 索引依赖这一区分。涉及 horizon、transition 或终止语义时，先确认自己处于哪条路径。

### 随机种子属于不同命名空间

`map seed`、diffusion `noise seed` 和 `policy action seed` 是不同实验变量。策略比较应按实验设计配对相应随机流；不要用一个笼统的全局 `seed` 或共享 RNG 替代它们。

相同 seed 也不自动保证跨设备、跨 precision 逐位一致。

### Reference trajectory 不是中心线、专家轨迹或 fallback

reference trajectory 是同一规划周期内冻结 planner 在共享观测和扩散随机流下生成的预测。它不是：

- lane / route centerline；
- expert / ground-truth trajectory；
- safety fallback 或失败后的替代控制器。

因此 reference-centered guidance 不能被解释为道路几何约束或专家监督。

### Base action 与 guidance action 不是同一空间

Exploration Policy 保存的 `base action u` 位于 `(0,1)^2`；实际 guidance action 为 `g = 2u - 1`，位于 `(-1,1)^2`。概率、log-prob、entropy 和 replay 记录必须使用各自对应的动作空间定义。

### 运动学执行、proxy energy 与 high-fidelity energy 必须分开解释

当前闭环直接写入规划轨迹点，不生成 steering、throttle 或 brake。它适合隔离规划行为，但不证明轨迹在真实或高保真车辆动力学下可跟踪。

MetaDrive proxy energy 只支持固定仿真和车辆条件下的相对比较；FASTSim 等精细车辆模型属于另一指标与累计边界。二者必须分别命名、分别记录，不能合并成“真实能耗”。

### 已有 route conditioning、guidance 或 PPO 不等于研究问题已经解决

当前 planner 已消费局部 `route_lanes`，但更长程或语义更明确的 road preview 仍是独立研究问题。仓库中存在 Exploration Policy、PPO、guidance 或某种 reward，也只说明对应链路已经实现。

没有实际实验记录支持时，不得声称：

- PPO 是最终采用的方法；
- 当前 reward 是最终 energy-oriented objective；
- 当前实现与论文未公开细节达到 parity；
- 新增 preview 已被模型有效利用；
- 当前方法已经改善能耗。

研究结论只能由 `docs/experiments/` 中实际运行证据支持；尚未回答的问题留在 `docs/research/`。

## 维护规则

- 稳定术语的新定义或定义变化写入 `CONTEXT.md`。
- 只有某个概念即使名称正确仍容易被误解时，才在本文件增加 gotcha。
- 当前实现行为变化写入 `system-contract.md`；研究设想变化写入 `docs/research/`；实验结论写入 `docs/experiments/`。
- 实验 ID、Issue 标签、临时阶段名和历史 artifact-format 标签默认不是稳定领域术语。
- 若拟议术语或行为与现有 ADR 冲突，显式指出，不要静默重新定义。
