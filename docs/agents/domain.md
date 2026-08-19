# Domain guidance for agents

本文件只保留**容易导致实现或实验解释错误的领域语义区分**。完整术语定义以 [`CONTEXT.md`](../../CONTEXT.md) 为准；shape、时间频率、算法和运行时行为以 [`system-contract.md`](system-contract.md) 为准。

不要把本文件扩展成第二份 glossary 或 system contract。只有某个概念即使名称正确仍很容易被智能体误解时，才在这里记录区分规则。

## 何时读取

涉及以下任务时优先读取本文件：

- evaluation 与 policy-guided rollout 的时间/MDP 语义；
- guidance、reference trajectory 或 Exploration Policy；
- seed、随机性和策略配对比较；
- 能耗指标与实验结论；
- 道路预瞄、route information 与尚未实现的研究设想。

需要精确定义时再读取 `CONTEXT.md`；需要当前实现细节时只读取 `system-contract.md` 的相关章节。

## 必须保持的语义区分

### 场景、回合与作业不是同一层级

- **scenario** 描述地图、交通条件和场景随机状态。
- **episode** 是一个 scenario 从 reset 到终止/截断/运行错误的单次闭环运行。
- **evaluation job** 由一份 resolved config、一个运行时和一个独立产物目录组成，可以顺序包含多个 episode。

不要用“场景结果”代替 episode 结果，也不要把 job 级配置或失败状态错误地下沉到单个 scenario 定义中。

### Evaluation planning cycle 与 rollout transition 不可互换

现有 evaluation 每次规划后执行预测轨迹前 `0.5 s`；policy-guided rollout 的一个 MDP transition 只执行第一个 `0.1 s` 点。两条路径的 reward、done、bootstrap 和时间索引语义不同。

任何修改涉及 horizon、transition、GAE、reward、终止或 trace 索引时，都必须先确认自己处于哪条路径。

### 随机种子属于不同命名空间

`map seed`、diffusion `noise seed` 和 `policy action seed` 是不同实验变量。策略比较需要按实验设计配对相应 seed；不要用一个笼统的“seed”字段或全局 RNG 替代它们。

相同 seed 也不自动意味着跨设备、跨精度逐位一致。

### Reference trajectory 不是道路中心线或专家轨迹

reference trajectory 是同一规划周期内由冻结 planner 在共享观测和扩散随机流下生成的预测。它不是：

- lane/route centerline；
- expert / ground-truth trajectory；
- safety fallback；
- 失败时可切换的替代控制器。

因此 reference-centered guidance 的“reference”不能被解释为道路几何约束或专家监督。

### Base action 与 guidance action 不同

Exploration Policy 保存的 `base action u` 位于 `(0,1)^2`；实际审计和执行的 `guidance action g` 由 `g = 2u - 1` 得到，位于 `(-1,1)^2`。概率、log-prob 和 entropy 的记账必须使用对应空间的定义，不能把二者混为一个动作字段。

### 运动学执行不代表低层动力学可执行

当前闭环把规划轨迹写入车辆状态，用于隔离和研究规划行为。它不生成 steering、throttle 或 brake，也不证明轨迹在真实或高保真车辆动力学下可跟踪。

因此安全、舒适性或能耗结论必须明确限定在当前 execution condition，除非另有低层动力学实验支持。

### Proxy energy 与 high-fidelity energy 不可混用

MetaDrive 中的代理能耗指标只支持固定仿真和车辆条件下的相对比较。FASTSim 等精细模型属于不同指标与累计边界；结果必须分别命名、分别记录，不能静默合并为“真实能耗”。

### 当前 route conditioning 不等于道路预瞄研究已完成

当前 planner 已消费局部 `route_lanes`，但“更长程或语义更明确的 road preview information”仍是独立研究问题。已有 route conditioning、PPO 基础设施或 guidance 能力都不能被描述为已经证明道路预瞄能改善能耗。

### 已实现能力不等于研究结论

仓库中存在 Exploration Policy、PPO、guidance 或某种 reward，只能说明对应实验链路已经实现。除非有实际实验记录支持，不得据此声称：

- PPO 是最终采用的方法；
- 当前 reward 是最终能耗目标；
- 当前实现与论文未公开细节达到 parity；
- 当前方法已经改善能耗。

研究结论只能由 `docs/experiments/` 中实际运行证据支持；尚未回答的问题留在 `docs/research/`。

## 术语维护规则

- 使用 `CONTEXT.md` 中的规范名称；近义词若会改变语义，不要为了文风替换。
- 新的稳定领域术语写入 `CONTEXT.md`，而不是在本文件永久定义。
- 实验 ID、Issue 标签、临时阶段名和历史 artifact-format 标签默认不是稳定领域词汇。
- 若拟议术语或行为与现有 ADR 冲突，显式指出冲突；不要在文档中静默重新定义。
- 当前实现行为变化时更新 `system-contract.md`；研究设想变化时更新 `docs/research/`。不要在本文件复制两者的细节。
