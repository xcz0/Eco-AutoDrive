# 领域语义易错点

本文件只保留会导致实现或实验解释错误的概念区分。精确定义归 [CONTEXT.md](../../CONTEXT.md)，shape、公式和运行时参数归 [system contract](system-contract.md) 及其分篇。

## Scenario、episode 与 evaluation job

Scenario 定义场景条件，episode 是单次闭环运行，job 是配置、运行时与产物的执行单位。job 级失败或结论不能下沉为 scenario 定义；报告应明确结果属于哪个 episode，避免用“场景结果”混淆层级。

## Evaluation cycle 与 rollout transition

两条路径执行的轨迹前缀不同，reward、done、bootstrap、GAE 和 trace 索引必须按所在路径解释。即使使用同一 policy checkpoint，evaluation 也不沿用训练 transition 的时间尺度。精确频率见[坐标、时间与单位](system-contract.md#坐标时间与单位)。

## 随机种子的命名空间

Map seed、diffusion noise seed 和 policy action seed 是独立实验变量。策略比较按实验设计配对相应随机流，不能用笼统的全局 seed 或共享 RNG 替代。相同 seed 不自动保证跨设备、跨 precision 逐位一致。

## Reference 与 guidance action

Reference 是同周期冻结 planner 在共享观测与扩散随机流下生成的预测，不能解释为中心线、专家/真值轨迹或安全 fallback。因此 reference-centered guidance 本身不构成道路几何约束或专家监督。

Policy 的 base action 与仿射变换后的 guidance action 属于不同概率空间；log-prob、entropy 和 replay 必须与所用空间一致。固定 guidance/人工干预与 Beta policy sampling 的边界值约束也不同，见[规划 guidance](contracts/planner.md#reference-planner-与正交-guidance)和[Exploration Policy](contracts/training.md#exploration-policy)。

## 运动学执行与能耗结论

直接写入轨迹点的运动学闭环只能隔离规划行为，不能证明 steering、throttle、brake 控制下的动力学可跟踪性。

MetaDrive proxy energy 只支持固定仿真和车辆条件下的相对比较；FASTSim 等精细模型有独立指标和累计边界。二者分别命名和记录，不能混为“真实能耗”；具体流见[能耗记录](system-contract.md#能耗记录)。

## 实现状态与研究结论

局部 route conditioning 已存在，不代表更长程或语义更明确的 road preview 已解决。Exploration Policy、PPO、guidance 和 reward 链路已实现，也不证明：

- PPO 或当前 reward 是最终研究方法/节能目标；
- 实现与论文未公开细节达到 parity；
- 新增 preview 已被有效利用，或方法已经改善能耗。

结论由[实际实验记录](../experiments/README.md)支持；未回答的问题留在 [research](../research/)。新增内容只保留“名称正确仍容易误解”的区分，术语定义和实现细节回写各自权威位置。与 ADR 冲突时显式指出，不静默重定义。
