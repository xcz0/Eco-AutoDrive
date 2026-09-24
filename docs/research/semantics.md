---
kind: semantics
status: proposed
scope: [planning, training, evaluation, interpretation]
read_when:
  - clarifying research terms or interpretation limits
---

# Semantics：概念与解释边界

这是 [Issue #102](https://github.com/xcz0/Eco-AutoDrive/issues/102) 的规范草案。
旧入口尚未切换，本篇不取代现行文档；`proposed` 表示迁移待审阅，不表示下述概念都是新提案。
本篇只定义概念。方法归 [planning/evaluation](protocols/planning-and-evaluation.md)、
[training](protocols/training.md) 和 [diagnostic studies](protocols/diagnostic-studies.md)；
软件保证归 [data/model](../contracts/data-and-model.md)、[execution](../contracts/execution.md)、
[training contract](../contracts/training.md) 和 [artifacts](../contracts/artifacts.md)。

## 场景、运行与时间单位

| 概念 | 含义与边界 |
| --- | --- |
| Scenario | 地图、交通条件和场景随机种子的组合；名称标识条件，不代表结果。 |
| Episode | 从 reset 到到达、截断、碰撞、出界或运行错误的单次闭环运行。预热属于初始化；正式指标窗口由 Protocol 定义。 |
| Planning cycle | 根据当前观测生成一次联合未来预测、执行其中一段、再重新规划的高层决策单位。预测时域不等于执行前缀。 |
| Rollout transition | 训练中的高层 MDP 转移，由一次 planner/policy decision 及其实际执行前缀构成；reward、边界和下一状态按该转移记账。 |
| Simulator substep | 闭环可观测的仿真推进单位；可能包含多个内部 physics step。不能把 physics、substep、decision 和 PPO update 当作同一时间单位。 |
| Evaluation job | 一份 resolved config、推理运行时与独立产物目录对应的执行单位，可包含多个 episode。 |
| Job-level parallelism | 隔离进程中同时运行多个 job；与同一 job 内多个环境共享 batch inference 的 vector execution 不同。 |
| Traffic history / warmup | 以当前帧结束的交通状态历史／形成该历史的初始化过程；不同于正式执行轨迹。 |

Evaluation 的 planning/substep 轴与 training 的 transition 轴承载不同聚合层级。
即使两者采用相同 cadence，也不能把 transition 聚合量当作逐 substep 事实，或推断其
episode 编排、RNG 生命周期与 artifact schema 相同。

## 随机变量

Map seed 控制程序化地图与场景生成；diffusion noise seed 控制规划的初始噪声及随机扩散转移；
policy action seed 控制策略动作抽样；training seed 是派生训练随机命名空间的实验标识；
minibatch seed 控制优化样本的遍历顺序。Bootstrap/MC 等统计重采样的 seed 属于其统计测量。
这些名字标识不同随机变量，不能用笼统的“相同 seed”抹去区别。

哪些流必须固定或配对由 Protocol 决定；draw/save/restore 时机和可复现范围由 Contract 定义。
相同 seed 本身不是跨设备、精度或执行条件逐位一致的证据。

## Reference、action 与 guidance

Reference trajectory 是同一规划周期中冻结 planner 在共享观测与扩散随机流下生成的预测，
是 guidance 的比较基准。它不是中心线、expert/ground truth，也不是安全 fallback。
Reference-centered guidance 因此不自动提供道路几何约束、专家监督或安全保证。

Base action 是策略分布中的基础随机变量；guidance action 是映射后作用于规划的横向、纵向控制量。
两者属于不同概率空间，亦不等于 DDIM transition 或候选选择事件。
固定人工干预与随机策略动作的有效域不同；精确定义见
[动作概率契约](../contracts/training.md#动作与概率记账)。

Orthogonal guidance 指相对 reference 切向与左法向定义的更新。
更新规则、neutral action 和梯度作用范围是研究方法，见
[guidance protocol](protocols/planning-and-evaluation.md#reference-centered-guidance)。

## 执行、道路信息与能耗

Kinematic execution 直接将轨迹点写入仿真车辆状态，用于隔离规划行为。
它不证明 steering、throttle、brake 控制下的动力学可跟踪性。
Tracking error 衡量给定执行条件下 target 与 actual 的差异，也不能代替车辆动力学验证。

Stable energy scenario 是可重复表达巡航、曲率、变道／合流、限速变化或有限交通交互的场景。
Road preview 是局部路线几何之外显式提供的前方曲率、限速、拓扑、交通或道路属性信息。
字段存在、进入模型与被有效利用是不同命题；局部 route conditioning 不等于 preview 问题已解决。

| 能耗概念 | 可以解释什么 | 不可混淆什么 |
| --- | --- | --- |
| Native simulator energy | 仿真器自身 phase boundary 上的能耗审计值 | 不等同于依据本项目实际执行轨迹计算的 proxy。 |
| Execution proxy energy | 固定仿真与车辆条件下用于相对比较的代理量 | 不直接代表真实车辆燃油／电耗；少走或提前失败造成的总量降低不等于节能改善。 |
| High-fidelity energy model | 依赖车辆参数与完整执行轨迹的外部模型复核，例如 FASTSim | 精细模型仍是模型；与 proxy 的单位、状态和累计窗口不同，不能相加或静默互换。 |

## 结束、缺失与结论

Termination 表示任务本身的终止；truncation 表示时间或采集限制导致的截断。
Bootstrap mask 表示 TD target 是否允许利用尾状态 value；是否停止 GAE 递归是另一件事。
精确 flag 关系由 [训练边界契约](../contracts/training.md#episode-与-bootstrap) 拥有。

运行错误是失败证据，不是可伪造成 terminal 的有效转移。Completed 表示 episode 结果完成记录，
不等于成功到达；碰撞和出界也可以具有完整结果。Partial、empty、undefined 与数值零含义不同。

实现可运行、软件测试通过、真实实验完成和科研结论被支持是四种不同状态。
接入 PPO、guidance 或 reward 不证明方法有效、PlannerRFT/nuPlan parity、真实车辆舒适性或节能改善。
结论必须带实验条件与证据边界；未公开的论文实现细节不能用本项目选择冒充。

## 迁移依据

概念来源：[CONTEXT](../../CONTEXT.md)、[domain](../agents/domain.md)。
精细能耗解释依据：[ADR 0032](../adr/0032-separate-online-proxy-and-offline-fastsim-energy.md)。
本次把旧文档混入的数值 cadence、动作分布、bootstrap 关系和实现状态分流至相应草案，
不产生新的研究结论。代码／测试导航放在对应 Contract，概念篇不维护实现目录表。
