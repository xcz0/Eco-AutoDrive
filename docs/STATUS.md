# 项目状态

**更新日期**：2026-08-04  
**当前里程碑**：基础代码构建阶段。阶段 0 已完成；阶段 1 已完成无交通闭环和通用交通观测接口。2–5 km 长路线交通矩阵已按用户要求在 12/20 回合后停止，当前证据覆盖 paired seeds`0..2`，不能表述为完整 seeds `0..4` 基线。下一目标仍是补齐强化学习代码并跑通最小流程。

## 已完成

### Diffusion Planner 基线

- 已实现与官方 checkpoint 参数层级兼容的模型主体、Encoder、DiT、归一化和 10 步 DPM-Solver++ baseline sampler；
- 已实现严格 EMA 权重加载、文件哈希、key、形状和参数量校验；
- 已有合成 observation、噪声和输出契约的单元/集成测试；
- 同一 CPU 输入和噪声下，本实现与 `ref/Diffusion-Planner` 最终输出最大绝对误差为 `3.1e-5`，既定容差为 `atol=5e-5, rtol=0`。

### MetaDrive 无交通闭环

- 已实现 `MetaDriveMapAdapter`、`TrajectoryMetaDriveEnv` 和 `KinematicTrajectoryPolicy`；
- 地图适配器可构造固定形状的普通 lane、route lane 和限速张量；
- 已实现严格限定为空场景的 `NoTrafficMetaDriveObservationAdapter`；
- `scripts/evaluate.py` 可用官方 EMA、固定噪声生成器和滚动重规划运行 `S`/`SC` 无交通闭环；
- 每次评测保存 resolved config、raw observation、噪声、完整预测、目标/实际状态、执行误差、限速审计、summary 和可选 GIF；
- 已修正 PGMap `1000 km/h` 未设置限速哨兵和运动学 waypoint 执行时序，详见 `CORRECTIONS.md`。
- Windows CPU 的 `S`/`SC`、噪声 seeds `0..4`、20 s 上限矩阵已完成且无驾驶失败；

### MetaDrive 有交通闭环

- 已实现 `MetaDriveObservationAdapter`、reset/10 Hz 子步交通快照、稳定 ID 跟踪、`[32,21,11]` 邻车历史和 `[5,10]` 静态物体编码；
- 首次推理前用四段显式静止轨迹生成真实 2 s 历史，预热不计入评测指标；
- 已增加 `Trigger` IDM 交通配置、2.44–3.90 km 长路线、交通审计、运行元数据和严格矩阵汇总；
- Windows CPU 已完成 seeds `0..2`、密度 `0.05/0.10`、两条路线的 12 回合部分矩阵；12 回合全部产物与接口门槛通过，但 0 回合到达，9 回合碰撞、3 回合出界；
- 原计划 seeds `0..4` 的 20 回合矩阵未完成：用户在 12 回合后要求停止，seeds `3..4` 没有可纳入统计的完整回合。

### 环境与工程

- Python 3.10、uv、Ruff、pytest、Hydra 和依赖配置已建立；
- `train.py` 保持占位入口，没有伪造 DPPO 训练实现。

## 已验证结论

| 结论 | 证据范围 | 不能推出什么 |
| --- | --- | --- |
| 官方 EMA 可严格加载并稳定推理 | 固定资产、CPU 合成输入、自动化测试 | 不等于复现官方 nuPlan 闭环指标 |
| 移植模型与上游最终输出数值接近 | 固定输入/噪声，最大误差 `3.1e-5` | 尚未完成真实 nuPlan 场景逐层对照 |
| PGMap 哨兵不能作为有效限速进入模型 | Windows simulator 回归，`S`/`SC` 固定 seed | 不证明所有程序化地图限速语义都已覆盖 |
| 运动学执行器可逐点匹配 waypoint | Windows 直线/变 heading 回归，最大位置误差约 `4.6e-4 m` | 不证明低层车辆动力学可跟踪 |
| 修正后 `S` seed 0 可运行 2 s 不出界 | Windows CPU、噪声 seed 0、4 个规划周期 | 不构成 20 s、多 seed、弯道或 CUDA 性能基线 |
| `S`/`SC` 在 seeds `0..4` 下无闭环失败 | Windows CPU；`S` 全部 10 s 到达，`SC` 全部运行 20 s | 不证明有交通、低层控制、长路线或 CUDA 性能 |
| 通用交通观测满足 21 帧/10 Hz/稳定 ID 契约 | 单元测试、11 个 simulator 测试、12 个长时 traffic trace | 不证明与真实 nuPlan 感知分布一致 |
| 官方 EMA 在已完成长时交通回合中未能到达 | 2 条长路线、密度 `0.05/0.10`、paired seeds `0..2` | 样本只有 12/20，不能形成 seeds `0..4` 正式性能结论 |

20 s 上限矩阵中，`S` 的 5 个回合均在 10 s 到达，平均行驶 `112.277 m`；`SC` 的 5 个回合均运行 20 s 至 `max_step`，平均行驶 `180.021 m`。10 个回合均无碰撞、无出界，最大位置/heading 执行误差分别为 `4.60045e-4 m` 和 `1.78748e-6 rad`。完整参数、逐 seed 结果和产物索引见 `experiments/README.md`。

长时交通部分矩阵的 12 个回合均观察到查询半径内车辆，trace 全部有限且时间轴一致；最大位置/heading 执行误差分别为约 `6.10e-5 m` 和 `2.59e-6 rad`。所有路线到达率为 0：直道 6/6 为 `crash_vehicle`；混合路线密度 `0.05` 的 3/3 为 `crash_vehicle`，密度 `0.10` 的 3/3 为 `out_of_road`。这些是有效失败结果，但统计仅覆盖 seeds `0..2`。

## 尚未完成

- 长时交通 seeds `3..4` 及完整 20 回合矩阵；当前按用户要求不继续运行；
- 程序化正式矩阵之外的行人、自行车和静态事故物体集成场景；
- 低层 steering/throttle 轨迹跟踪及动力学可执行性验证；
- DPPO stochastic sampler、逐步 log-prob、critic、rollout buffer、GAE 和 PPO 更新；
- 长程 `RoutePreviewBuilder` / `RoadPreviewEncoder`；
- MetaDrive 代理能耗奖励接入和基线；
- FASTSim trace meter、车辆配置和精细能耗复核；
- 可运行的 `train.py`；
- 2–5 km 路线的完整多 seed 性能评测和训练。

## 后续验证与训练

- 资源允许时把 20 s 无交通矩阵扩展到噪声 seeds `0..15`；

## 下一步

1. 分析现有 12 回合的碰撞/出界 trace，明确交通域偏移与长弯道失效机制，不使用回退或后处理；
2. 实现并单测 DPPO stochastic sampler 及逐步 log-prob 重算；
3. 接入 critic、rollout buffer、GAE、PPO 更新和可运行的 `train.py`；
4. 使用小规模本机配置跑通“采样—奖励—优势估计—参数更新—保存/恢复”的最小强化学习流程；
6. 最小强化学习流程跑通后，训练并开展多 seed、长时闭环和性能评测；代理能耗出现可复现优化趋势后，再接入 FASTSim 做精细复核。

任何一步失败都应保留 trace、配置和失败状态；不得通过后处理、回退控制器或挑选成功 seed 推进里程碑。
