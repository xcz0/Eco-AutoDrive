# 项目状态

**更新日期**：2026-08-04  
**当前里程碑**：基础代码构建阶段。阶段 0 已完成；阶段 1 仅完成无交通短程闭环的接口修正与本机验证。当前目标是补齐系统代码逻辑并在本机跑通最小强化学习流程。

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
- `evaluate.py` 可用官方 EMA、固定噪声生成器和滚动重规划运行 `S`/`SC` 无交通闭环；
- 每次评测保存 resolved config、raw observation、噪声、完整预测、目标/实际状态、执行误差、限速审计、summary 和可选 GIF；
- 已修正 PGMap `1000 km/h` 未设置限速哨兵和运动学 waypoint 执行时序，详见 `CORRECTIONS.md`。

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

修正后的 2 s 直道运行使用 18 条 `50 km/h` lane，速度约 `10.45–10.87 m/s`，行驶`21.21 m`，以 `max_step` 截断。完整参数和产物索引见 `experiments/README.md`。

## 尚未完成

- 通用 `MetaDriveObservationAdapter`：ego/邻车 21 帧历史、静态物体和有交通运行时检查；
- 低密度或响应式交通闭环；
- 低层 steering/throttle 轨迹跟踪及动力学可执行性验证；
- DPPO stochastic sampler、逐步 log-prob、critic、rollout buffer、GAE 和 PPO 更新；
- 长程 `RoutePreviewBuilder` / `RoadPreviewEncoder`；
- MetaDrive 代理能耗奖励接入和基线；
- FASTSim trace meter、车辆配置和精细能耗复核；
- 可运行的 `train.py`；
- 2–5 km 及更长路线的训练/评测。

## 后续验证与训练

- `S` 与 `SC`、噪声 seeds `0..4`（资源允许时 `0..15`）的 20 s 无交通矩阵；
- `SC` 中 18 条补充 `50 km/h` 与 12 条保留 `20 km/h` 的完整闭环表现；
- 上游与本实现对相同 raw observation、相同噪声的 encoder/solver 逐层数值对照；
- 真实 nuPlan 输入与 MetaDrive 输入的字段分布对照；这需要 nuPlan 场景和地图数据，只有 devkit 源码不能替代。
- 最小强化学习流程在本机跑通后，再使用服务器进行 CUDA 训练、长时运行和性能评测；

## 下一步

1. 完成通用有交通 observation、奖励和 rollout 所需的基础接口，并为张量、单位、时间轴和失败条件补齐本机回归测试；
2. 实现并单测 DPPO stochastic sampler 及逐步 log-prob 重算；
3. 接入 critic、rollout buffer、GAE、PPO 更新和可运行的 `train.py`；
4. 使用小规模本机配置跑通“采样—奖励—优势估计—参数更新—保存/恢复”的最小强化学习流程；
5. 在不阻塞代码构建的前提下补充上游逐层数值对照和必要的短程闭环诊断；
6. 最小强化学习流程跑通后，再使用服务器训练并开展多 seed、长时闭环和性能评测；代理能耗出现可复现优化趋势后，再接入 FASTSim 做精细复核。

任何一步失败都应保留 trace、配置和失败状态；不得通过后处理、回退控制器或挑选成功 seed 推进里程碑。
