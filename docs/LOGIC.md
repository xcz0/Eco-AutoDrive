# 系统逻辑

本文只定义 Eco-AutoDrive 在逻辑上必须怎样工作。实现进度见 `STATUS.md`，方案理由见`DECISIONS.md`，实验结果见 `../experiments/README.md`。

## 系统边界

目标系统由三条边界清晰的链组成：

```text
闭环执行：MetaDrive 状态 -> 官方格式观测 -> Diffusion Planner -> 8 s 轨迹 -> 执行前 0.5 s -> 更新 MetaDrive 状态 -> 重新规划

强化学习：观测 -> DPPO 随机去噪链 -> 轨迹与逐步 log-prob -> 环境奖励 -> rollout / GAE -> PPO 更新 actor 与 critic

能耗计量：实际执行状态 -> 时间/速度 trace -> 明确命名的能耗模型 -> 训练奖励或独立评测指标
```

`sim_state`、`model_obs`、`diffusion_chain` 和 `energy_state` 必须分开保存。模型预测不能覆盖仿真真实状态；MetaDrive 代理能耗和 FASTSim 能耗不能静默互换或混合累计。

当前实现只覆盖第一条链的无交通运动学子集。未实现部分仍须遵守本文契约。

## checkpoint 与上游契约

加载过程必须校验 missing/unexpected keys、张量形状和参数总数。不得采用会把非 DDP checkpoint 过滤为空字典的隐式前缀规则。`normalization.json` 属于 checkpoint 输入分布契约，不是可选预处理。

业务代码不得从 `ref/` 导入。上游的 nuPlan 数据提取和 planner 生命周期只用于理解字段语义，MetaDrive 边界必须在本项目重写。

## 模型张量契约

官方配置固定以下形状；改变 `80`、`25`、`20` 等维度会改变 checkpoint 参数形状：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `ego_current_state` | `[B, 10]` | `x, y, cos(h), sin(h), vx, vy, ax, ay, steering, yaw_rate` |
| `neighbor_agents_past` | `[B, 32, 21, 11]` | 2 s、10 Hz 历史和类型编码 |
| `static_objects` | `[B, 5, 10]` | 位姿、尺寸和类型编码 |
| `lanes` | `[B, 70, 20, 12]` | 中心、切向、左右边界相对向量、信号 one-hot |
| `lanes_speed_limit` | `[B, 70, 1]` | m/s；进入模型前以 `std=20` 归一化 |
| `lanes_has_speed_limit` | `[B, 70, 1]` | 限速有效掩码 |
| `route_lanes` | `[B, 25, 20, 12]` | 连续 route roadblock 上的局部车道 |
| `route_lanes_speed_limit` | `[B, 25, 1]` | 生成但当前 decoder 不消费 |
| `route_lanes_has_speed_limit` | `[B, 25, 1]` | 生成但当前 decoder 不消费 |
| `prediction` | `[B, 11, 80, 4]` | ego 与 10 个邻车的 `x, y, cos(h), sin(h)` |

`lanes` 单点通道顺序固定为：

```text
[center_x, center_y,
 delta_x, delta_y,
 left_boundary_dx, left_boundary_dy,
 right_boundary_dx, right_boundary_dy,
 traffic_green, traffic_yellow, traffic_red, traffic_unknown]
```

所有输入在交给 `PretrainedDiffusionPlanner` 前是 raw、未归一化张量。padding 必须严格全零；归一化后 padding 仍须恢复为全零。真实对象位于局部原点时，朝向、尺寸或类型字段必须避免其被误判为 padding。

动态体最多保留 32 个，按当前距离排序；历史为含当前帧的 21 帧。decoder 联合预测 ego 和前 10 个邻车。未来 80 帧不含当前帧、间隔 0.1 s。内部扩散状态含固定当前点，形状为`[B, 11, 81, 4]`；每个去噪步都必须保持第 0 点不变。

`RouteEncoder` 当前只读取 `route_lanes[..., :4]`，即中心和切向。不得声称现有 checkpoint 已使用 route 边界、信号或 route 限速。

## 坐标、时间与单位

- 模型几何坐标以规划时刻 ego 后轴中心为原点，局部 x 向前、y 向左；
- MetaDrive 车辆中心与后轴中心之间的偏移必须按车辆朝向显式转换；
- heading 以 `[cos(h), sin(h)]` 进入轨迹，角差使用最短有向角；
- 模型轨迹频率为 10 Hz，共 80 个未来点，即 8 s；
- MetaDrive 物理步长为 0.02 s，`decision_repeat=5`，每个环境子步为 0.1 s；
- 高层每次执行 5 个点，即 0.5 s 后重新规划，规划频率为 2 Hz；
- 程序化地图配置限速使用 `km/h`，字段名带 `_kmh`；模型限速使用 `m/s`；
- 限速单位转换只能在地图适配边界执行一次；
- 能耗、距离、速度、加速度、角速度的字段名必须标出单位。

同一物理场景全局平移或旋转后，局部模型输入应保持等价。地图、目标轨迹和实际车辆状态必须使用同一后轴/车辆中心转换约定。

## MetaDrive 地图逻辑

地图输入来自完整道路网络，而不是 MetaDrive 默认 lidar observation：

```text
NodeRoadNetwork
  -> 按 ego 距离筛选并排序
  -> 每条 lane 沿完整弧长重采样 20 点
  -> 转换到 ego 后轴局部坐标
  -> 截断并严格零填充为 70 条普通 lane / 25 条 route lane
```

route lane 由 `navigation.checkpoints` 的连续 road edge 识别。程序化地图没有可靠交通灯状态，有效 lane 使用 unknown `[0, 0, 0, 1]`，padding lane 维持全零。

MetaDrive 0.4.3 的 PGMap 用精确 `1000 km/h` 表示“未设置程序化 lane 限速”。环境 reset 后必须：

1. 读取所有 lane 原始限速；
2. 只把精确 `1000 km/h` 的 lane 替换为必填配置 `programmatic_lane_speed_limit_kmh`；
3. 保留已有有限、正值且在领域上界内的真实限速；
4. 再次确认哨兵全部消失并记录替换/保留计数；
5. 地图适配器若仍见到 `1000 km/h`，立即报错并带 lane id 和原始值。

环境和模型必须看到相同的限速语义。不得只在模型适配器中静默把哨兵改成某个默认值，也不得统一覆盖已有真实限速。

## 观测适配逻辑

通用 `MetaDriveObservationAdapter` 应直接从仿真对象和地图 API 构造官方字典，负责 ego、动态体 21 帧历史、静态物体、普通 lane 和 route lane。

`NoTrafficMetaDriveObservationAdapter` 是更窄的已实现接口：

- 配置必须显式满足 `traffic_density=0`、`random_traffic=false`、`accident_prob=0`；
- reset 后再次拒绝任何动态或静态交通参与者；
- `neighbor_agents_past` 和 `static_objects` 使用官方全零 padding；
- 与官方推理入口一致，`ego_current_state` 固定为 `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]`；
- 该适配器不得用于有交通场景，也不能被描述为通用适配器。

官方推理不消费 ego 历史速度；从静止 MetaDrive 状态开始时模型仍可能直接产生其训练分布中的速度先验。这是要测量的接口/分布问题，不能通过伪造 ego 历史来隐藏。

## 扩散与采样逻辑

预训练模型使用连续时间线性 VP-SDE，`beta_min=0.1`、`beta_max=20`，预测干净状态`x_start`。官方 baseline 推理从 `0.5 * N(0, I)` 的未来噪声开始，运行 10 步、二阶、logSNR 跳步、multistep DPM-Solver++，最后 `denoise_to_zero=True`。

给定观测和初始噪声，baseline solver 是确定性的；每个规划周期必须从持久化、固定种子的`torch.Generator` 取得新的标准正态噪声，并保存该噪声。

官方 DPM-Solver++ 只产生确定性更新和最终样本，没有 DPPO 所需的随机转移分布或逐步log-prob。因此 DPPO sampler 必须作为独立路径实现显式随机高斯反向转移，并能：

- 返回完整去噪状态、时间、均值、方差和逐步 log-prob；
- 用 rollout 时保存的状态重算相同 log-prob；
- 只对明确选定的 actor 动作维度计算 PPO 概率；
- 保持当前点约束和官方归一化；
- 不修改 baseline sampler 来伪造概率。

## 轨迹执行逻辑

当前运动学接口接收完整 `float32 [80, 4]` 后轴局部轨迹。每次高层 `env.step()` 只执行前 5 点。对第 `i` 个未来点，第 `i` 个 0.1 s 子步结束并采样 observation/reward/termination 前：

- vehicle center 必须位于转换后的目标 world center；
- heading 必须等于目标 world heading；
- velocity 来自相邻 center 以 0.1 s 做有限差分；
- angular velocity 来自相邻 heading 的最短角差；
- 下一规划周期的锚点必须等于上一周期最终实际状态。

`KinematicTrajectoryPolicy` 属于轨迹级运动学闭环，不生成 steering/throttle，不能证明低层动力学可执行性。未来若换为 PID、Stanley 或 MPC，应保持规划器和 rollout 边界不变，并另立实验口径。

模型原始轨迹必须原样执行和保存。不得平滑、裁剪、旋转、投影到中心线、选择最佳噪声 seed、在失败前切换控制器或在异常时返回零轨迹。

## 能耗逻辑

研究目标不是单独最小化能耗，而是：在满足到达、安全和旅行时间约束的前提下，最小化单位有效进度能耗。训练奖励应由可审计分量组成：

```text
r_t = w_p * route_progress
    - w_E * normalized_interval_energy
    - w_T * elapsed_time
    - w_c * collision
    - w_o * off_route_or_out_of_road
    - w_j * comfort_penalty
    + terminal_reward
```

各分量、权重、`energy_metric` 名称和原始能耗值必须分别记录。缺少当前阶段必需的能耗字段时立即失败，不得返回零值或换用另一口径。只奖励低能耗会使停车成为伪最优，因此安全、进度、旅行时间和失败率必须与能耗共同报告。

MetaDrive 的 `step_energy` / `episode_energy` 只是在同一环境和车辆配置下比较趋势的代理指标；它主要依赖速度和距离，不能支撑真实燃油或车辆级能效结论。

FASTSim 接收实际执行得到的时间—速度行程，用于更精细的能耗计算。没有坡度数据时必须明确声明 grade 为 0 或不提供相应结论，不能假设 MetaDrive 程序化地图含高程。

每个能耗结果必须记录：模型名称、车辆配置、输入 trace、采样间隔、单位和累计边界。只有当MetaDrive 代理指标出现可复现的优化趋势后，才用 FASTSim 做精细复核；两者结论分开报告。

DPPO actor 由冻结的预训练主体和显式可训练的 preview/adapter 组成；初始策略只应对 ego、实际执行的前 5 个轨迹点聚合 log-prob，不能让未执行的远期点直接承担当步环境奖励。critic 使用与当前环境步对齐的场景特征、道路预瞄和 ego/路线状态。PPO loss、value loss、基线锚定约束和任何平滑正则必须分项记录；平滑正则只能约束训练，不得在评测时篡改原始输出。

## 评测与产物逻辑

每次评测必须固定并记录：代码 commit、未提交 diff、上游 commit、checkpoint/hash、resolved config、Hydra overrides、地图/场景 seed、噪声 seed、设备和依赖环境。

无交通闭环的每个场景至少产生：

- `summary.json`：终止状态、距离、速度、路线完成率、限速审计、checkpoint 信息和执行误差；
- `trace.npz`：完整 raw observation、初始噪声、`[11, 80, 4]` 预测、规划锚点、目标与实际状态、逐点误差、奖励和终止标志；
- `closed_loop.gif`：原始 8 s 规划与实际执行前缀的可视化；
- resolved config 和 Hydra overrides。

模型驾驶表现与接口测试分开判定。接口硬门槛包括：权重和张量契约正确、输出有限、时间轴一致、终止状态明确、目标/实际轨迹误差在回归阈值内。当前基础代码构建阶段只要求本机逻辑测试和最小强化学习流程验证，不产出服务器训练性能结论。强化学习流程跑通并进入服务器训练阶段后，性能结论必须来自预先定义并完整记录运行环境的实验矩阵，不能只报告成功 seed；是否使用 Docker 在该阶段另行确认。

策略比较必须使用相同地图、交通和噪声的 paired seeds，并同时报告均值、中位数、置信区间、到达率、route completion、碰撞/出界、旅行时间和速度。能耗主结论只能在成功且旅行时间满足预设约束的回合上比较，同时另报全体回合失败率，防止以停车或任务失败换取表面节能。
