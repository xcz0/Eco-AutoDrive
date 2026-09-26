---
kind: contract
status: active
scope: [observation, model, coordinates, padding]
read_when:
  - changing observation shapes coordinates normalization or model boundaries
---

# Data and model contract

本篇 MUST／MUST NOT 表示强制的软件要求；研究方法见
[planning/evaluation](../research/protocols/planning-and-evaluation.md)，策略 context 另见
[training contract](training.md)。不以当前模块组织定义规范。

## 坐标、单位与模型边界

模型局部坐标 MUST 以规划时刻 ego 后轴中心为原点，x 向前、y 向左。MetaDrive vehicle center 与后轴偏移必须按 heading 显式转换；地图、目标和实际状态采用一致的中心约定。
Heading 编码为 `[cos(h),sin(h)]`，角差使用最短有向角。全局平移／旋转后的等价场景应产生等价局部输入。

地图配置限速使用 km/h，模型限速使用 m/s，只在地图适配边界转换一次。
能耗、距离、速度、加速度和角速度字段 MUST 标明单位。
Raw observation 在 normalizer 前未归一化；padding 全零，normalization 后仍恢复全零。
真实对象位于局部原点时，heading、尺寸或类型必须防止它被误判为 padding。

Checkpoint MUST 严格加载官方 `ema_state_dict`，缺失／额外 keys 和 shape 不匹配直接失败。
配套 normalization 使用官方参数配置，MUST 保持其输入／输出含义，不能静默 fallback。
Planner MUST 冻结、eval mode；编码上下文脱离训练图，policy backward 不产生 planner `.grad`。
本篇不要求新增独立 checkpoint metadata、tensor 数／参数数校验或兼容机制。

## Observation 与 prediction ABI

Planner 输入字段共享正 batch 维度 `B` 与 runtime device；数值字段有限 `float32`，validity mask 为 bool。
CPU raw observation 与 trace 原值保留；供计算的 device 副本可以使用解析后的混合精度。
观测 schema 的维度是 ABI，不是可自由调整的运行参数。

| 字段 | 每个 batch item 形状 | 内容 |
| --- | --- | --- |
| `ego_current_state` | `[10]` | x, y, cos(h), sin(h), vx, vy, ax, ay, steering, yaw_rate |
| `neighbor_agents_past` | `[32,21,11]` | 过去 2 s 加当前帧的动态体状态和类型 |
| `static_objects` | `[5,10]` | 静态位姿、尺寸、类型 |
| `lanes` | `[70,20,12]` | 普通 lane 几何与灯态 |
| `lanes_speed_limit` | `[70,1]` | m/s |
| `lanes_has_speed_limit` | `[70,1]` | Bool validity |
| `route_lanes` | `[25,20,12]` | 局部路线 lane |

地图边界另提供 route 限速／validity `[B,25,1]`。这些字段的存在不构成模型使用承诺；
迁移来源明确指出 planner 不消费 route 限速，route encoder 只读取 `route_lanes[..., :4]`。
不得据 ABI 宣称 route 边界、灯态或限速已进入模型条件；消费范围变化须作为显式研究变量。

Lane 单点通道 MUST 按以下顺序：

```text
center_x, center_y, delta_x, delta_y,
left_boundary_dx, left_boundary_dy, right_boundary_dx, right_boundary_dy,
traffic_green, traffic_yellow, traffic_red, traffic_unknown
```

Joint prediction 为 `[B,11,80,4]` 的 `x,y,cos(h),sin(h)`：ego 加前 10 个邻车，未来 80 点不含当前点，10 Hz、8 s。内部 diffusion state 含固定当前点，为 `[B,11,81,4]`。
执行 ABI 是有限 `float32 [80,4]` ego 后轴局部轨迹，非零 heading；混合精度结果在 host producer 原值转换为 float32，不做修复。
共享尺寸／采样间隔由同一 ABI 定义拥有，producer tests 保证其输出。

## 地图与限速输入

地图 MUST 来自完整 road network，不能用 lidar observation 替代。Lane 按真实几何距离过滤，
再按距离及稳定 ID 确定性排序，沿完整弧长重采样后截断／零填充到 ABI。
Route 由 navigation checkpoints 的连续 road edges 定义；route 结束处理见 [execution](execution.md)。
非 terminal 缺 route MUST 失败，不返回旧 observation 或伪造路线。
只有已确认离开 route 的 terminal、且局部无 route edge 时，才允许 terminal route 全零 padding。

无可靠交通灯状态时有效 lane 为 unknown `[0,0,0,1]`，padding 仍全零。
MetaDrive 横向正方向向右，必须转换成模型左边界优先语义。

程序化地图精确 `1000 km/h` 哨兵 MUST 在 reset 以显式配置替换；已有有限、正且不超过 `130 km/h` 的真实限速保留。残留哨兵／非法限速直接报 lane 和原值，不猜默认值。
可选 per-block profile 按初始 block 后的生成顺序给出，长度严格匹配；只替换相应 block 的哨兵。
Reset audit 记录 profile、替换／保留数量及实际应用 lane 数。
Lane 长宽接受 Python/NumPy 真实数值标量，拒绝 bool、数组、非有限或非正值。

## 交通历史与对象选择

历史为 reset 帧及连续 simulator 快照形成的 21 帧，包含当前帧；
首次交通推理前所需 20 个 warmup 子步由 execution contract 保证。
批量 append MUST 先检查整批连续性，失败不部分更新历史。
只选择当前帧查询半径内存在的对象，按当前距离和稳定 ID 排序，最多 32 动态体、5 静态物体。
历史缺帧从当前向过去保留最近可用状态；这是既有编码规则，不意味着可以修补一般缺失样本。
动态类型为 vehicle/pedestrian/bicycle 三通道 one-hot，静态类型四通道；padding 全零。

No-traffic 输入 MUST 同时满足 `traffic_density=0`、`random_traffic=false`、`accident_prob=0`，
并在 reset 和后续捕获状态确认无动态／静态交通。若存在真实对象则失败，不将其抹成 padding。

## 固定干预动作

固定 guidance action 为有限 `float32 [B,2]`，顺序 `(lateral,longitudinal)`，同 sample device，
逐值处于闭区间 `[-1,1]`；非法值不得 clip。这是固定人工干预接口，不经过 Beta/log-prob，
不能用它放宽 training contract 中严格开区间的随机策略动作。
Reference 非有限或 heading norm 不超过配置 epsilon 时 MUST 失败；重复点零速保持可审计。
DDIM transition state 保持初始 state dtype，denoiser 的混合精度输出在 sampler 边界显式转回。

## 来源与导航

迁移来源：[旧 planner](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/agents/contracts/planner.md)、[system contract](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/agents/system-contract.md)；
接受依据包括 [ADR 0001](../adr/0001-preserve-official-baseline.md)、
[0004](../adr/0004-require-explicit-programmatic-speed-limits.md)、
[0005](../adr/0005-separate-traffic-observation-boundaries.md)、
[0013](../adr/0013-add-reference-centered-orthogonal-guidance.md)。
正文未迁入加载调用顺序、cache/索引实现或网络 walkthrough；仅保留跨边界语义。

| 修改面 | 实现定位 | 相关测试（未运行） |
| --- | --- | --- |
| Observation shape/padding | [ABI](../../src/eco_planner/contracts.py)、[observation](../../src/eco_planner/envs/observation/) | [observation](../../tests/planning/test_observation.py) |
| 地图／交通适配 | [MetaDrive boundary](../../src/eco_planner/envs/metadrive/)、[history](../../src/eco_planner/envs/observation/history.py) | [simulation tests](../../tests/simulation/) |
| Guidance / prediction | [diffusion](../../src/eco_planner/planning/diffusion/) | [guidance](../../tests/planning/test_guidance.py)、[sampling](../../tests/planning/test_sampling.py) |
