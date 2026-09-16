# 模型输入与规划契约

涉及 checkpoint、observation、地图/交通、sampler 或 reference guidance 时读取；坐标与时间见[公共契约](../system-contract.md#坐标时间与单位)。这是 [system contract](../system-contract.md) 的分篇，遵循入口中的使用与维护规则。

## Checkpoint 与模型输入

* 官方 planner checkpoint 使用 `torch.load(..., map_location="cpu", weights_only=True)` 加载，并读取其中的 `ema_state_dict`。
* EMA state dict 的键名在加载模型前按当前适配规则改写：移除 `module.` 前缀，并将首个 `encoder.encoder.`、`decoder.decoder.` 分别改写为 `encoder.`、`decoder.`。
* 改写后的 state dict 通过 `model.load_state_dict(..., strict=True)` 加载；缺失或额外模型键以及不兼容的 tensor shape 由 PyTorch 的严格 state-dict 加载失败处理，不提供兼容回退。
* `CheckpointLoadReport` 中的 EMA tensor 数和参数总数是成功提取 state dict 后计算的报告值；当前实现不将其与独立 metadata、预期 tensor 数或预期参数总数进行额外一致性校验。
* `args.json` 单独解析为 `OfficialDiffusionPlannerConfig`，其中的 observation/state normalization 直接构成 planner 的输入与输出归一化配置；当前加载路径不把这些字段与 checkpoint 内独立 metadata 做一致性比较。
* 冻结的预训练 planner 始终处于 eval mode；该入口不得进入训练模式。
* checkpoint 在 CPU 上完成 state-dict 提取、键名改写和严格加载，再由 `Fabric.setup_module` 移至运行设备并包装标准 `forward`；不得绕过该装配边界手工移动闭环模型。
* planner 必需 observation 字段如下。除 validity mask 为 `torch.bool` 外，字段均为有限 `torch.float32`，位于 planner 的 runtime device，且共享正 batch 维度。

| 字段                      | 每个 batch item 的形状 | 语义                                                         |
| ----------------------- | ----------------- | ---------------------------------------------------------- |
| `ego_current_state`     | `[10]`            | `x, y, cos(h), sin(h), vx, vy, ax, ay, steering, yaw_rate` |
| `neighbor_agents_past`  | `[32, 21, 11]`    | 动态体过去 2 s 和当前帧的状态与类型编码                                     |
| `static_objects`        | `[5, 10]`         | 静态物体位姿、尺寸和类型编码                                             |
| `lanes`                 | `[70, 20, 12]`    | 普通 lane 几何与交通灯状态                                           |
| `lanes_speed_limit`     | `[70, 1]`         | 普通 lane 限速，单位 m/s                                          |
| `lanes_has_speed_limit` | `[70, 1]`         | 普通 lane 限速有效性                                              |
| `route_lanes`           | `[25, 20, 12]`    | 局部路线 lane                                                  |

地图适配器还生成 `route_lanes_speed_limit [B,25,1]` 和 `route_lanes_has_speed_limit [B,25,1]`，但当前 planner 输入验证和 decoder 均不消费它们。`RouteEncoder` 只读取 `route_lanes[..., :4]`；不得声称当前模型使用 route 边界、交通灯或 route 限速。

`lanes` 和 `route_lanes` 的单点通道顺序固定为：

```text
[center_x, center_y,
 delta_x, delta_y,
 left_boundary_dx, left_boundary_dy,
 right_boundary_dx, right_boundary_dy,
 traffic_green, traffic_yellow, traffic_red, traffic_unknown]
```

进入 normalizer 前的 observation 必须保持 raw、未归一化表示。padding 必须严格全零，归一化后仍恢复为全零；真实对象处于局部原点时，朝向、尺寸或类型字段必须防止其被误判为 padding。

动态体最多保留 32 个，按当前距离再按稳定 ID 确定性排序。decoder 联合预测 ego 和前 10 个邻车，输出为 `[B,11,80,4]` 的 `x, y, cos(h), sin(h)`；未来 80 帧不含当前帧。内部扩散状态包含固定当前点，形状为 `[B,11,81,4]`。

## 地图与限速

地图输入来自完整 `NodeRoadNetwork`，不得用默认 lidar observation 代替。地图 reset 时缓存完整 lane 几何并建立 STRtree；运行时索引只做保守候选粗筛，最终仍调用 MetaDrive `lane.distance` 精确过滤并按距离、稳定 ID 确定性排序。每条 lane 沿完整弧长重采样 20 点，再截断或零填充为 70 条普通 lane 和 25 条 route lane。route lane 由 navigation checkpoints 的连续 road edge 识别，且该完整 edge 集同时定义 `MetaDriveBackend` 的 route 边界：仍位于 MetaDrive lane、但当前 lane edge 不在该集合中的 ego 必须作为 `out_of_road` terminal。仅在已确认的该 terminal 且局部没有 route edge 时，terminal observation 的 route fields 才保留全零 padding；非 terminal route 缺失必须失败，不得返回旧 observation 或伪造 route。

程序化地图没有可靠交通灯状态：有效 lane 使用 unknown `[0,0,0,1]`，padding lane 保持全零。MetaDrive 横向坐标正方向指向右侧，适配器必须显式转换成模型的左边界优先语义。

MetaDrive 0.4.3 PGMap 用精确 `1000 km/h` 表示未设置限速。环境 reset 后必须只替换该精确哨兵，保留已有有限、正值且不超过 `130 km/h` 的真实限速，确认哨兵全部消失并记录替换与保留数量。地图适配器若仍发现哨兵或非法限速，必须指出 lane 和原始值并失败；不得统一覆盖真实限速或猜测默认值。

能耗 benchmark 可选 `programmatic_lane_speed_limit_profile_kmh`。它按 PGMap 初始 block 之后的生成 block 顺序指定限速，长度必须与该 block 数严格一致；只覆盖这些 block 中的未设置限速哨兵，已有显式限速仍保留。reset audit 必须记录 profile 及实际应用 lane 数。

lane 长度与宽度必须接受 Python 或 NumPy 的真实数值标量，同时拒绝 bool、数组、非有限值和非正值。

## 交通观测

观测 ABI 由 `eco_planner.contracts` 固定，schema 归 `envs.observation`；维度不是运行时配置，checkpoint 在加载边界校验与 ABI 一致。MetaDrive 边界把参与者类型、位置、朝向、速度和尺寸校验并转换为不可变 DTO 与 typed map arrays。`TrafficHistory`、`TrafficSceneEncoder` 和 `ObservationBuilder` 是唯一特征编码链，返回 CPU、unbatched TensorDict；MetaDrive 模块只捕获状态。Serial B=1 使用 TensorDict 原生 stack，vector 直接消费 `ParallelEnv` 的嵌套 `observation` batch，Fabric 将同 schema TensorDict 传到 planner device，无额外字段枚举、collator 或 dict 转换。

`TrafficHistory` 由 reset 帧与连续 0.1 s 快照组成严格 21 帧历史；batch append 先检查整批连续性，失败时历史保持不变。编码器只选择当前帧查询半径内的对象，按距离和稳定 ID 排序，最多保留 32 个动态体和 5 个静态物体。历史缺帧从当前向过去保持最近可用状态；当前帧不存在的对象不选入。动态类型使用 vehicle/pedestrian/bicycle 三通道 one-hot，静态类型使用四通道 ABI，padding 全零。

首次交通推理前固定 ego，推进背景交通 20 个 0.1 s 子步，与 reset 帧形成完整历史。预热不计入正式指标，但保存状态、奖励、终止标志及动态/静态对象数量；ego 位移达到 `1e-3 m` 或预热提前结束时失败。

No-traffic observation pipeline 只允许显式满足 `traffic_density=0`、`random_traffic=false`、`accident_prob=0` 的场景。它在 reset 校验配置，并复用每个 execution 已捕获的 `TrafficFrame` 检查初始与后续场景为空；若存在任何动态或静态交通对象必须失败。`ObservationBuilder` 生成的邻车历史和静态物体字段为全零 padding。该入口不得用于有交通场景。

`VectorMetaDriveEnv.reset(..., slots=...)` 在 worker-owned `MetaDriveEnvSlot` 内完成与 serial 路径相同的 backend reset、history warmup 和首个 observation，再按请求的 slot 顺序返回批量 TensorDict 与完整 reset sidecar；换图时该 slot 同时重建 environment、observation pipeline 和 map cache。`TorchRLMetaDriveEnv._reset` 在观测构建时若遇到 MetaDrive navigation 损坏，捕获专用 local-route exception、调用 `MetaDriveEnvSlot.recreate_environment` 重建 env 并重试一次 reset；此恢复路径不改变正常 reset 行为。`reset(..., slots=...)` 和 `step(..., slots=...)` 统一通过 TorchRL partial `_reset` 和 `_step` mask 只推进指定 slot；worker 异常带回 slot 和 operation，并使整个 vector run 明确失败。

## 扩散与随机性

预训练模型使用连续时间线性 VP-SDE 并预测 `x_start`。Hydra 必须显式选择 sampler；默认 `dpm10` 保持官方 baseline：从 `0.5 * N(0,I)` 的未来噪声开始，执行 10 步二阶 multistep DPM-Solver++，结束于 `t=1/1000` 后额外预测一次 `x_start`。其公式和初始尺度不得由配置覆盖。

可选 `ddim5` 从标准高斯未来噪声开始，在 `t = [1.0, 0.8, 0.6, 0.4, 0.2]` 预测 `x_start`，依次转移到 `[0.8, 0.6, 0.4, 0.2, 0.0]`。该均匀连续时间子序列是本项目复现决定，不是论文公开事实。评测配置使用 `ddim_stochasticity=0`；非零值必须显式提供同设备 `torch.Generator`。DDIM transition state 保持初始状态的 dtype；mixed-precision denoiser 输出在 sampler 边界显式转换到该 dtype。`0.5 * N(0,I)` DDIM 仅作为带独立 parity 标签的项目隔离变体，不得解释为 PlannerRFT parity。

`ddim_stochasticity=0` 时，sampler 向 diffusers scheduler 传递 `variance_noise=None`，不创建与 sample 同形的零 tensor；该路径不消费 transition RNG，且与显式零 noise 的 scheduler 输出逐位一致。非零 stochasticity 和调用者显式提供 `variance_noise` 的路径保持原有随机流。

sampler 配置必须显式记录固定值 `implementation=diffusers`。两种 `diffusers` scheduler 都使用由项目连续 VP-SDE 离散化的 `trained_betas`，而非其默认 beta schedule；项目不维护任何 local solver 数值更新公式。DPM10 使用 DPM-Solver++、二阶 multistep、均匀 lambda spacing，结束于最小训练 sigma 后额外以 `t=0.001` 预测一次 `x_start`；模型时间由 scheduler 的实际 sigma 恢复，不能直接使用其离散 timestep。DDIM-5 模型时间仍严格为 `[1.0, 0.8, 0.6, 0.4, 0.2]`。`PlanningSampler` 是规划器唯一的 sampler 边界：它封装 profile、backend 选择和 backend 专属参数，规划器不得按具体 sampler 类型分支。产物中的 sampler metadata 必须保存该后端选择。

给定 observation 和初始噪声，baseline sampler 以及 `ddim_stochasticity=0` 的 DDIM 是确定性的。每个逻辑 slot 创建一个由噪声 seed 初始化的持久化 `torch.Generator`；每个规划周期先从对应 slot generator 取得新的标准正态噪声，随机 DDIM transition 再从同一 slot generator 顺序取样。batch runtime 接收每个 slot 各自的 generator，并在每个 transition 逐 slot 组织随机张量，使 batch composition 不改变其他 slot 的 RNG 消费。trace 保存未缩放的标准正态初始噪声；resolved config、作业 summary、回合 summary 和 runtime metadata 保存 sampler 名称、步数、初始尺度、stochasticity、timestep 与 parity 标签。地图 seed 与噪声 seed 必须分别记录。

`runtime.seed` 必须传给 `Fabric.seed_everything`，并作为每回合噪声 generator 的 seed；地图 seed 仍由 scenario 独立指定。自动设备解析只允许 CPU 或 CUDA。`runtime.precision=auto` 在 CPU 上解析为 `32-true`，在 CUDA 上优先解析为 `bf16-mixed`，不支持 BF16 时解析为 `16-mixed`。实际设备和解析后精度必须写入产物；只有显式 `32-true` 可作为严格 FP32 数值基线。相同 seed 不保证跨设备或跨精度逐位一致。

CUDA 评测设置 `torch.set_float32_matmul_precision("high")`；该选择只影响允许浮点差异的性能路径，不得改变终止原因、planning-cycle 数或 simulator-step 数。

## Reference planner 与正交 guidance

默认 `guidance=none`，DPM-10 与无 guidance DDIM-5 保持独立入口。可选 `guidance=orthogonal_reference` 只允许标准高斯 DDIM-5；DPM-10 或 `ddim5_project_noise` 与 active guidance 组合时必须失败。

每个规划周期只使用一份严格加载、冻结且 eval-mode 的官方 EMA 模型，并只计算一次 scene 和 route encoding。reference 与 guided pass 共享当前 observation、这些 encoding、initial noise 和 DDIM transition draws；在 reference 前显式取得每个非末步 `variance_noise`，并将同一组 tensor 传给两次 scheduler step；该随机流协调由 `PlanningSampler` 负责。reference 每个规划周期刷新。

reference 切向由其有限、非退化的 `[cos(h), sin(h)]` 归一化得到，左法向为 `[-sin(h), cos(h)]`。速度由当前物理点和 80 个未来物理点按 `0.1 s` 后向差分，单位为 m/s；重复点作为 `0 m/s` 保存和计数，不触发回退。非有限 reference 或 heading norm 不超过配置 epsilon 时立即失败。

固定 guidance action 为有限 `float32 [B,2]`，与 sample 同设备且逐值位于 `[-1,1]`；不得裁剪。正 lateral 表示左移，正 longitudinal 表示沿 reference heading 加速。横向目标为 `2.5 m * lateral_scale`，纵向目标为 `0.25 * longitudinal_scale * reference_along_track_speed`。该实现使用 ADR 0013 的 reference-centered energy-gradient delta，使 `(0,0)` 精确退化为同次 unguided reference。

每个 DDIM denoise step 对 physical ego objective 经 state normalizer 和冻结 DiT 链式求 normalized noisy joint sample 梯度。正常 DDIM transition 后以单位系数做负梯度更新，再恢复 current-state constraint。应用前将当前点和 10 个邻车的梯度置零，只更新 ego 80 个 future channels；未应用的 neighbor gradient norm 仍进入审计。每步更新后 detach，冻结参数不得获得 `.grad`。

guided trace 额外保存完整 reference joint prediction、action、横向目标、纵向目标速度变化、五步 objective delta、应用梯度 L2/max、原始 neighbor gradient L2 和零速计数。上述 centered energy、单位系数、离散速度与 ego-only scope 是项目复现决定，不得描述为 PlannerRFT 作者公开实现。
