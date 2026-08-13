# System Contract

本文是当前已实现系统行为的规范性 reference。它不规定尚未实现的强化学习、道路预瞄扩展、低层控制器或最终能耗模型。

## 系统边界

当前闭环链路为：MetaDrive 状态构造成官方格式 raw observation，冻结的官方 EMA Diffusion Planner 生成 8 s 联合轨迹，环境执行 ego 轨迹前 0.5 s，然后从实际仿真状态重新规划。

Hydra/OmegaConf 只存在于 CLI 配置边界。入口必须通过 `parse_evaluation_config` 将完整配置解析为
严格且字段冻结的 `EvaluationJobConfig`；runner、episode、runtime 和 execution 组件只接收该类型或
其子模型，不读取 `DictConfig`。`env` 子树是传给 MetaDrive 的开放第三方配置，保留为普通映射；
本项目消费的 horizon、traffic 和 evaluation 字段仍必须由顶层模型交叉校验。

仿真真实状态、模型观测、模型预测和能耗记录必须分别保存。模型预测不得覆盖仿真状态，不同能耗指标不得静默互换或混合累计。业务代码不得从 `ref/` 导入运行时实现。

推理由单进程、单设备 Lightning Fabric 运行时装配，不使用 Trainer。MetaDrive 观测适配器只生成CPU raw tensor；Fabric 统一负责观测传输、模型设备和 forward 精度。`runtime.devices` 必须为 1，不得在同一评测作业内启动多进程闭环。

跨评测作业的进程并行由 ADR 0012 定义。traffic matrix 默认使用两个 Joblib `loky` worker；
每个进程仍保持一个 MetaDrive、一个 `devices=1` Fabric runtime 和一个 artifact writer。CPU
显式验证线程预算；CUDA 只允许两个进程共享一张可见 GPU，并要求确定性配置和正式运行前显存
preflight。smoke、no-traffic 和普通 full 运行保持串行，多 GPU 调度不属于该入口。

## Checkpoint 与模型输入

- 加载官方资产时必须校验 metadata keys、EMA keys、张量形状、EMA tensor 数和参数总数；加载必须使用严格 state-dict 匹配，不提供兼容回退。
- `args.json` 中的 observation/state normalization 是 checkpoint 输入分布契约的一部分，缺失或不匹配时必须失败。
- 冻结的预训练 planner 始终处于 eval mode；该入口不得进入训练模式。
- checkpoint 在 CPU 上完成结构校验和严格 state-dict 加载，再由 `Fabric.setup_module` 移至运行设备并包装标准 `forward`；不得绕过该装配边界手工移动闭环模型。
- planner 必需 observation 字段如下。除 validity mask 为 `torch.bool` 外，字段均为有限 `torch.float32`，位于 planner 的 runtime device，且共享正 batch 维度。

| 字段 | 每个 batch item 的形状 | 语义 |
| --- | --- | --- |
| `ego_current_state` | `[10]` | `x, y, cos(h), sin(h), vx, vy, ax, ay, steering, yaw_rate` |
| `neighbor_agents_past` | `[32, 21, 11]` | 动态体过去 2 s 和当前帧的状态与类型编码 |
| `static_objects` | `[5, 10]` | 静态物体位姿、尺寸和类型编码 |
| `lanes` | `[70, 20, 12]` | 普通 lane 几何与交通灯状态 |
| `lanes_speed_limit` | `[70, 1]` | 普通 lane 限速，单位 m/s |
| `lanes_has_speed_limit` | `[70, 1]` | 普通 lane 限速有效性 |
| `route_lanes` | `[25, 20, 12]` | 局部路线 lane |

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

## 坐标、时间与单位

- 模型局部坐标以规划时刻 ego 后轴中心为原点，x 向前、y 向左。
- MetaDrive vehicle center 与后轴中心的偏移必须按车辆 heading 显式转换；地图、目标轨迹和实际车辆状态使用同一车辆中心约定。
- heading 使用 `[cos(h), sin(h)]`，角差使用最短有向角。
- 模型轨迹为 10 Hz 的 80 个未来点，共 8 s；MetaDrive 物理步长为 0.02 s，`decision_repeat=5`，对外子步为 0.1 s。
- 每个规划周期只执行前 5 点，即 0.5 s，规划频率为 2 Hz。
- 程序化地图限速配置使用 km/h，模型限速使用 m/s；单位转换只在地图适配边界执行一次。
- 能耗、距离、速度、加速度和角速度字段名必须显式标出单位。
- 场景全局平移或旋转后，等价状态应产生等价的局部模型输入。

## 地图与限速

地图输入来自完整 `NodeRoadNetwork`，不得用默认 lidar observation 代替。地图 reset 时缓存完整
lane 几何并建立 STRtree；运行时索引只做保守候选粗筛，最终仍调用 MetaDrive `lane.distance`
精确过滤并按距离、稳定 ID 确定性排序。每条 lane 沿完整弧长重采样 20 点，再截断或零填充为
70 条普通 lane 和 25 条 route lane。route lane 由 navigation checkpoints 的连续 road edge 识别。

程序化地图没有可靠交通灯状态：有效 lane 使用 unknown `[0,0,0,1]`，padding lane 保持全零。MetaDrive 横向坐标正方向指向右侧，适配器必须显式转换成模型的左边界优先语义。

MetaDrive 0.4.3 PGMap 用精确 `1000 km/h` 表示未设置限速。环境 reset 后必须只替换该精确哨兵，保留已有有限、正值且不超过 `130 km/h` 的真实限速，确认哨兵全部消失并记录替换与保留数量。地图适配器若仍发现哨兵或非法限速，必须指出 lane 和原始值并失败；不得统一覆盖真实限速或猜测默认值。

lane 长度与宽度必须接受 Python 或 NumPy 的真实数值标量，同时拒绝 bool、数组、非有限值和非正值。

## 交通观测

`MetaDriveObservationAdapter` 使用 reset 帧和连续 0.1 s 交通快照构造严格的 21 帧历史。每个快照必须在仿真观测时刻捕获为不可变值，并校验对象 ID、类型、位置、heading、速度和尺寸；批量追加历史必须先验证整批，再一次性提交，失败时历史保持不变。

当前帧查询半径内的对象按距离和 ID 排序。历史对象缺帧时使用从当前帧向过去保持最近可用状态的官方填充语义；当前帧不存在的对象不得被选入。首次交通推理前，当前实现固定 ego 并推进背景交通 20 个 0.1 s 子步，连同 reset 帧形成 21 帧历史。该预热不计入正式指标，必须保存状态、奖励、终止标志及动态/静态对象数量；ego 位移达到 `1e-3 m` 或预热提前结束时必须失败。

`NoTrafficMetaDriveObservationAdapter` 只允许显式满足 `traffic_density=0`、`random_traffic=false`、`accident_prob=0` 的场景。reset 后若存在任何动态或静态交通对象必须失败；邻车历史和静态物体字段为全零 padding。该入口不得用于有交通场景。

## 扩散与随机性

预训练模型使用连续时间线性 VP-SDE 并预测 `x_start`。Hydra 必须显式选择 sampler；默认
`dpm10` 保持官方 baseline：从 `0.5 * N(0,I)` 的未来噪声开始，执行 10 步二阶 multistep
DPM-Solver++，结束于 `t=1/1000` 后额外预测一次 `x_start`。其公式和初始尺度不得由配置覆盖。

可选 `ddim5` 从标准高斯未来噪声开始，在 `t = [1.0, 0.8, 0.6, 0.4, 0.2]` 预测 `x_start`，
依次转移到 `[0.8, 0.6, 0.4, 0.2, 0.0]`。该均匀连续时间子序列是本项目复现决定，不是论文公开
事实。评测配置使用 `ddim_stochasticity=0`；非零值必须显式提供同设备
`torch.Generator`。DDIM transition state 保持初始状态的 dtype；mixed-precision denoiser 输出在
sampler 边界显式转换到该 dtype。`0.5 * N(0,I)` DDIM 仅作为带独立 parity 标签的项目隔离变体，
不得解释为 PlannerRFT parity。

sampler 配置必须显式记录固定值 `implementation=diffusers`。两种 `diffusers` scheduler 都使用由项目
连续 VP-SDE 离散化的 `trained_betas`，而非其默认 beta schedule；项目不维护任何 local solver 数值
更新公式。DPM10 使用 DPM-Solver++、二阶
multistep、均匀 lambda spacing，结束于最小训练 sigma 后额外以 `t=0.001` 预测一次 `x_start`；
模型时间由 scheduler 的实际 sigma 恢复，不能直接使用其离散 timestep。DDIM-5 模型时间仍严格为
`[1.0, 0.8, 0.6, 0.4, 0.2]`。`PlanningSampler` 是规划器唯一的 sampler 边界：它封装 profile、
backend 选择和 backend 专属参数，规划器不得按具体 sampler 类型分支。产物中的 sampler metadata
必须保存该后端选择。

给定 observation 和初始噪声，baseline sampler 以及 `ddim_stochasticity=0` 的 DDIM 是确定性的。
每个回合创建一个由噪声 seed 初始化的持久化 `torch.Generator`；每个规划周期先从中取得新的标准
正态噪声，随机 DDIM transition 再从同一 generator 顺序取样。trace 保存未缩放的标准正态初始
噪声；resolved config、作业 summary、回合 summary 和 runtime metadata 保存 sampler 名称、步数、
初始尺度、stochasticity、timestep 与 parity 标签。地图 seed 与噪声 seed 必须分别记录。

`runtime.seed` 必须传给 `Fabric.seed_everything`，并作为每回合噪声 generator 的 seed；地图 seed 仍由 scenario 独立指定。自动设备解析只允许 CPU 或 CUDA。`runtime.precision=auto` 在 CPU 上解析为 `32-true`，在 CUDA 上优先解析为 `bf16-mixed`，不支持 BF16 时解析为 `16-mixed`。实际设备和解析后精度必须写入产物；只有显式 `32-true` 可作为严格 FP32 数值基线。相同 seed 不保证跨设备或跨精度逐位一致。

## Reference planner 与正交 guidance

默认 `guidance=none`，DPM-10 与无 guidance DDIM-5 保持独立入口。可选
`guidance=orthogonal_reference` 只允许标准高斯 DDIM-5；DPM-10 或
`ddim5_project_noise` 与 active guidance 组合时必须失败。

每个规划周期只使用一份严格加载、冻结且 eval-mode 的官方 EMA 模型，并只计算一次 scene
encoding。reference 与 guided pass 共享当前 observation、initial noise 和 DDIM transition draws；
在 reference 前显式取得每个非末步 `variance_noise`，并将同一组 tensor 传给两次 scheduler step；
该随机流协调由 `PlanningSampler` 负责。
reference 每个规划周期刷新。

reference 切向由其有限、非退化的 `[cos(h), sin(h)]` 归一化得到，左法向为
`[-sin(h), cos(h)]`。速度由当前物理点和 80 个未来物理点按 `0.1 s` 后向差分，单位为 m/s；
重复点作为 `0 m/s` 保存和计数，不触发回退。非有限 reference 或 heading norm 不超过配置 epsilon
时立即失败。

固定 guidance action 为有限 `float32 [B,2]`，与 sample 同设备且逐值位于 `[-1,1]`；不得裁剪。
正 lateral 表示左移，正 longitudinal 表示沿 reference heading 加速。横向目标为
`2.5 m * lateral_scale`，纵向目标为
`0.25 * longitudinal_scale * reference_along_track_speed`。阶段 2 使用 ADR 0013 的
reference-centered energy-gradient delta，使 `(0,0)` 精确退化为同次 unguided reference。

每个 DDIM denoise step 对 physical ego objective 经 state normalizer 和冻结 DiT 链式求 normalized
noisy joint sample 梯度。正常 DDIM transition 后以单位系数做负梯度更新，再恢复 current-state
constraint。应用前将当前点和 10 个邻车的梯度置零，只更新 ego 80 个 future channels；未应用的
neighbor gradient norm 仍进入审计。每步更新后 detach，冻结参数不得获得 `.grad`。

guided trace 额外保存完整 reference joint prediction、action、横向目标、纵向目标速度变化、五步
objective delta、应用梯度 L2/max、原始 neighbor gradient L2 和零速计数。上述 centered energy、
单位系数、离散速度与 ego-only scope 是项目复现决定，不得描述为 PlannerRFT 作者公开实现。

## 轨迹执行

运动学接口只接收有限的 `float32 [80,4]` ego 后轴局部轨迹。混合精度 forward 的完整预测必须在保存 trace 和进入环境前原值转换为 `float32`。每次 action 只校验并转换一次，环境与运动学 policy 共享同一份已准备的世界轨迹。每个 0.1 s 子步将 vehicle center、heading、由相邻 center 有限差分得到的 velocity，以及由最短 heading 角差得到的 angular velocity 写入 MetaDrive；下一规划周期以最后实际状态为锚点。

`TrajectoryMetaDriveEnv` 的 Gym observation 只是一元素 `float32` 零数组；planner 输入必须由本项目
traffic/no-traffic observation adapter 构造。headless 无交通环境不初始化传感器；有交通环境只保留
背景 IDM policy 必需的共享 lidar，不把它作为 planner observation。headless、非录制执行把每个
0.1 s 对外子步的五个 `0.02 s` Bullet substep 交给一次原生 `doPhysics` 调用；渲染或 MetaDrive
episode recording 继续使用上游通用 step 路径。

该接口不生成 steering、throttle 或 brake，也不证明低层车辆动力学可执行性。原始轨迹必须原样执行和保存；不得平滑、裁剪、限幅、旋转、投影到中心线、选择最佳噪声 seed、切换回退控制器，或在异常时返回零轨迹。

MetaDrive `step` 返回的轨迹数组、交通快照和终止字段必须在环境边界一次性解析为不可变的
`TrajectoryExecutionRecord`。数组必须具有约定形状、dtype 和有限值，交通帧必须是非空 tuple，
终止标志必须是 bool；evaluation 的 trace、rendering 和 summary 组件不得继续读取原始 `info`
映射。

## 能耗记录

- 每种能耗指标使用独立名称、单位和累计边界。
- MetaDrive 代理能耗与 FASTSim 等精细模型不得混用。
- 能耗结果必须关联实际执行 trace、采样间隔、车辆配置、场景特征和终止类型。
- 程序化地图没有原生坡度时不得假设高程信息。
- 结果必须明确限定在运动学执行条件；当前系统不定义最终奖励或优化目标。

## 评测与产物

每个评测作业必须保存 resolved config、Hydra overrides、runtime Git metadata、tracked diff、地图/场景 seed、噪声 seed、Fabric 请求与解析后的 accelerator/precision、实际设备、依赖环境和场景特征。`tracked_diff.patch` 必须存在，但干净工作区时允许为空。

每个回合至少保存 `summary.json` 和 `trace.npz`；开启视频时保存闭环 GIF。trace 必须包含 raw observation、初始噪声、完整联合预测、规划锚点、目标与实际状态、逐点误差、奖励、终止标志；交通回合还保存预热、对象 ID、交通数量、最近交通距离和历史有效性。

当前唯一支持的产物 schema version 为 3。job summary、episode summary 和 runtime metadata
使用 `extra=forbid` 的严格 Pydantic 模型；读取器拒绝缺失、额外或类型错误字段。trace 显式保存
`complete`、`partial` 或 `empty` 状态、初始状态 validity、普通及 route lane 的限速与有效性，
NPZ 读取时必须校验完整字段集合、形状、dtype、有限性和跨数组长度关系。生产者和消费者均不接受
v1/v2 产物，不提供兼容、推断或迁移层。

runner 只捕获显式 `EpisodeFailure`：保存阶段、异常类型、消息和 traceback 后继续同一作业的后续
场景，作业最终标记失败且 CLI 返回非零。配置、checkpoint、Fabric 初始化、artifact IO 和未分类
程序错误立即传播。

所有回合都进入分析，按场景特征、运行阶段和终止类型标注。失败回合不得删除，完成与失败回合不得在缺少标注时合并解释。策略比较必须使用相同地图、交通配置和噪声 seed。

矩阵汇总只接受预定义网格；部分矩阵必须是该网格的非空子集。缺少必需配置、summary、trace、metadata 或可视化产物时必须失败，不得把坏样本静默跳过。
