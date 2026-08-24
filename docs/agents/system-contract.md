# System Contract

本文是当前已实现系统行为的规范性 reference。它规定 evaluation 以及 policy-guided rollout、GAE、PPO training 的数据和执行契约；不规定低层控制器或最终能耗模型。

## 使用与维护

本文件只描述**已经实现并应由当前代码遵守的持久系统契约**，不是 roadmap、研究计划或实验日志。

使用时按任务读取相关章节，不要因为文件是权威 reference 就默认把全文载入上下文。若任务只涉及 sampler，就读取扩散/guidance 章节；若只涉及 PPO，就读取 Exploration Policy、rollout 和 GAE/PPO 章节。跨模块行为变化时再扩大读取范围。

当前实现事实最终仍由代码、测试和机器可读配置体现。若它们与本文件冲突，先明确指出差异；若本次修改改变了持久的数据或执行语义，同步修正本文件，而不是保留两套解释。

以下内容不要写入本文件：

- 尚未确定的方法或目标 → `docs/research/`；
- 设计为什么这样选择 → `docs/adr/`；
- 某次实际运行的配置、结果和 provenance → `docs/experiments/`；
- 尚未完成的实现任务和验收标准 → GitHub Issues。

不要为了兼容历史 artifact 或旧实现而记录不存在于当前代码中的行为。历史事实需要保留时，放入对应实验记录或 ADR。

## 导航

- **系统边界**：evaluation runtime、配置边界、设备与并行模型。
- **Checkpoint 与模型输入**：官方权重加载、observation 字段、shape 和 padding。
- **坐标、时间与单位**：局部坐标、采样频率、execution horizon 和单位转换。
- **地图与限速 / 交通观测**：MetaDrive map/traffic adapter 契约。
- **扩散与随机性**：DPM/DDIM、scheduler、seed、precision 与随机流。
- **Reference planner 与正交 guidance**：固定 reference guidance 的数学和审计边界。
- **Exploration Policy / Policy-guided rollout / GAE、PPO 与训练**：learned guidance 与 RL 数据流。
- **轨迹执行**：运动学执行边界与 `TrajectoryExecutionRecord`。
- **能耗记录**：指标隔离和结论限制。
- **评测与产物**：JSON/NPZ、failure、trace 和 matrix 汇总契约。

## 系统边界

当前闭环链路为：MetaDrive 状态构造成官方格式 raw observation，冻结的官方 EMA Diffusion Planner 生成 8 s 联合轨迹，环境执行 ego 轨迹前 0.5 s，然后从实际仿真状态重新规划。

Hydra/OmegaConf 只存在于 CLI 配置边界。入口必须通过 `parse_evaluation_config` 将完整配置解析为严格且字段冻结的 `EvaluationJobConfig`；runner、episode、runtime 和 execution 组件只接收该类型或其子模型，不读取 `DictConfig`。`env` 子树是传给 MetaDrive 的开放第三方配置，保留为普通映射；本项目消费的 horizon、traffic 和 evaluation 字段仍必须由顶层模型交叉校验。

仿真真实状态、模型观测、模型预测和能耗记录必须分别保存。模型预测不得覆盖仿真状态，不同能耗指标不得静默互换或混合累计。业务代码不得从 `ref/` 导入运行时实现。

推理由单进程、单设备 Lightning Fabric 运行时装配，不使用 Trainer。MetaDrive 观测适配器只生成 CPU raw tensor；Fabric 统一负责观测传输、模型设备和 forward 精度。普通 evaluation 在 `evaluation.execution.vector_env_slots=null` 时保持单环境串行；设置正 slot 数时，同一 evaluation job 用 `VectorMetaDriveEnv` 以固定 slots 同时推进 scenario，并由同一主进程 planner runtime 做 batch inference。每个 slot 在独立 `spawn` worker 内持有一个 `TrajectoryMetaDriveEnv`、observation adapter、traffic history 和 map cache。主进程只传入 scenario/`float32 [80,4]` trajectory，并取回单 observation、reward、termination、`TrajectoryExecutionRecord`、trace 所需的 planning anchor/traffic audit/warmup record 与 env/observation/IPC timing；worker 不加载 planner 或 CUDA state。vector evaluation 必须使用 `execution.mode=serial`、关闭 video，因而不与 Joblib job-level parallelism 嵌套。

raw observation 在 CPU 边界按当前 observation/trace 契约的 shape、dtype 和有限性完整校验并原值写入 trace；Fabric 设备副本只供计算使用，可按 resolved mixed precision 转为 FP16/BF16。batch runtime 每个规划周期只同步把 ego execution trajectories `[B,T,4]` 转为执行所需的 host `float32`；串行 evaluation 与 rollout 入口从该 batch result 取其唯一 slot。完整 prediction、初始噪声、reference 与 guidance diagnostics 则在独立 CUDA transfer stream 排队，并由 artifact/replay 调用方在 simulator step 后显式取得 audit result、转为 trace 约定 dtype 后检查有限性。

Benchmark 是 `scripts/` 下的非安装诊断工具，不属于 evaluation 或 RL 的稳定公共 API。它只通过显式关闭的 profiling hook 在现有 planner、vector worker 和 PPO 边界计时，不改变调度、随机流或数值语义；关闭 profiling 时不得增加 CUDA 同步。planner 的同步 execution trajectory device→CPU 与延迟 audit transfer 分开解释；vector worker 的 command wait、busy、IPC、busy imbalance 和 batch wall 是不同边界，不得互相换名；policy rollout 的 decision 与 bootstrap batch 必须分开统计，serial/vector 对照必须调用各自真实 collector。evaluation 模式汇总只接受 execution mode 与 resolved config/runtime metadata 一致、且 scenario workload 完全相同的输入。所有测量规模、warmup、正式样本和 repeats 均由 benchmark 配置显式给出，保留原始样本及统计中位数/极值；结果不得反向成为未验证的运行时默认值。

跨评测作业的进程并行由 ADR 0012 定义。traffic matrix 固定使用两个 Joblib `loky` worker；每个进程仍保持一个 MetaDrive、一个单设备 Fabric runtime 和一个 artifact writer。CPU 显式验证线程预算；CUDA 只允许两个进程共享一张可见 GPU，并要求确定性配置和正式运行前显存 preflight。smoke、no-traffic 和普通 full 运行保持串行，多 GPU 调度不属于该入口。

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

## 坐标、时间与单位

* 模型局部坐标以规划时刻 ego 后轴中心为原点，x 向前、y 向左。
* MetaDrive vehicle center 与后轴中心的偏移必须按车辆 heading 显式转换；地图、目标轨迹和实际车辆状态使用同一车辆中心约定。
* heading 使用 `[cos(h), sin(h)]`，角差使用最短有向角。
* 模型轨迹为 10 Hz 的 80 个未来点，共 8 s；MetaDrive 物理步长为 0.02 s，`decision_repeat=5`，对外子步为 0.1 s。
* evaluation 每个规划周期只执行前 5 点，即 0.5 s，规划频率为 2 Hz；policy-guided rollout 只执行第 1 点，即 0.1 s，规划频率为 10 Hz。两条入口不得混用 transition、reward、done 或 bootstrap 语义。
* 程序化地图限速配置使用 km/h，模型限速使用 m/s；单位转换只在地图适配边界执行一次。
* 能耗、距离、速度、加速度和角速度字段名必须显式标出单位。
* 场景全局平移或旋转后，等价状态应产生等价的局部模型输入。

## 地图与限速

地图输入来自完整 `NodeRoadNetwork`，不得用默认 lidar observation 代替。地图 reset 时缓存完整 lane 几何并建立 STRtree；运行时索引只做保守候选粗筛，最终仍调用 MetaDrive `lane.distance` 精确过滤并按距离、稳定 ID 确定性排序。每条 lane 沿完整弧长重采样 20 点，再截断或零填充为 70 条普通 lane 和 25 条 route lane。route lane 由 navigation checkpoints 的连续 road edge 识别。

程序化地图没有可靠交通灯状态：有效 lane 使用 unknown `[0,0,0,1]`，padding lane 保持全零。MetaDrive 横向坐标正方向指向右侧，适配器必须显式转换成模型的左边界优先语义。

MetaDrive 0.4.3 PGMap 用精确 `1000 km/h` 表示未设置限速。环境 reset 后必须只替换该精确哨兵，保留已有有限、正值且不超过 `130 km/h` 的真实限速，确认哨兵全部消失并记录替换与保留数量。地图适配器若仍发现哨兵或非法限速，必须指出 lane 和原始值并失败；不得统一覆盖真实限速或猜测默认值。

能耗 benchmark 可选 `programmatic_lane_speed_limit_profile_kmh`。它按 PGMap 初始 block 之后的生成 block 顺序指定限速，长度必须与该 block 数严格一致；只覆盖这些 block 中的未设置限速哨兵，已有显式限速仍保留。reset audit 必须记录 profile 及实际应用 lane 数。

lane 长度与宽度必须接受 Python 或 NumPy 的真实数值标量，同时拒绝 bool、数组、非有限值和非正值。

## 交通观测

MetaDrive adapters produce one CPU `SingleObservation` without a planner batch dimension.
`collate_observations` is the sole batch boundary: it stacks a same-schema sequence into a
`BatchObservation` with leading `[B, ...]` dimensions. B=1 planner, evaluation, and rollout
paths use this collator; `SingleObservation` / `BatchObservation` 的字段、shape 和 dtype 由
jaxtyping `TypedDict` 表达，collator 运行时只拒绝空序列，不重复校验 schema 或 tensor 类型。
它不包含 MetaDrive、planner 或 device-placement 逻辑。

`MetaDriveObservationAdapter` 使用 reset 帧和连续 0.1 s 交通快照构造严格的 21 帧历史。每个快照必须在仿真观测时刻捕获为不可变值；捕获边界校验 MetaDrive 参与者类型、位置、heading、速度和尺寸，内部编码链路直接消费已捕获的明确类型。批量追加历史只校验时间连续性，并在整批确认后一次性提交，失败时历史保持不变。

当前帧查询半径内的对象按距离和 ID 排序。历史对象缺帧时使用从当前帧向过去保持最近可用状态的官方填充语义；当前帧不存在的对象不得被选入。首次交通推理前，当前实现固定 ego 并推进背景交通 20 个 0.1 s 子步，连同 reset 帧形成 21 帧历史。该预热不计入正式指标，必须保存状态、奖励、终止标志及动态/静态对象数量；ego 位移达到 `1e-3 m` 或预热提前结束时必须失败。

`NoTrafficMetaDriveObservationAdapter` 只允许显式满足 `traffic_density=0`、`random_traffic=false`、`accident_prob=0` 的场景。reset 后若存在任何动态或静态交通对象必须失败；邻车历史和静态物体字段为全零 padding。该入口不得用于有交通场景。

`VectorMetaDriveEnv.reset` 在 worker 内完成相同的 adapter reset 和 history warmup，再返回单 observation 与 reset 时的 route completion；`reset_at` 和 `step_at` 只影响指定 slot。slot scenario 的 map 必须与该 worker 的环境配置一致；worker 异常带回 slot 和 operation，并使整个 vector run 明确失败。

## 扩散与随机性

预训练模型使用连续时间线性 VP-SDE 并预测 `x_start`。Hydra 必须显式选择 sampler；默认 `dpm10` 保持官方 baseline：从 `0.5 * N(0,I)` 的未来噪声开始，执行 10 步二阶 multistep DPM-Solver++，结束于 `t=1/1000` 后额外预测一次 `x_start`。其公式和初始尺度不得由配置覆盖。

可选 `ddim5` 从标准高斯未来噪声开始，在 `t = [1.0, 0.8, 0.6, 0.4, 0.2]` 预测 `x_start`，依次转移到 `[0.8, 0.6, 0.4, 0.2, 0.0]`。该均匀连续时间子序列是本项目复现决定，不是论文公开事实。评测配置使用 `ddim_stochasticity=0`；非零值必须显式提供同设备 `torch.Generator`。DDIM transition state 保持初始状态的 dtype；mixed-precision denoiser 输出在 sampler 边界显式转换到该 dtype。`0.5 * N(0,I)` DDIM 仅作为带独立 parity 标签的项目隔离变体，不得解释为 PlannerRFT parity。

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

## Exploration Policy

Exploration Policy 由 fixed-slot vector rollout collector（B=1 也走同一契约）接入 learned-guidance closed loop，并由 PPO optimizer 更新；它不接入既有 evaluation runner。它的输入固定为：冻结 scene tokens `[B,N,H]` 及 bool padding mask、冻结 route/navigation token `[B,1,H]` 及 bool validity/padding mask，以及 ego-local physical reference trajectory `float [B,80,4]`。reference 使用 10 Hz、米和 `[cos(h),sin(h)]`。所有 feature 必须同 batch、dtype 和 device，且有限；每个 batch item 至少有一个有效 context token。完整的有限值和有效 token 校验发生在 rollout 回传 CPU 的边界或显式调试校验中；policy/PPO 热路径只检查结构契约。

rollout runtime 在 eval-mode、所有参数 `requires_grad=False` 的官方模型上，通过 `prepare_policy_guidance()` 一次准备 scene/navigation encoding 和 reference。输出 detach 后才进入 policy；policy backward 不得为 planner 产生 `.grad` 或改变 planner 权重。普通 planner `encode()`、fixed-guidance evaluation runner 和官方 checkpoint state-dict 层级保持不变。

reference 先经 MLP-Mixer 编码，再作为 query 对拼接后的 scene/navigation tokens 做 masked cross-attention；actor 与 value 共享融合 trunk。actor 产生 lateral/longitudinal 的严格正 `alpha,beta [B,2]`，value 严格为 `[B]`。reference 输入固定为 `[B,80,4]`；其余层数、维度、attention heads、dropout、初始 concentration 和最小 concentration 都是 Hydra 必需字段。concentration 使用 `softplus(raw)+minimum_concentration`；actor head 对称初始化为 `alpha=beta`，所以初始 guidance 均值为零，但 Beta 方差非零。

每个 planning cycle 只定义单候选 `K=1`。base action `u` 严格位于 `(0,1)^2`，rollout 保存 `u`；audited guidance action 为 `g=2u-1`，严格位于 `(-1,1)^2`。training sampling 使用调用者提供的独立 policy `torch.Generator` 和 `rsample`；普通 `sample` 也只消费该 generator；deterministic evaluation 使用 mean，不公开 mode。边界 action、非有限参数或非法 shape/dtype/device 立即失败，不得 clamp。两个维度独立时：

```text
joint_log_prob_u = sum_i log Beta(u_i; alpha_i, beta_i)
joint_log_prob_g = joint_log_prob_u - 2 * log(2)
joint_entropy_g = sum_i H(Beta_i) + 2 * log(2)
```

policy action generator 不得改变 PyTorch 全局 RNG，且与 map/noise seed 分离。policy export checkpoint 只包含其 checkpoint `format_version` 与 Exploration Policy 自身的 trainable state dict。该 `format_version` 属于 policy checkpoint 文件结构，不作为 evaluation 或 RL 产物的 schema-version 选择机制。网络结构、Beta 参数化、仿射映射和初始化见 ADR 0016，均为本项目复现决定，不得描述为 PlannerRFT 作者公开实现。

## Policy-guided rollout

rollout 使用独立的 single-device Fabric runtime 和 `guidance=orthogonal_policy`。fixed-slot collector 每个 planning round collate 所有 slot observations，执行一次 batched inference，再把 ego trajectory scatter 回对应 worker；一个 transition 先准备冻结 scene/navigation encoding 与 DDIM reference，以独立 policy generator 抽取 `rsample`，再用同一份 encoding、initial noise 与 DDIM transition randomness 完成 guided pass。planner 保持 eval/frozen；普通 evaluation 和固定 action guidance 路径不变。

collector 为每个 slot episode 构造两个按时间维组织的 TensorDict。每个 slot 维持独立、持久的 noise/action generator；episode terminal 或 truncation 只 reset 该 slot，且 GAE 不跨其 episode boundary。PPO training trajectory 仅含 policy context、guidance action、old transformed joint log-prob/value、reward、terminated/truncated 及其 GAE `next` 边界；已脱离计算图的 PPO fields 在其收集设备上保留至 PPO update。CPU audit/replay trajectory 保留 policy context、base/guidance action、old transformed joint log-prob/value、Beta 参数、initial noise、消费前 diffusion/policy RNG state、map/noise/policy seed、planning-cycle index、reward/audit 以及 terminated/truncated。training 与 audit 在 episode 收集完成时分别按各自 schema 校验，不做逐字段运行时 equality audit。rollout 每次决策只同步复制供 MetaDrive 执行的 ego trajectory；其余 audit/replay 字段在 simulator step 后经独立 CUDA transfer stream 等待并写入，且其 CPU copy 不决定 training trajectory 的设备生命周期。rollout 必须设置`trajectory_execution_steps=1`；evaluation config 要求 5。不保存 DDIM denoise chain。纯 truncation 与 rollout-limit tail 通过同一 batched bootstrap pass 保存各 slot 最终状态的冻结 critic value；terminal tail 保存零。

## GAE、PPO 与训练

每个 `RolloutEpisode` 的 PPO training TensorDict 直接包含当前 value 和 `next` 的 reward、done、terminated、state value；audit/replay TensorDict 不带 GAE 派生字段。episode 最后一项总是 GAE recursion boundary；真实 terminal 不 bootstrap，纯 truncation 与 rollout-limit tail bootstrap，advantage 不跨 episode 泄漏。

TorchRL `GAE` 产生未标准化 advantage 与 value target。多个 episode 仅在 GAE 后拼接，advantage 只在完整 PPO batch 上使用 sample standard deviation 标准化一次；少于两个样本、零方差或非有限统计立即失败。设备传输前 PPO batch 严格只选择 policy context、guidance action、old transformed joint log-prob、advantage 和 value target；`next`、reward、边界标记和 collection-time value 仅用于 GAE，不进入 `ClipPPOLoss` 更新。TorchRL `ClipPPOLoss` 的 actor TensorDict adapter 一次执行 `ExplorationPolicy` 并写入 `alpha`、`beta` 与当前 value；critic adapter 复用该 value，不重复执行 shared trunk。PPO ratio 使用保存的 old transformed joint log-prob，entropy 含仿射 Jacobian，不使用 DDIM transition probability。value loss 为 unclipped L2，policy、value 与 entropy loss 共同更新 policy actor head、value head 和共享 trunk。

可 sweep 的训练参数由 `RLTrainingJobConfig` 和 YAML profile 决定；PPO 固定使用完整 batch advantage normalization、unclipped L2 value loss、Adam 与 cosine scheduler。每个逻辑 slot 拥有独立、持久的 diffusion noise 与 policy action generator；实际 seeds 由固定 SeedSequence namespace 和 training seed 确定性派生。训练状态 checkpoint 使用 Fabric 保存 policy、optimizer、scheduler、PPO minibatch RNG、CPU/CUDA RNG 以及已完成 update 的 loop state，可从 checkpoint 恢复。训练前后冻结 planner 参数 hash 必须相同，只有 Exploration Policy 可被 optimizer 更新。

RL 训练输出与 evaluation 输出使用各自独立的数据边界。每个训练 episode 的 NPZ 在输出边界由 TensorDict 转换，保存 policy context、Beta 参数、base/guidance action、old log-prob/value、initial noise、两条 RNG state、reward/audit、done 和 seed；不保存 DDIM denoise chain。每次运行保存 resolved config、runtime metadata、tracked diff、policy export checkpoints、training-state checkpoint 和严格 summary。rollout 内部错误直接终止训练，不保存 partial trajectory。

## 轨迹执行

运动学接口的静态契约是有限的 `float32 [80,4]` ego 后轴局部轨迹。混合精度 forward 的 ego trajectory 必须在 evaluation/rollout host producer 中原值转换为 `float32`；完整 prediction 在 audit result 边界转换并保存 trace。shape、dtype、有限性和非零 heading 由 producer 测试保证，执行路径不做运行时重复校验；环境与运动学 policy 共享同一份已准备的世界轨迹。每个 0.1 s 子步将 vehicle center、heading、由相邻 center 有限差分得到的 velocity，以及由最短 heading 角差得到的 angular velocity 写入 MetaDrive；下一规划周期以最后实际状态为锚点。

`TrajectoryMetaDriveEnv` 的 Gym observation 只是一元素 `float32` 零数组；planner 输入必须由本项目 traffic/no-traffic observation adapter 构造。headless 无交通环境不初始化传感器；有交通环境只保留背景 IDM policy 必需的共享 lidar，不把它作为 planner observation。headless、非录制执行把每个 0.1 s 对外子步的五个 `0.02 s` Bullet substep 交给一次原生 `doPhysics` 调用；渲染或 MetaDrive episode recording 继续使用上游通用 step 路径。

该接口不生成 steering、throttle 或 brake，也不证明低层车辆动力学可执行性。原始轨迹必须原样执行和保存；不得平滑、裁剪、限幅、旋转、投影到中心线、选择最佳噪声 seed、切换回退控制器，或在异常时返回零轨迹。

`TrajectoryMetaDriveEnv.step` 在环境边界直接用本次执行缓冲、交通快照和 MetaDrive 终止字段构造不可变的 `TrajectoryExecutionRecord`，并通过 `info["trajectory_execution"]` 返回；不再先展开为逐字段 `info` 数组后由调用方二次解析。数组的 shape/dtype 由固定容量执行缓冲、jaxtyping 接口契约和 producer 测试保证；evaluation、RL、trace、rendering 和 summary 组件只消费该 record，不读取原始轨迹字段。

## 能耗记录

* 每种能耗指标使用独立名称、单位和累计边界。
* `metadrive_fuel_proxy` 按 MetaDrive `BaseVehicle` 的公式，从实际保存的 0.1 s kinematic execution trace 的相邻位置和执行速度重算；不得读取 kinematic `set_position()` 链路中的上游 `step_energy` 或 `episode_energy`。
* 该指标记录 `total_ml`、`distance_m` 和 `ml_per_km`，后者在零距离时为 null。失败回合若存在 partial trace 也记录已产生的能耗；空 trace 没有能耗值。
* MetaDrive 代理能耗与 FASTSim 等精细模型不得混用。
* 能耗结果必须关联实际执行 trace、采样间隔、车辆配置、场景特征和终止类型。
* 程序化地图没有原生坡度时不得假设高程信息。
* 结果必须明确限定在运动学执行条件；当前系统不定义最终奖励或优化目标。

## 评测与产物

每个评测作业必须保存 resolved config、Hydra overrides、runtime Git metadata、tracked diff、地图/场景 seed、噪声 seed、Fabric 请求与解析后的 accelerator/precision、实际设备、依赖环境和场景特征。`tracked_diff.patch` 必须存在，但干净工作区时允许为空。

每个回合至少保存 `summary.json` 和 `trace.npz`；开启视频时保存闭环 GIF。trace 必须包含 raw observation、初始噪声、完整联合预测、规划锚点、目标与实际状态、逐点误差、奖励、终止标志；交通回合还保存预热、对象 ID、交通数量、最近交通距离和历史有效性。

trace recorder 必须在回合开始时按最大 planning/warmup 容量，根据当前 trace field contract 预分配数组并直接写入槽位；`finalize()` 只暴露已记录切片。`trace.npz` 使用标准未压缩 NPZ，以降低长程写盘墙钟。

当前 evaluation 产物采用无版本的数据契约：

* job summary、episode summary 和 runtime metadata 使用严格且冻结的 Pydantic 模型，并设置 `extra="forbid"` 与 `allow_inf_nan=False`；读取时验证必需字段、字段类型及模型中实现的跨字段不变量。Completed episode summary 必须携带结构化 `energy`；partial failed episode 可携带对应能耗，empty failed episode 为 null。
* `trace.npz` 的字段集合、shape、dtype 和有限性由 `TRACE_FIELDS` / `validate_trace_arrays` 明确定义。
* trace 字段必须是预期的 NumPy array；缺失或未声明的数组都会导致验证失败。
* guided trace 的 guidance 数组必须完整出现或完整缺失，不能只保存其中一部分。
* 动态数组必须在 planning、simulator 和 warmup 轴上保持一致；实现还校验 trace status、planning-cycle 数、simulator-step 数、warmup 数、plan index 顺序、五步 execution prefix、terminal flag 位置、非负计数以及其他已实现的跨数组不变量。
* trace 显式保存 `complete`、`partial` 或 `empty` 状态、initial-state validity、普通及 route lane 的限速与有效性。
* JSON 和 NPZ 的当前契约均不依赖 `schema_version` 选择解析路径。读取器直接按照当前 Pydantic 模型或 trace array contract 验证输入，不合成缺失字段。

runner 只捕获显式 `EpisodeFailure`：保存阶段、异常类型、消息和 traceback 后继续同一作业的后续场景，作业最终标记失败且 CLI 返回非零。配置、checkpoint、Fabric 初始化、artifact IO 和未分类程序错误立即传播。

所有回合都进入分析，按场景特征、运行阶段和终止类型标注。失败回合不得删除，完成与失败回合不得在缺少标注时合并解释。策略比较必须使用相同地图、交通配置和噪声 seed。

矩阵汇总只接受预定义网格；部分矩阵必须是该网格的非空子集。缺少必需配置、summary、trace、metadata 或可视化产物时必须失败，不得把坏样本静默跳过。
