# System contract

本文件及分篇记录当前已实现的数据与执行契约。按任务选择下表入口；分篇是同一套契约的唯一归属位置。领域误解先看 [domain.md](domain.md)，精确定义见 [CONTEXT.md](../../CONTEXT.md)。

## 按任务读取

| 涉及的工作 | 读取位置 |
| --- | --- |
| job composition、资源预算、Fabric、serial/vector/job-parallel、host transfer、benchmark | [运行时与配置](contracts/runtime.md) |
| checkpoint、observation shape/padding、地图/交通、扩散随机流、reference guidance | [模型输入与规划](contracts/planner.md) |
| policy 动作与概率、rollout/reward、GAE/PPO、checkpoint 评测、训练跟踪 | [Policy、rollout 与训练](contracts/training.md) |
| fixed-batch 诊断、校准、guidance intervention、scalar protocol、离线分析与报告 | [实验工具与离线分析](contracts/experiments.md) |
| 坐标、时间、轨迹执行、能耗流、evaluation 指标与产物 | 本文件下文 |

## 使用与维护

- 当前实现事实以代码和测试、机器可读配置为先；与文档冲突时明确指出，按本次任务修正，不能融合成第三种解释。
- 持久的数据或执行语义变化只更新对应章节；新增分篇时在上表给出明确读取条件，并更新受影响的链接。
- 设计理由归 [ADR](../adr/)，未确定的方法归 [research](../research/)，实际运行配置、结果和 provenance 归[实验记录](../experiments/README.md)，待实现任务与验收标准归 GitHub Issues。
- 仅记录当前实现；历史产物行为留在对应实验记录或 ADR。

## 系统边界

当前 evaluation 闭环链路为：MetaDrive 状态构造成官方格式 raw observation，冻结的官方 EMA Diffusion Planner 生成 8 s 联合轨迹，环境执行 ego 轨迹前 0.5 s，然后从实际仿真状态重新规划。

仿真真实状态、模型观测、模型预测和能耗记录必须分别保存。模型预测不得覆盖仿真状态，不同能耗指标不得静默互换或混合累计。业务代码不得从 `ref/` 导入运行时实现。

配置、持久化文件以及 MetaDrive、TorchRL、Diffusers、Fabric 等第三方返回值只在首次进入项目 typed domain 的边界校验和转换一次；下游受控数据流依赖明确类型、生产者测试与本节规定的 shape/单位契约，不为静态类型收窄重复执行 `isinstance` 或 Optional 状态检查。有限性、随机流、冻结参数及其他会改变实验语义的显式校验不受此规则影响。

## 坐标、时间与单位

* 模型局部坐标以规划时刻 ego 后轴中心为原点，x 向前、y 向左。
* MetaDrive vehicle center 与后轴中心的偏移必须按车辆 heading 显式转换；地图、目标轨迹和实际车辆状态使用同一车辆中心约定。
* heading 使用 `[cos(h), sin(h)]`，角差使用最短有向角。
* 模型轨迹为 10 Hz 的 80 个未来点，共 8 s；MetaDrive 物理步长为 0.02 s，`decision_repeat=5`，对外子步为 0.1 s。
* evaluation 每个规划周期只执行前 5 点，即 0.5 s，规划频率为 2 Hz；policy-guided rollout 只执行第 1 点，即 0.1 s，规划频率为 10 Hz。两条入口不得混用 transition、reward、done 或 bootstrap 语义。
* 这些共享 ABI 值仅定义于 `eco_planner.contracts`。其中 `TRAFFIC_HISTORY_FRAMES=21` 包含当前帧，`TRAFFIC_HISTORY_WARMUP_STEPS=20` 是形成该完整历史所需的过去子步数；MetaDrive physics step 与 decision repeat 必须显式验证其乘积等于 0.1 s。
* 程序化地图限速配置使用 km/h，模型限速使用 m/s；单位转换只在地图适配边界执行一次。
* 能耗、距离、速度、加速度和角速度字段名必须显式标出单位。
* 场景全局平移或旋转后，等价状态应产生等价的局部模型输入。

## 轨迹执行

运动学接口的静态契约是有限的 `float32 [80,4]` ego 后轴局部轨迹。混合精度 forward 的 ego trajectory 必须在 evaluation/rollout host producer 中原值转换为 `float32`；完整 prediction 在 audit result 边界转换并保存 trace。shape、dtype、有限性和非零 heading 由 producer 测试保证，执行路径不做运行时重复校验；环境与运动学 policy 共享同一份已准备的世界轨迹。每个 0.1 s 子步将 vehicle center、heading、由相邻 center 有限差分得到的 velocity，以及由最短 heading 角差得到的 angular velocity 写入 MetaDrive；下一规划周期以最后实际状态为锚点。

`MetaDriveBackend` 的 Gym observation 只是一元素 `float32` 零数组，仅满足 MetaDrive 自身接口；它不是 planner environment，仅为渲染和 MetaDrive-native audit 通过 façade 暴露。planner 输入必须通过 `MetaDriveEnvSlot.reset()` 返回的首 state 或 `step()` 返回的下一 state，由本项目 traffic/no-traffic observation pipeline 和 canonical builder 构造。headless 无交通环境不初始化传感器；有交通环境只保留背景 IDM policy 必需的共享 lidar，不把它作为 planner observation。headless、非录制执行把每个 0.1 s 对外子步的五个 `0.02 s` Bullet substep 交给一次原生 `doPhysics` 调用；渲染或 MetaDrive episode recording 继续使用上游通用 step 路径。

该接口不生成 steering、throttle 或 brake，也不证明低层车辆动力学可执行性。原始轨迹必须原样执行和保存；不得平滑、裁剪、限幅、旋转、投影到中心线、选择最佳噪声 seed、切换回退控制器，或在异常时返回零轨迹。

`MetaDriveEnvSlot.step()` 调用 trajectory executor：executor 使用 canonical local-to-world conversion，逐个请求 backend 的 0.1 s transition；stateful transition extractor 从连续 ego state、目标点、交通快照、route/lane 与 typed MetaDrive termination outcome 生成 objective-neutral `TransitionMetrics`。executor 在 terminal/truncation 时停止并生成 domain-owned `TrajectoryExecutionResult(execution, metrics, terminated, truncated)`；slot 提交 traffic frames、构造下一 `EnvSlotState`，再作为原子 `EnvSlotStep` 返回。`TrajectoryExecutionRecord` 只保存 start/target/actual trajectory、traffic、route 与 termination 等执行事实。backend、executor、slot、worker 都不携带 builtin reward、训练 reward callback 或 reward-specific audit。RL collector 从结果的单步 metrics 生成 `RewardResult`；evaluation recorder 从同一结果映射客观 trace 字段，不保存 reward。数组的 shape/dtype 由固定容量执行缓冲、jaxtyping 接口契约和 producer 测试保证。

## 能耗记录

* 每种能耗指标使用独立名称、单位和累计边界。`envs.domain.energy` 拥有 `EnergyTrace`、`EnergyMetrics` 与 provider protocol；provider 接收有限、非负且时间严格递增的 `EnergyTrace(time_s[N+1], speed_mps[N+1], step_distance_m[N])`，返回带 metric 名称、实际距离、可选 J 和可选 mL 的 `EnergyMetrics`；Wh 与各 per-km quantity 只由该统一结果派生，零距离时 per-km 为未定义。
* MetaDrive native audit 直接保存上游每个 simulator 子步产生的 `step_energy` 和 reset-bounded `episode_energy`，单位均为 mL；这些事实保存在 `TransitionMetricInput`，evaluation trace 映射为 `warmup_native_*` / `executed_native_*`。当前 kinematic waypoint phase ordering 下这些值可能恒为零，只能审计上游 phase boundary，不进入 reward 或 evaluation energy summary。
* `envs.domain.energy.MetaDriveFuelProxyProvider` 在 environment execution boundary 使用相邻实际 center position、实际执行速度和 MetaDrive 原公式逐子步计算；`TransitionMetrics` 保存统一 `EnergyMetrics`，reward/evaluation 在各自 audit/artifact 边界映射为既有 `*_fuel_proxy_step_energy_ml` 与配套 `*_step_distance_m`。evaluation summary 的 `total_ml`、`distance_m` 和 `ml_per_km` 只聚合 trace 中该流，不得重算公式或改用 native 流。

* 该指标记录 `total_ml`、`distance_m` 和 `ml_per_km`，后者在零距离时为 null。失败回合若存在 partial trace 也记录已产生的能耗；空 trace 没有能耗值。
* `envs.domain.fastsim` 的 `fastsim_fuel_energy` 仅在完整实际执行轨迹结束后离线运行。首个 adapter 固定使用 FASTSim 3.0.6 内置 conventional `2012_Ford_Fusion.yaml`，由显式 grade、环境温度和初始海拔构造 cycle，输出 fuel energy J/Wh，不推导 fuel mL。它不进入在线 reward 或默认 evaluation artifact；MetaDrive proxy 与 FASTSim 不得相加、替换名称或混合解释。具体边界见 ADR 0032。
* 能耗结果必须关联实际执行 trace、采样间隔、车辆配置、场景特征和终止类型。
* 程序化地图没有原生坡度时不得假设高程信息。
* 结果必须明确限定在运动学执行条件；两个 PlannerRFT reward profile（`plannerrft_energy_v1` 与 `plannerrft_no_energy_v1`）都是 smoke-only PlannerRFT-style MetaDrive adaptation，不代表 PlannerRFT/nuPlan scorer parity、真实车辆舒适性或已验证的节能目标。
* 若 completed episode 的执行距离为零，`ml_per_km` 为 null；固定 matrix 不对该未定义指标 bootstrap，而是明确失败。

## 评测与产物

每个评测作业必须保存 resolved config、Hydra overrides、runtime Git metadata、地图/场景 seed、噪声 seed、Fabric 请求与解析后的 accelerator/precision、实际设备、依赖环境和场景特征。所有运行入口均不再保存 tracked diff 或复制 Python source；Git head、branch、dirty status 与运行环境仍由现有 metadata 记录，checkpoint 身份和 replay 核验保留。正式结果必须来自 clean commit，见实验记录规范；不增加运行前检查框架。

每个回合至少保存 `summary.json` 和 `trace.npz`；开启视频时保存闭环 GIF。trace 必须包含 raw observation、初始噪声、完整联合预测、规划锚点、目标与实际状态、逐点误差、逐子步 wrapped route heading error、逐子步 native MetaDrive energy、execution-recomputed fuel proxy、实际 distance 和终止标志；交通回合还保存预热、对象 ID、交通数量、最近交通距离和历史有效性。

`evaluation.artifacts.summary.compute_episode_metrics` 是通用 closed-loop metric 的唯一计算路径：它从完整 execution trace 与最终 typed execution record 生成一个 `evaluation_episode` 聚合单位的 `EpisodeMetrics`。其中 distance 是 initial state 到每个实际 executed state 的几何路径长度；mean speed 与 stopped fraction 分别是 simulator steps 上的算术平均和速度小于 `0.1 m/s` 的比例；route completion 是最终 execution record 的值；arrival、collision 与 out-of-road 是最终 terminal outcome；total energy、energy distance 与 energy intensity 只从 execution-recomputed fuel-proxy trace 流聚合。wrong-direction 指标从 trace 的 `executed_route_heading_errors_rad`（每个执行子步相对 route forward tangent 的 wrapped heading error，与 reward safety gate 同源）计算：任何子步误差严格大于 π/2 即 `wrong_direction=true`，`wrong_direction_fraction` 为该子步比例；该指标只记录，不改变 termination。summary、matrix 与实验报告消费这些 typed episode/training summaries，不得另从 trace array 或 resolved config 重算同名通用指标。matrix report 的 episode 行包含 `wrong_direction`，scenario 统计包含 `wrong_direction_rate`。

trace recorder 必须在回合开始时按最大 planning/warmup 容量，根据当前 trace field contract 预分配数组并直接写入槽位；`finalize()` 只暴露已记录切片。`trace.npz` 使用标准未压缩 NPZ，以降低长程写盘墙钟。

当前 evaluation 产物只支持当前代码定义的数据契约，不携带格式版本字段，也不提供历史产物迁移或兼容转换：

* job summary、episode summary 和 runtime metadata 使用严格且冻结的 Pydantic 模型，并设置 `extra="forbid"` 与 `allow_inf_nan=False`。Completed episode summary 只在嵌套 `metrics` 中保存通用 episode-level 指标；partial failed episode 可携带对应能耗，empty failed episode 为 null。
* `trace.npz` 的字段集合、shape、dtype 和有限性由 `TRACE_FIELDS` / `validate_trace_arrays` 明确定义。
* trace 字段必须是预期的 NumPy array；缺失或未声明的数组都会导致验证失败。
* guided trace 的 guidance 数组必须完整出现或完整缺失，不能只保存其中一部分。
* 动态数组必须在 planning、simulator 和 warmup 轴上保持一致；实现还校验 trace status、planning-cycle 数、simulator-step 数、warmup 数、plan index 顺序、五步 execution prefix、terminal flag 位置、非负计数以及其他已实现的跨数组不变量。
* trace 显式保存 `complete`、`partial` 或 `empty` 状态、initial-state validity、普通及 route lane 的限速与有效性。
* `evaluation.artifacts.trace` 只保存轻量字段声明和 I/O structural validation；依赖 Torch/MetaDrive 的在线预分配与记录由 `evaluation.episodes.recorder` 拥有。reader 在 I/O 边界完整校验一次，`evaluation.artifacts.io` 只补充 trace 与 typed episode result 的计数/status 对齐、route/traffic、seed 配对和接口误差语义；实验的 retention、safety 或统计接受规则保留在各自 `experiments` 模块。
* `evaluation.engine` 是在线 job 编排入口；serial/vector 回合控制流位于 `evaluation.episodes`，planner adapter、runtime 和 decision boundary 位于 `evaluation.inference`，`evaluation.artifacts.report` 是通用离线矩阵报告入口，typed reader/writer 位于 `evaluation.artifacts.io`。仓库内消费者通过 `eco_planner.evaluation` 的延迟公开接口访问这些能力；读取 summary、trace 或 report 不得加载 Torch、MetaDrive/Panda3D 或 GIF rendering。
* validation 与 report 不得把 Hydra resolved config 当作第二个 metric/result source；resolved config 仅保留为运行 provenance。

engine 只捕获显式 `EpisodeFailure`：保存阶段、异常类型、消息和 traceback 后继续同一作业的后续场景，作业最终标记失败且 CLI 返回非零。配置、checkpoint、Fabric 初始化、artifact IO 和未分类程序错误立即传播。

所有回合都进入分析，按场景特征、运行阶段和终止类型标注。失败回合不得删除，完成与失败回合不得在缺少标注时合并解释。策略比较必须使用相同地图、交通配置和噪声 seed。

矩阵汇总只接受预定义网格；部分矩阵必须是该网格的非空子集。缺少必需配置、summary、trace、metadata 或可视化产物时必须失败，不得把坏样本静默跳过。
