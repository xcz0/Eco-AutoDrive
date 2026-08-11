# System Contract

本文是当前已实现系统行为的规范性 reference。它不规定尚未实现的强化学习、道路预瞄扩展、低层控制器或最终能耗模型。

## 系统边界

当前闭环链路为：MetaDrive 状态构造成官方格式 raw observation，冻结的官方 EMA Diffusion Planner 生成 8 s 联合轨迹，环境执行 ego 轨迹前 0.5 s，然后从实际仿真状态重新规划。

仿真真实状态、模型观测、模型预测和能耗记录必须分别保存。模型预测不得覆盖仿真状态，不同能耗指标不得静默互换或混合累计。业务代码不得从 `ref/` 导入运行时实现。

## Checkpoint 与模型输入

- 加载官方资产时必须校验 metadata keys、EMA keys、张量形状、EMA tensor 数和参数总数；加载必须使用严格 state-dict 匹配，不提供兼容回退。
- `args.json` 中的 observation/state normalization 是 checkpoint 输入分布契约的一部分，缺失或不匹配时必须失败。
- 冻结的预训练 planner 始终处于 eval mode；该入口不得进入训练模式。
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

地图输入来自完整 `NodeRoadNetwork`，不得用默认 lidar observation 代替。lane 按 ego 后轴距离筛选并确定性排序，每条 lane 沿完整弧长重采样 20 点，再截断或零填充为 70 条普通 lane 和 25 条 route lane。route lane 由 navigation checkpoints 的连续 road edge 识别。

程序化地图没有可靠交通灯状态：有效 lane 使用 unknown `[0,0,0,1]`，padding lane 保持全零。MetaDrive 横向坐标正方向指向右侧，适配器必须显式转换成模型的左边界优先语义。

MetaDrive 0.4.3 PGMap 用精确 `1000 km/h` 表示未设置限速。环境 reset 后必须只替换该精确哨兵，保留已有有限、正值且不超过 `130 km/h` 的真实限速，确认哨兵全部消失并记录替换与保留数量。地图适配器若仍发现哨兵或非法限速，必须指出 lane 和原始值并失败；不得统一覆盖真实限速或猜测默认值。

lane 长度与宽度必须接受 Python 或 NumPy 的真实数值标量，同时拒绝 bool、数组、非有限值和非正值。

## 交通观测

`MetaDriveObservationAdapter` 使用 reset 帧和连续 0.1 s 交通快照构造严格的 21 帧历史。每个快照必须在仿真观测时刻捕获为不可变值，并校验对象 ID、类型、位置、heading、速度和尺寸；批量追加历史必须先验证整批，再一次性提交，失败时历史保持不变。

当前帧查询半径内的对象按距离和 ID 排序。历史对象缺帧时使用从当前帧向过去保持最近可用状态的官方填充语义；当前帧不存在的对象不得被选入。首次交通推理前，当前实现固定 ego 并推进背景交通 20 个 0.1 s 子步，连同 reset 帧形成 21 帧历史。该预热不计入正式指标，必须保存状态、奖励、终止标志及动态/静态对象数量；ego 位移达到 `1e-3 m` 或预热提前结束时必须失败。

`NoTrafficMetaDriveObservationAdapter` 只允许显式满足 `traffic_density=0`、`random_traffic=false`、`accident_prob=0` 的场景。reset 后若存在任何动态或静态交通对象必须失败；邻车历史和静态物体字段为全零 padding。该入口不得用于有交通场景。

## 扩散与随机性

预训练模型使用连续时间线性 VP-SDE 并预测 `x_start`。官方 baseline 从 `0.5 * N(0,I)` 的未来噪声开始，执行 10 步二阶 multistep DPM-Solver++，最后 denoise 到零时刻。

给定 observation 和初始噪声，baseline sampler 是确定性的。每个回合创建一个由噪声 seed 初始化的持久化 `torch.Generator`；每个规划周期从中取得新的标准正态噪声，并把该噪声完整保存。地图 seed 与噪声 seed 必须分别记录。

## 轨迹执行

运动学接口只接收有限的 `float32 [80,4]` ego 后轴局部轨迹。每个 0.1 s 子步将 vehicle center、heading、由相邻 center 有限差分得到的 velocity，以及由最短 heading 角差得到的 angular velocity 写入 MetaDrive；下一规划周期以最后实际状态为锚点。

该接口不生成 steering、throttle 或 brake，也不证明低层车辆动力学可执行性。原始轨迹必须原样执行和保存；不得平滑、裁剪、限幅、旋转、投影到中心线、选择最佳噪声 seed、切换回退控制器，或在异常时返回零轨迹。

## 能耗记录

- 每种能耗指标使用独立名称、单位和累计边界。
- MetaDrive 代理能耗与 FASTSim 等精细模型不得混用。
- 能耗结果必须关联实际执行 trace、采样间隔、车辆配置、场景特征和终止类型。
- 程序化地图没有原生坡度时不得假设高程信息。
- 结果必须明确限定在运动学执行条件；当前系统不定义最终奖励或优化目标。

## 评测与产物

每个评测作业必须保存 resolved config、Hydra overrides、runtime Git metadata、tracked diff、地图/场景 seed、噪声 seed、设备、依赖环境和场景特征。`tracked_diff.patch` 必须存在，但干净工作区时允许为空。

每个回合至少保存 `summary.json` 和 `trace.npz`；开启视频时保存闭环 GIF。trace 必须包含 raw observation、初始噪声、完整联合预测、规划锚点、目标与实际状态、逐点误差、奖励、终止标志；交通回合还保存预热、对象 ID、交通数量、最近交通距离和历史有效性。

所有回合都进入分析，按场景特征、运行阶段和终止类型标注。失败回合不得删除，完成与失败回合不得在缺少标注时合并解释。策略比较必须使用相同地图、交通配置和噪声 seed。

矩阵汇总只接受预定义网格；部分矩阵必须是该网格的非空子集。缺少必需配置、summary、trace、metadata 或可视化产物时必须失败，不得把坏样本静默跳过。
