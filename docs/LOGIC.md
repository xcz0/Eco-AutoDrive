# 系统逻辑

本文只定义当前已实现系统必须遵守的逻辑与数据契约。实现进度见 `STATUS.md`，方案选择见 `DECISIONS.md`，尚未确定的研究设想见 `RESEARCH.md`。

## 系统边界

当前闭环链路为：

```text
MetaDrive 状态
  -> 官方格式观测
  -> Diffusion Planner
  -> 8 s 轨迹
  -> 执行前 0.5 s
  -> 更新 MetaDrive 状态
  -> 重新规划
```

仿真真实状态、模型观测、模型预测和能耗记录必须分开保存。模型预测不得覆盖仿真状态；不同能耗指标不得静默互换或混合累计。

当前实现覆盖无交通和车辆交通下的轨迹级运动学闭环。强化学习、道路预瞄扩展和精细能耗模型尚未形成实现契约。

## checkpoint 与上游契约

加载过程必须校验 missing/unexpected keys、张量形状和参数总数。`normalization.json` 属于模型输入分布契约，不是可选预处理。

业务代码不得从 `ref/` 导入。上游 nuPlan 数据提取和 planner 生命周期只用于理解字段语义，MetaDrive 边界由本项目实现。

## 模型张量契约

官方配置固定以下形状：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `ego_current_state` | `[B, 10]` | `x, y, cos(h), sin(h), vx, vy, ax, ay, steering, yaw_rate` |
| `neighbor_agents_past` | `[B, 32, 21, 11]` | 2 s、10 Hz 历史和类型编码 |
| `static_objects` | `[B, 5, 10]` | 位姿、尺寸和类型编码 |
| `lanes` | `[B, 70, 20, 12]` | 中心、切向、左右边界相对向量、信号 one-hot |
| `lanes_speed_limit` | `[B, 70, 1]` | m/s |
| `lanes_has_speed_limit` | `[B, 70, 1]` | 限速有效掩码 |
| `route_lanes` | `[B, 25, 20, 12]` | 局部路线车道 |
| `route_lanes_speed_limit` | `[B, 25, 1]` | 已生成，当前 decoder 不消费 |
| `route_lanes_has_speed_limit` | `[B, 25, 1]` | 已生成，当前 decoder 不消费 |
| `prediction` | `[B, 11, 80, 4]` | ego 与 10 个邻车的 `x, y, cos(h), sin(h)` |

`lanes` 单点通道顺序固定为：

```text
[center_x, center_y,
 delta_x, delta_y,
 left_boundary_dx, left_boundary_dy,
 right_boundary_dx, right_boundary_dy,
 traffic_green, traffic_yellow, traffic_red, traffic_unknown]
```

所有输入在交给模型前是 raw、未归一化张量。padding 必须严格全零，归一化后仍须恢复为全零。真实对象位于局部原点时，朝向、尺寸或类型字段必须避免其被误判为 padding。

动态体最多保留 32 个，按当前距离排序；历史为含当前帧的 21 帧。decoder 联合预测 ego 和前 10 个邻车。未来 80 帧不含当前帧，间隔 0.1 s。内部扩散状态含固定当前点，形状为 `[B, 11, 81, 4]`。

`RouteEncoder` 当前只读取 `route_lanes[..., :4]`，即中心和切向。不得声称现有模型已使用 route 边界、信号或 route 限速。

## 坐标、时间与单位

- 模型局部坐标以规划时刻 ego 后轴中心为原点，x 向前、y 向左；
- MetaDrive 车辆中心与后轴中心之间的偏移按车辆朝向显式转换；
- heading 以 `[cos(h), sin(h)]` 表示，角差使用最短有向角；
- 模型轨迹为 10 Hz、80 个未来点，共 8 s；
- MetaDrive 物理步长为 0.02 s，`decision_repeat=5`，环境子步为 0.1 s；
- 每次高层执行 5 个点，即 0.5 s 后重新规划，规划频率为 2 Hz；
- 程序化地图限速配置使用 `km/h`，模型限速使用 `m/s`；
- 限速单位转换只在地图适配边界执行一次；
- 能耗、距离、速度、加速度和角速度字段名必须标出单位。

同一场景全局平移或旋转后，局部模型输入应保持等价。地图、目标轨迹和实际车辆状态必须使用同一车辆中心约定。

## MetaDrive 地图逻辑

地图输入来自完整道路网络，而不是默认 lidar observation：

```text
NodeRoadNetwork
  -> 按 ego 距离筛选并排序
  -> 每条 lane 沿完整弧长重采样 20 点
  -> 转换到 ego 后轴局部坐标
  -> 截断并零填充为 70 条普通 lane / 25 条 route lane
```

route lane 由 `navigation.checkpoints` 的连续 road edge 识别。程序化地图没有可靠交通灯状态，有效 lane 使用 unknown `[0, 0, 0, 1]`，padding lane 保持全零。

MetaDrive 0.4.3 的 PGMap 使用精确 `1000 km/h` 表示未设置限速。环境 reset 后必须：

1. 读取所有 lane 原始限速；
2. 只替换精确 `1000 km/h` 的 lane；
3. 保留已有有限、正值且合法的真实限速；
4. 确认哨兵全部消失并记录替换与保留数量；
5. 地图适配器若仍发现哨兵，立即报错并指出 lane 和原始值。

环境和模型必须看到相同的限速语义，不得统一覆盖已有真实限速。

## 观测适配逻辑

`MetaDriveObservationAdapter` 从仿真对象和地图 API 构造官方字典，负责 ego、动态体 21 帧历史、静态物体、普通 lane 和 route lane。环境在 reset 后和每个 0.1 s 子步结束时保存不可变交通快照。

对象 ID、类型、位置、heading、速度和尺寸必须通过边界校验。当前帧 100 m 范围内对象按距离和 ID 确定性排序；历史不足时按官方从当前向过去的填充语义处理。

当前有交通入口在首次推理前固定 ego 并推进背景交通 20 个 0.1 s 子步，由 reset 帧和 20 个真实子步组成 21 帧历史。预热不计入正式评测指标，并必须保存预热状态、终止标志和交通数量。该方式会使背景车辆对静止 ego 作出反应，因此只作为当前已实现条件，不代表理想初始交通分布。

`NoTrafficMetaDriveObservationAdapter` 只用于明确无交通场景：

- 配置必须满足 `traffic_density=0`、`random_traffic=false`、`accident_prob=0`；
- reset 后再次拒绝动态或静态交通参与者；
- `neighbor_agents_past` 和 `static_objects` 使用全零 padding；
- `ego_current_state` 与官方推理入口保持一致；
- 该适配器不得用于有交通场景。

## 扩散与采样逻辑

预训练模型使用连续时间线性 VP-SDE，预测干净状态 `x_start`。官方 baseline 从 `0.5 * N(0, I)` 的未来噪声开始，运行 10 步 DPM-Solver++，最后执行 `denoise_to_zero=True`。

给定观测和初始噪声，baseline solver 是确定性的。每个规划周期必须从持久化、固定种子的 `torch.Generator` 取得新的标准正态噪声，并保存该噪声。

当前文档不规定强化学习所需的随机采样器或概率计算方式；相关方案在确定并实现后再加入系统契约。

## 轨迹执行逻辑

运动学接口接收完整 `float32 [80, 4]` 后轴局部轨迹。每次高层 `env.step()` 只执行前 5 点。对每个 0.1 s 子步：

- vehicle center 位于转换后的目标 world center；
- heading 等于目标 world heading；
- velocity 由相邻 center 的有限差分得到；
- angular velocity 由相邻 heading 的最短角差得到；
- 下一规划周期锚点等于上一周期最终实际状态。

该接口不生成 steering、throttle 或 brake，也不证明低层动力学可执行性。当前阶段仍允许用它研究轨迹规划和相对能耗，但结论必须明确限定为运动学执行条件。

模型原始轨迹必须原样执行和保存，不得平滑、裁剪、旋转、中心线投影、选择最佳噪声 seed、切换回退控制器或在异常时返回零轨迹。

## 能耗记录逻辑

当前阶段只规定记录边界，不规定最终奖励或优化目标：

- 每种能耗指标必须使用独立名称、单位和累计边界；
- MetaDrive 代理能耗与 FASTSim 等精细模型不得混用；
- 能耗结果必须关联实际执行 trace、采样间隔、车辆配置和场景特征；
- MetaDrive 程序化地图没有原生坡度时，不得假设存在高程信息；
- 运动学执行条件及其动力学局限必须随能耗结果一起说明。

具体能耗模型、奖励形式和约束关系尚未确定，见 `RESEARCH.md`。

## 评测与产物逻辑

每次评测必须记录 resolved config、Hydra overrides、地图或场景 seed、噪声 seed、设备、依赖环境和场景特征。

每个场景至少产生：

- `summary.json`：终止状态、距离、速度、路线完成率、限速审计和执行误差；
- `trace.npz`：raw observation、初始噪声、完整预测、规划锚点、目标与实际状态、逐点误差、奖励和终止标志；
- 可选闭环 GIF；
- resolved config 和 Hydra overrides。

有交通场景还应保存预热状态、对象 ID、交通数量、最近交通距离和历史有效性。

所有场景都必须进入分析，不限于成功到达回合。报告应按以下维度分类：

- 场景特征：直道、曲线、变道、合流、限速变化和交通密度；
- 运行阶段：预热、正常闭环、背景交通异常阶段；
- 终止类型：到达、时间截断、碰撞、出界或运行错误；
- 可比指标：能耗、速度、有效进度、舒适性和安全事件。

失败场景不能删除，但不得与完成场景在缺少标注的情况下合并解释。策略比较应使用相同地图、交通配置和噪声种子；长时背景车辆异常结果必须单独标注，不用于否定中程基本驾驶能力。
