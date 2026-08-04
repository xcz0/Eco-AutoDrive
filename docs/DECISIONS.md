# 设计决策

本文记录会影响系统边界或实验解释的重要选择。`已实施` 表示当前代码已采用；`计划采用` 只表示
方向已确定，不代表功能可用。

## 决策索引

| ID | 决策 | 状态 |
| --- | --- | --- |
| D-001 | 保持官方 Diffusion Planner checkpoint 兼容 | 已实施 |
| D-002 | MetaDrive 数据适配重写，`ref/` 不作为运行时依赖 | 已实施 |
| D-003 | 首阶段使用 2 Hz 重规划、10 Hz 运动学轨迹执行 | 已实施 |
| D-004 | 程序化 lane 限速必须显式配置并在两层边界校验 | 已实施 |
| D-005 | 无交通适配器与通用有交通适配器分离 | 已实施 |
| D-006 | 原始轨迹不做后处理或失败回退 | 已实施 |
| D-007 | baseline sampler 与 DPPO stochastic sampler 分离 | 部分实施 |
| D-008 | 长程道路预瞄以零初始化残差注入 | 计划采用 |
| D-009 | 先用 MetaDrive 代理能耗，再用 FASTSim 精细复核 | 计划采用 |
| D-010 | RL 主循环使用 Lightning Fabric | 计划采用 |
| D-011 | Linux/Docker/CUDA 是正式验收环境 | 已实施 |

## D-001 保持官方 checkpoint 兼容

**选择**：保留官方 `Diffusion_Planner`、Encoder、DiT、Mixer、初始化和归一化的参数层级与数值
语义，严格加载官方 EMA；MetaDrive、DPPO 和能耗能力从边界扩展。

**理由**：研究目标是评估预训练 Diffusion Planner 经能耗 RL 微调后的变化。若同时改变网络主体，
无法把差异归因于环境适配或 DPPO，并会失去官方权重基线。

**未选方案**：从头训练一个结构相似的 planner，或为代码整洁重排 DiT 残差结构。这些方案都会
改变权重语义和基线行为。

**后果**：80 帧未来、25 条 route lane、20 点 lane 等维度受 checkpoint 约束；任何架构扩展都
必须保持原路径初始行为可对照。

## D-002 重写仿真器边界，不从 `ref/` 运行

**选择**：模型主体做兼容移植；ego/agent/map/route 提取和 planner 生命周期基于 MetaDrive API
重新实现。`ref/` 只用于阅读和数值对照。

**理由**：上游数据处理直接依赖 nuPlan 类型、地图 API 和 Pacifica 参数，无法作为通用模型库使用。
把上游快照加入运行时会造成隐式依赖和难以固定的许可/版本边界。

**未选方案**：在业务代码中直接导入 `ref/Diffusion-Planner` 或用 nuPlan planner wrapper 包装
MetaDrive。

**后果**：每项字段顺序、padding、坐标和单位都要用独立测试证明等价；公开发布前仍需确认上游
源码许可和署名要求。

## D-003 首阶段采用运动学轨迹闭环

**选择**：Diffusion Planner 保持输出 80 个 10 Hz 点；`TrajectoryMetaDriveEnv` 每 0.5 s 接收
完整轨迹，`KinematicTrajectoryPolicy` 直接执行前 5 个点。

**理由**：这与官方 8 s/10 Hz 输出天然对齐，能先隔离验证坐标、地图、观测、采样和滚动规划，
无需把低层控制误差混入模型接口验证。

**未选方案**：一开始就让模型输出 `[steering, throttle_brake]`，或先加入 PID/MPC 跟踪。

**后果**：当前闭环只能证明轨迹级接口和运动学执行，不能证明 steering/throttle 动力学可执行性；
未来低层控制必须作为独立阶段和独立实验口径。

## D-004 显式处理 PGMap 限速哨兵

**选择**：`programmatic_lane_speed_limit_kmh` 是必填实验条件。环境 reset 后只替换精确
`1000 km/h` 哨兵，保留已有真实限速；地图适配器再次拒绝残留哨兵。

**理由**：`1000 km/h` 是 MetaDrive 0.4.3 的“未设置”元数据，不是有效道路限速。环境边界
修正让仿真器和模型共享同一语义，模型边界校验防止配置失效后继续产生分布外输入。

**未选方案**：在 Python 中默认 50 km/h、把所有 lane 统一覆盖为 50 km/h、将 1000 视为
unknown，或只在模型适配器静默转换。

**后果**：配置缺失立即失败。对当前 `SC`，原有 20 km/h 曲线 lane 必须保留；统一限速只能另立
名称做消融实验。

## D-005 分离无交通与通用观测适配器

**选择**：用窄接口 `NoTrafficMetaDriveObservationAdapter` 支持明确为空的场景；未来另建通用
`MetaDriveObservationAdapter` 处理邻车 21 帧历史和静态物体。

**理由**：无对象时全零 padding 符合官方契约，但它不能被外推为有交通数据已正确构造。窄接口
能在当前阶段对越界输入立即失败。

**未选方案**：让一个未完成的通用适配器在发现交通参与者时忽略它们或退化为全零。

**后果**：当前评测入口必须同时校验配置和运行时对象；有交通功能完成前不能复用该入口。

## D-006 不修饰模型失败

**选择**：原始模型轨迹直接进入既定执行器并完整落盘，不做平滑、限幅、中心线投影、seed 挑选或
备用控制器切换。

**理由**：这些处理会把接口错误、域偏移或模型失效伪装为性能改善，使 RL 和能耗结论不可归因。

**未选方案**：为提高闭环成功率加入启发式安全层作为默认路径。

**后果**：失败回合是合法实验结果。若未来研究安全过滤器，必须作为显式方法、独立对照组和独立
指标报告。

## D-007 分离 baseline 与 DPPO sampler

**选择**：保留官方 10 步 DPM-Solver++ 作为推理 baseline；另建 DPPO-compatible stochastic
sampler，显式返回逐步高斯转移和 log-prob。

**理由**：官方 solver 的现有封装位于 `no_grad`，只返回确定性更新后的最终样本。DPPO 把去噪链
视为内层 MDP，需要可重算的随机策略概率，无法从 solver 输出事后补算。

**未选方案**：修改 baseline solver 让其伪装成随机策略，或只对最终轨迹定义一个近似 log-prob。

**后果**：baseline 路径已实施；DPPO sampler、critic、rollout 和 PPO 更新仍未实现。两条路径
必须分别测试，不能让 RL 改造破坏官方基线。

## D-008 长程预瞄采用零初始化残差

**选择**：沿导航路线构造 200–500 m 的曲率、限速和拓扑预瞄序列，以小型
`RoadPreviewEncoder` 编码，并通过零初始化投影残差注入 DiT 条件。

**理由**：现有 `route_lanes` 是约 100 m 的局部几何，且 decoder 不消费 route 限速。直接扩展
固定 route tensor 会破坏 checkpoint 形状；零初始化残差能使新增模块初始时近似不改变基线。

**未选方案**：把远期路线硬塞入 25 条局部 route lane、改变 checkpoint 固定输入维度，或默认
MetaDrive 平面地图含坡度。

**后果**：该设计尚未实现。新增预瞄必须有“零残差等价官方基线”和距离/单位测试。

## D-009 分阶段引入能耗模型

**选择**：训练早期使用 MetaDrive `step_energy` / `episode_energy` 观察同配置下的相对趋势；只有
趋势可复现后，才用 FASTSim 对实际速度 trace 做精细复核，并根据复核结果决定是否进入训练奖励。

**理由**：先打通 RL 信号能减少同时调试 planner、PPO 和车辆能耗模型的耦合；FASTSim 更适合
最终车辆级解释，但需要明确车辆配置和高质量行程输入。

**未选方案**：从第一轮 RL 就强耦合 FASTSim，或把 MetaDrive 代理指标解释为真实燃油消耗。

**后果**：两种指标必须分名、分单位、分累计边界报告。FASTSim 依赖已声明但尚未接入业务逻辑。

## D-010 使用 Lightning Fabric 承载 RL 循环

**选择**：DPPO 主循环计划使用 Lightning Fabric 管理设备、混合精度、DDP、日志和 checkpoint，
同时保留项目自定义 rollout/PPO 循环。

**理由**：标准 Lightning Trainer 更适合批式监督学习，难以自然表达环境采样、去噪内层 MDP 和
PPO 多 epoch 更新；完全手写分布式运行又会增加与研究目标无关的维护成本。

**未选方案**：把 PPO 强行改写为标准 Trainer step，或自行实现设备/分布式抽象。

**后果**：这是未实现方向，不得从 `lightning` 依赖推断训练入口已经可用。

## D-011 以 Linux/Docker/CUDA 作为正式验收

**选择**：Windows 只做编辑、格式化、CPU 快速测试和可用的 simulator 回归；依赖锁定、完整
MetaDrive/FASTSim、CUDA 和正式实验以 Ubuntu 22.04 / CUDA 12.4 Docker 环境为准。

**理由**：训练与部署目标在 Linux/CUDA，上游仿真和数值库可能存在平台差异。单一固定容器比
开发机偶然可运行更可复现。

**未选方案**：把 Windows 本机完整运行视为正式性能验收。

**后果**：任何只在 Windows 得到的性能结果都必须标为本机诊断；在 Docker 复验前不能升级为
正式基线。
