# Diffusion Planner 源码复用说明

> 分析对象：`ref/Diffusion-Planner`  
> 本地版本：`a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`（2026-03-10）  
> 上游仓库：<https://github.com/ZhengYinan-AIR/Diffusion-Planner>  
> 用途：为 Eco-AutoDrive 的 MetaDrive 适配、开源基线复现和 DPPO 改造提供实现参考。

## 1. 一句话概括

该源码将 ego、动态交通参与者、静态物体和局部矢量地图编码为场景 token，再用带路线条件的 DiT 联合生成 ego 与 10 个邻车未来 8 s 轨迹；训练采用连续时间 VP-SDE 的 `x_start` 回归，推理采用 10 步 DPM-Solver++。

它是一个面向 nuPlan 的完整实验实现，而不是与仿真器无关的模型库。模型主体可以复用，nuPlan 数据提取和 Planner 包装应在本项目中重新实现。

## 2. 端到端数据流

```text
nuPlan history/map
  -> DataProcessor.observation_adapter
  -> ego 局部坐标、固定数量裁剪/补零、ObservationNormalizer
  -> Encoder
       动态体历史 MLP-Mixer
       静态体 MLP
       车道折线 MLP-Mixer
       token 位置/类型编码 + Transformer 自注意力融合
  -> Decoder / DiT
       当前点 + 带噪未来轨迹
       路线编码 + 扩散时间作为 adaLN 条件
       joint ego/neighbor self-attention + 场景 cross-attention
  -> DPM-Solver++ 10 步去噪
  -> [ego + 10 neighbors, 80, x/y/cos/sin]
  -> 仅取 ego 轨迹并转回 nuPlan 全局轨迹
```

训练时不运行 DPM-Solver。代码直接在随机连续时间 `t` 对真值未来轨迹加噪，模型预测干净轨迹 `x_start`，使用均方误差训练。

## 3. 默认张量契约

默认值来自 `train_predictor.py`，这些维度同时决定预训练 checkpoint 的参数形状，初次复现时不应修改。

| 名称 | 默认形状 | 内容 |
| --- | --- | --- |
| `ego_current_state` | `[B, 10]` | `x, y, cos(h), sin(h), vx, vy, ax, ay, steering, yaw_rate` |
| `neighbor_agents_past` | `[B, 32, 21, 11]` | 2 s、10 Hz 历史；`x, y, cos(h), sin(h), vx, vy, width, length, type_one_hot(3)` |
| `static_objects` | `[B, 5, 10]` | `x, y, cos(h), sin(h), width, length, type_one_hot(4)` |
| `lanes` | `[B, 70, 20, 12]` | 中心点、切向量、左右边界相对向量、交通灯 one-hot |
| `lanes_speed_limit` | `[B, 70, 1]` | m/s；进入模型前按 `std=20` 归一化 |
| `lanes_has_speed_limit` | `[B, 70, 1]` | 布尔有效标志 |
| `route_lanes` | `[B, 25, 20, 12]` | 位于连续 route roadblock 上的局部车道 |
| `route_lanes_speed_limit` | `[B, 25, 1]` | 当前模型未消费 |
| `route_lanes_has_speed_limit` | `[B, 25, 1]` | 当前模型未消费 |
| 推理输出 `prediction` | `[B, 11, 80, 4]` | ego 加 10 个邻车的 `x, y, cos(h), sin(h)` |

关键约定：

- 所有几何量以当前 ego 后轴中心为原点，ego 朝向为局部 x 正方向。
- 历史长度为 21 帧，包含当前帧；未来为 80 帧，不包含当前帧，间隔 0.1 s。
- 0 张量同时充当 padding。归一化后代码会把 padding 再置 0，因此适配器必须保持完全一致的掩码语义。
- 动态体按当前时刻到 ego 的距离排序，最多保留 32 个，其中行人和自行车优先保留最多 10 个；decoder 只联合预测前 10 个动态体。
- 内部扩散状态包含固定当前点，形状为 `[B, 11, 81, 4]`；loss 和最终输出都会去掉第 0 个当前点。
- `normalization.json` 是 checkpoint 输入分布的一部分，不是可选预处理。

## 4. 核心模块说明

| 源文件 | 职责 | 对本项目的建议 |
| --- | --- | --- |
| `model/diffusion_planner.py` | 组合 encoder/decoder，并定义初始化方式 | 保持模块层级和参数名，便于原 checkpoint 严格加载 |
| `model/module/encoder.py` | 动态体、静态体、车道编码及场景 token 融合 | 可直接移植，输入必须满足上述契约 |
| `model/module/decoder.py` | 路线编码、DiT、训练/推理分支和当前点约束 | 保留 DiT 主体；将采样策略从 decoder 中拆出更适合 DPPO |
| `model/module/dit.py` | 时间嵌入、adaLN-Zero、自注意力、场景 cross-attention | 可直接移植；不要为“标准化写法”改残差结构，否则行为和权重语义会变化 |
| `model/diffusion_utils/sde.py` | 线性 VP-SDE，`beta_min=0.1`、`beta_max=20` | 作为 BC 训练和 DPPO 离散时间表的基准定义 |
| `model/diffusion_utils/sampling.py` | 10 步、二阶、multistep DPM-Solver++ 基线采样 | 原样保留为官方 ODE 基线，不用于 PPO 概率计算 |
| `loss.py` | ego planning loss 与有效邻车 prediction loss | 可用于开源 BC 基线复现，不直接作为 PPO loss |
| `utils/normalizer.py` | 观测与输出状态归一化 | 可直接复用或等价重写，必须做数值一致性测试 |
| `data_process/*.py` | nuPlan ego/agent/map/route 提取 | 只参考算法和字段顺序；MetaDrive 侧重新实现 |
| `planner/planner.py` | nuPlan `AbstractPlanner` 生命周期和轨迹转换 | 不复用；由 MetaDrive 环境适配器和轨迹执行器替代 |

### Encoder

- 共有 `32 + 5 + 70 = 107` 个场景 token，hidden size 为 192。
- 每个动态体的 21 帧历史先经通道投影和 MLP-Mixer 池化为一个 token。
- 每条 lane 的 20 个点以相同方式池化，并叠加速度限制与交通灯 embedding。
- 动态体、静态体和 lane token 拼接后，加入由位置、朝向和类别组成的 7 维位置编码，再经过 3 层自注意力。
- `FusionEncoder` 强制令第一个 token 有效，以避免全 padding attention；无动态体场景下，这个 token 可能仍是全零占位符。

### Decoder / DiT

- 将每个参与者的完整 81 点状态展平为 324 维，再投影为 192 维 token。
- ego 和 neighbor 使用不同的 participant embedding，11 个参与者 token 之间执行 self-attention。
- `route_lanes` 被压缩为一个全局路线条件 token，与扩散时间 embedding 相加后控制 adaLN-Zero。
- 每层再对 encoder 的 107 个场景 token 做 cross-attention。
- 默认直接预测干净状态 `x_start`，而不是噪声 `epsilon`。

路线输入有一个容易忽略的限制：`RouteEncoder` 实际只读取 `route_lanes[..., :4]`，即中心点和切向量；边界、交通灯、路线速度限制字段当前都不参与 decoder 条件。

## 5. 训练和推理行为

### 行为克隆训练

1. 将 ego 和 neighbor 未来 heading 转为 `cos/sin`。
2. 使用 `StateNormalizer` 归一化未来状态。
3. 对未来 80 点采样 `t ~ Uniform(1e-3, 1)`，按线性 VP-SDE 前向加噪；当前点不加噪。
4. DiT 预测整段 81 点轨迹，loss 仅计算未来 80 点。
5. 总 loss 为 `neighbor_prediction_loss + alpha_planning_loss * ego_planning_loss`，默认 `alpha=1`。
6. 使用 AdamW、梯度范数裁剪 5、EMA 0.999；官方入口默认 DDP。

### 基线推理

- 初态由归一化当前状态和 `0.5 * N(0, I)` 的未来噪声拼接而成。
- DPM-Solver++ 参数固定为 10 步、二阶、`logSNR` 跳步、multistep、`denoise_to_zero=True`。
- 每次 solver 更新后都覆盖第 0 个时间点，保证当前状态不被去噪改变。
- 整个采样位于 `torch.no_grad()` 中，只返回最终样本；即使底层 solver 可选返回中间状态，现有封装也没有转移分布、方差或 log-prob。

因此原采样器只能作为开源推理基线。DPPO 必须新增显式随机反向转移，不能通过给 DPM-Solver 输出补算 log-prob 来替代。

## 6. 推荐复用边界

### 直接复用并保持 checkpoint 兼容

- `Diffusion_Planner`、`Encoder`、`DiT`、`MixerBlock` 和权重初始化结构；
- `StateNormalizer` / `ObservationNormalizer` 的数值规则；
- 线性 VP-SDE 参数和 `x_start` 训练目标；
- DPM-Solver++ 作为“官方基线”推理路径；其 solver 更新在给定初始噪声后是确定的，但现有入口每次会重新采样初始噪声。

首次加载必须使用严格校验：记录 missing/unexpected keys，并验证实际加载参数数量。不要沿用 `planner.py` 中“仅保留带 `module.` 前缀键”的隐式假设；非 DDP checkpoint 在该逻辑下可能得到空字典。

### 参考后重写

- ego 后轴局部坐标变换；
- 动态体按距离排序、历史对齐、缺帧填充和类型编码；
- lane 插值、方向统一、左右边界相对向量和交通灯编码；
- route lane 连通裁剪；
- 模型输出局部轨迹到 MetaDrive waypoint 的转换。

这些逻辑当前直接依赖 nuPlan 类型、地图 API 和 Pacifica 车辆参数，不能直接用于 MetaDrive。

### 新增实现

- `MetaDriveObservationAdapter`：生成完全相同的模型输入字典；
- `DPPOTrajectorySampler`：返回轨迹、完整去噪链、每步均值/方差和可重算 log-prob；
- ego-only adapter/LoRA、critic 和 PPO 更新逻辑；
- 长距离 `RoadPreviewEncoder`；现有 `route_lanes` 仅覆盖 100 m 查询半径内的局部路线，不应直接承担 200–500 m 预瞄；
- MetaDrive `TrajectoryExecutor` 与 FASTSim trace/reward。

## 7. 移植时应显式处理的问题

1. **维度耦合**：`output_dim=(future_len+1)*4`，`RouteEncoder` 的输入层为 `route_num*lane_len`，修改 80/25/20 都会破坏 checkpoint 形状。
2. **配置名不一致**：数据处理按 `route_len` 构造路线点数，但 `RouteEncoder` 使用 `lane_len`。默认二者均为 20，所以问题被隐藏；本项目应显式断言二者相等，或修正实现并处理权重迁移。
3. **未消费字段**：route speed limit/availability 被数据管线生成和归一化，但没有送入 RouteEncoder。长程能耗条件需要新增编码器，不能假设原模型已利用路线限速。
4. **坐标基准**：源码使用 ego 后轴中心；MetaDrive 常用车辆中心。若不补偿轴距方向偏移，地图、邻车和输出轨迹会系统性错位。
5. **padding 判定**：多个模块用“特征全为 0”判断无效项。真实对象恰在局部原点时，必须依靠朝向、尺寸或类型等非零字段避免误判。
6. **联合预测**：decoder 的 attention 同时更新 ego 和邻车 token。DPPO 初期只应对 ego 输出计算策略概率，并冻结邻车预测路径或添加 ego-only 残差。
7. **实验性 guidance**：`model/guidance` 依赖 nuPlan Pacifica 尺寸，且代码不是训练主路径；碰撞约束应按 MetaDrive 车辆几何重新实现，不作为首版复用目标。
8. **源码许可**：该本地参考快照根目录未发现独立 `LICENSE` 文件。复制源码进入正式包或发布衍生代码前，应确认上游许可与引用要求。

## 8. 建议落地映射

| Eco-AutoDrive 目标位置 | 来源或处理方式 |
| --- | --- |
| `src/eco_planner/models/diffusion_planner.py` | 保持原模型模块命名的兼容实现或薄包装 |
| `src/eco_planner/models/normalization.py` | 等价实现 `normalizer.py`，加载明确的归一化配置 |
| `src/eco_planner/models/baseline_sampler.py` | 封装原 DPM-Solver++，只用于 baseline/eval |
| `src/eco_planner/models/dppo_sampler.py` | 独立新增随机 sampler，不修改 baseline sampler |
| `src/eco_planner/envs/observation_adapter.py` | 参考 `data_process`，基于 MetaDrive API 重写 |
| `src/eco_planner/envs/trajectory_executor.py` | 替代 nuPlan `planner.py` 的输出转换与执行 |
| `third_party/Diffusion-Planner/` | 若正式 vendor，上游 commit、许可和本地补丁需固定记录 |

## 9. 最小验证清单

- 用固定合成 batch 验证全部输入、encoder 输出 `[B,107,192]` 和最终输出 `[B,11,80,4]`。
- 验证平移/旋转同一场景后，局部模型输入保持一致。
- 验证 MetaDrive 车辆中心到后轴中心的变换及逆变换。
- 使用相同 checkpoint、输入噪声和 DPM 参数，对比参考源码与移植模型输出。
- 严格检查 checkpoint key 覆盖率及参数总数。
- 验证所有 padding 在归一化后仍为 0，真实对象不会被误判为 padding。
- 单测 current-state constraint：每个去噪步的第 0 点必须等于观测当前状态。
- 分别测试 baseline sampler 的确定性复现和 DPPO sampler 的 log-prob 重算一致性。

## 10. 阅读入口

建议按以下顺序阅读源码：

1. `train_predictor.py`：默认配置和训练装配；
2. `data_process/data_processor.py`：模型输入的来源；
3. `model/module/encoder.py`：场景 token；
4. `model/module/decoder.py`：扩散状态、路线条件和推理；
5. `loss.py`：训练目标；
6. `model/diffusion_utils/sampling.py`：官方推理采样；
7. `planner/planner.py`：nuPlan 闭环边界。
