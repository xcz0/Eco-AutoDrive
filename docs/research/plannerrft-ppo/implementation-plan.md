# PlannerRFT PPO-only 分阶段实现与验收

## 使用方式

本文件是候选实施顺序，不是进度表。阶段只有在对应 GitHub Issue 被创建后才成为 active work；
实际运行结果只登记在 `experiments/README.md`。每一阶段都必须保留上一阶段的独立入口和证据，
不得为了让后续训练运行而放宽边界校验。

## 阶段总览

| 阶段 | 可交付物 | 训练对象 | 进入下一阶段的必要证据 |
| --- | --- | --- | --- |
| 0 | 当前 HEAD 的冻结 baseline 复核 | 无 | checkpoint、输入、噪声、DPM 输出和闭环产物可复现 |
| 1 | 可切换 5-step DDIM | 无 | sampler 数学测试、确定性/随机性测试和 paired 闭环对照 |
| 2 | reference planner 与横纵向 guidance | 无 | guidance 为零退化及正负方向/量纲测试 |
| 3 | Exploration Policy 与 value head | policy forward only | Beta 参数、变换、log-prob、entropy、value shape 全部可验证 |
| 4 | closed-loop rollout 数据契约 | 无 | 时间轴、action、reward、done、bootstrap 和 seed 完整对齐 |
| 5 | GAE 与 PPO optimizer | actor/value | 手算批次、旧新策略比率和参数冻结测试通过 |
| 6 | 小规模闭环训练 | actor/value | reward/分布变化可重复，且无 silent failure |
| 7 | 配对消融与规模化 | actor/value | learned guidance 优于预注册对照，结论按终止类型分层 |

## 阶段 0：重建不可变 baseline

### 目的

E-000 证明过固定合成输入上的上游数值对齐，E-004/E-006 提供过部分闭环证据；它们不自动证明
未来实现起点的当前 HEAD 仍完全一致。开始 sampler 修改前，应在新 Issue 对应 commit 上重跑并
登记 baseline。

### 必需记录

- 当前 Git commit、tracked diff、依赖环境、设备和解析后 precision；
- checkpoint/`args.json` 路径、EMA tensor 数和参数总数；
- 地图 seed、噪声 seed、完整 Hydra overrides；
- DPM 的初始噪声、完整联合预测、实际执行 trace、终止类型和延迟；
- 非 GPU/simulator/slow 测试、显式 simulator 测试和格式检查结果。

### 退出条件

当前 HEAD 满足 system contract，且新产物没有覆盖旧实验。此阶段不增加 PPO、guidance 或新
reward。

## 阶段 1：实现可切换的 5-step DDIM

### 代码边界

建议在 `src/eco_planner/models/` 下增加独立 DDIM sampler，并由显式 Hydra `sampler` 配置选择；
不得改写 `BaselineDpmSampler` 的公式或默认语义。sampler 公共 API 至少显式接收：

- 初始扩散状态、denoise callable 和 current-state constraint；
- 有序 timestep/schedule；
- 采样步数；
- `ddim_stochasticity`（避免与 guidance scale 同名）；
- 用于随机 transition 的显式 `torch.Generator`。

输入必须验证 shape、dtype、device 和有限性。输出仍为归一化空间的完整联合状态，逆归一化和
`float32 [B, 11, 80, 4]` 边界由现有 planner runtime 负责。

PlannerRFT 公开算法描述 reference DDIM 从标准高斯开始，而当前 baseline 契约使用
`0.5 * N(0,I)`。实现前必须通过 G-11 明确 DDIM 的初始分布、五个 timestep 和 endpoint；不能
为保持当前行为而声称它等同论文，也不能为追论文文字而改写 baseline DPM。

### 数学与随机性测试

- 用 VP-SDE 的 `alpha(t)`/`sigma(t)` 和 `x_start` prediction 手算单步 transition；
- `ddim_stochasticity=0` 在固定 observation/initial noise 下逐值确定；
- 非零 stochasticity 使用独立显式 seed，同 seed 重放一致、不同 seed 确实改变 transition；
- 五步 schedule 单调、端点明确，current-state constraint 每步保持；
- 非法步数、缺 generator、非有限张量、device/dtype 不一致立即失败；
- DPM 回归测试保持原样通过。

### 闭环对照

在预注册的小型稳定能耗场景网格上配对地图 seed 和初始噪声，比较 DPM-10 与 DDIM-5。此阶段
只判断 sampler 替换是否造成明显驾驶退化，不把论文 nuPlan 分数当作 MetaDrive 通过阈值。
训练/evaluation 的 stochasticity 取值必须来自 resolved config，并分别登记。

## 阶段 2：reference planner 与 orthogonal guidance

### 先关闭的接口问题

必须明确 reference trajectory 的产生 sampler、随机性和刷新频率；横向法向的坐标基准、纵向
速度的离散定义、zero-speed 处理、单位和最大幅度也必须写入配置/ADR。不能用中心线投影、轨迹
平滑或裁剪来制造预期方向。

### 代码边界

- reference planner 从同一经过严格加载的 checkpoint 建立冻结、eval-mode 路径；
- scene encoding 是否共享一次 forward 必须显式决定，不能无意产生两份可训练 encoder；
- guidance 只作用于 DDIM 采样边界，并返回可审计的 reference、scale、目标量和梯度统计；
- planner 输出仍走现有原始轨迹验证和运动学执行边界。

### 必需测试

- 所选公式必须定义唯一 neutral action；若决定 neutral action 为零，则它在相同 sampler
  随机性下必须退化为 unguided DDIM；
- 横向 scale 的正负值产生相反的参考法向趋势；纵向 scale 的正负值产生相反速度趋势；
- rigid transform 后 guidance 结果等价；
- 所有计算使用明确的米、秒、m/s 和 10 Hz 时间轴；
- reference 退化、重复点、零速或非有限值按预先决定的错误边界失败，不静默回退；
- reference/planner/encoder 参数全部冻结，反向图只保留 guidance 所需的 sample gradient。

只有固定 guidance 能稳定通过上述测试，才进入 policy 实现。

## 阶段 3：Exploration Policy

### 候选概率接口

policy 输入候选为冻结 scene tokens `[B, N, H]` 和 ego reference trajectory
`[B, 80, 4]`；是否额外传入 route token/mask 属于 design gate。输出应是：

- lateral Beta 的两个严格正参数；
- longitudinal Beta 的两个严格正参数；
- value `[B]`；
- 可用于审计的 action、joint log-prob 和 entropy。

论文没有公开 Beta `[0,1]` 到 guidance `[-1,1]` 的映射。若复现者选择
`u ~ Beta(alpha, beta)`、`guidance = 2u - 1`，必须明确 PPO 存储的是 `u` 还是变换后 action，
并在 log-prob/entropy 中一致处理变换。不得把该仿射变换描述成作者事实，也不得靠 clamp 把
非法 Beta 参数或边界 action 伪装成有效值。数值稳定参数化及初始零均值方式是复现者选择，需
记录理由。

### 必需测试

- 参数 shape、严格正性、有限性和 device/dtype；
- sample、reparameterized sample（若使用）、mode/mean evaluation 的语义分别明确；
- joint log-prob 等于两个独立维度之和，old/new policy 在相同输入/action 上可重算；
- action support 与 guidance support 严格一致；
- 对称初始化产生零均值 guidance，但不能把分布退化为确定性点；
- value shape 为 `[B]`，不存在意外 broadcast；
- checkpoint 只保存/加载预期可训练参数，并验证冻结模型没有梯度或权重变化。

网络层数、hidden dimension、Beta 初始化常数若官方未公开，必须作为 Hydra 必需字段，不得写成
声称与作者相同的默认值。

## 阶段 4：closed-loop rollout 契约

### MDP 时间步先决条件

当前 `env.step()` 聚合五个 0.1 s 子步，论文描述则是执行所选轨迹第一个动作后重规划。必须先在
[design-gates.md](design-gates.md) 关闭 G-01：选择 0.5 s 规划周期 MDP，或修改为 0.1 s
论文口径。两者的 reward、discount、终止和 rollout length 不能混用。

### 每个 transition 的最低字段

```text
policy observation or frozen encoding
reference trajectory and any policy mask
base Beta action u and transformed guidance action
old joint log-prob, old value
diffusion-noise seed/state and exploration-policy seed/state
reward with component values and units
terminated, truncated, bootstrap mask
scenario/map seed, planning-cycle index, executed substep count
```

不为 PPO-only 保存 DDIM 去噪链，除非存在独立的诊断需求；去噪 transition 不是该 PPO actor 的
action。若每个场景生成多个 guidance 候选再随机选择，buffer 必须能证明被执行候选的抽样概率和
old log-prob 完全一致。

### 必需测试

- 固定小回合逐项对齐 observation/action/reward/next observation；
- terminal 不 bootstrap，time-limit truncation 是否 bootstrap 由显式决定控制；
- rollout 在回合边界不会泄漏 GAE；
- 5 个 substep 中提前终止时，executed step count 与 reward 累计一致；
- policy action RNG 与 diffusion noise/map seed 分离且都可重放；
- buffer 拒绝缺字段、错误 shape、非有限值和混合 device。

## 阶段 5：GAE 与 PPO optimizer

### 奖励先决条件

PlannerRFT 的 reward 组件不能直接等同于 MetaDrive 内置 reward。必须先逐项定义 collision、
drivable-area compliance、wrong-direction、TTC、有效进度、comfort 和 speed 等候选量在本仓库的
数据来源、单位、采样频率、累计边界和 terminal 语义。能耗项若加入，需作为本项目扩展单独命名，
不能冒充论文 parity reward。

### 数学测试

- 对短手算序列验证 discounted return、TD residual 和 GAE；
- 分别覆盖 terminal、truncation、rollout 尾部 bootstrap；
- 优势标准化只在预定 batch 边界发生，零方差情况显式处理；
- PPO ratio 由 guidance policy 的 joint log-prob 得到，不使用 DDIM transition probability；
- clipped policy loss、unclipped/clipped value loss选择、entropy 和总 loss 与手算一致；
- minibatch 索引无遗漏/重复越界，epoch 间 old log-prob 不变；
- 梯度范数、优化器和 scheduler 均来自配置；非有限 loss/gradient 立即失败；
- optimizer step 后只有 actor/value 参数变化。

论文给出的超参数可作为 parity profile；本机 smoke profile 应单独配置，不能用较小规模运行冒充
论文设置。

## 阶段 6：小规模 closed-loop 训练

### 目的

先验证完整数据流和学习信号，不追最终分数。使用 2--8 个环境、16--32 个 MDP steps 等小规模
值时，必须标为 smoke profile，并记录与 parity profile 的全部差异。

### 预注册检查

- 固定场景分布、map/noise/policy seeds、训练步数和评测间隔；
- actor/value loss、KL、clip fraction、entropy、value explained variance、gradient norm；
- lateral/longitudinal Beta 参数、action 均值/方差和边界附近概率质量；
- reward 每一组件、终止类型、有效进度和执行误差；
- 冻结参数 hash 在训练前后相同。

### 退出条件

至少两个独立训练 seed 可重放；策略分布随观测变化；没有通过停驶、频繁失败或非法轨迹获得更高
分数；全部失败回合保留。loss 下降本身不构成通过。

## 阶段 7：消融与规模化

### 必需对照

1. 5-step DDIM，无 guidance；
2. uniform guidance；
3. 固定 Beta guidance；
4. learned Exploration Policy。

评测使用相同场景、地图 seed、diffusion noise seed；随机 guidance 还需预注册自身 seed。主要
指标和停止规则必须在看结果前确定。报告 reward、有效进度、安全、速度/旅行时间、舒适性、代理
能耗指标和终止类型，不用单一累计 reward 掩盖失败。

### 规模化门槛

论文级并行环境数、rollout length、batch、PPO epochs 和学习率只在小规模正确性成立后启用。
当前单设备 evaluation runtime 不承担多环境训练；应建立独立 training orchestrator，并通过 ADR
决定进程、设备、seed 派生、环境生命周期、失败传播和产物聚合。

### 可以支持的最终结论

若 learned guidance 在预注册配对消融中稳定优于对照，只能声称 PlannerRFT 的 PPO exploration
branch 在本项目 MetaDrive 运动学执行条件下得到论文级机制复现。没有 nuPlan 同配置、官方代码和
GRPO 时，不得声称逐行复现、论文数值 parity 或完整 PlannerRFT 部署模型复现。
