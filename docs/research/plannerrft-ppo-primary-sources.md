# PlannerRFT PPO / Exploration Optimization 一手资料核查

> 状态：研究记录，不是当前系统契约、ADR、active task 或实验结果。
>
> 核查日期：2026-08-11。
>
> 范围：仅核查 PlannerRFT 的 PPO / Exploration Optimization，以及它与官方
> Diffusion-Planner 的接口边界；暂不设计或实现 GRPO。

工程适配、阶段验收与 design gates 见
[PlannerRFT PPO-only 复现研究](plannerrft-ppo/README.md)。

## 结论先行

PlannerRFT 不是“用 PPO 更新 Diffusion Planner 的 DiT”。完整方法有两条优化分支：

- PPO 更新新增的 **Exploration Policy**，其动作是横向、纵向两个 guidance scale；
- GRPO 更新 **Fine-tuned DiT** 的轨迹分布。

训练时，冻结的 Reference DiT 产生参考轨迹，Exploration Policy 基于参考轨迹和场景特征输出
两个 Beta 分布，采样 guidance scale 后进行正交引导去噪。论文规定最终推理移除 Reference DiT、
Exploration Policy 和 guided denoising，只保留 5-step DDIM 的 Fine-tuned DiT。因此若暂不实现
GRPO 并冻结 DiT，PPO-only 阶段只能验证“训练期探索组件是否可学习”，不能声称复现了
PlannerRFT 的最终部署模型或论文闭环性能提升。

来源：[PlannerRFT §4.1、§4.4、§4.5](https://arxiv.org/html/2601.12901)，
[PlannerRFT Appendix B](https://arxiv.org/html/2601.12901)，
[官方项目页 Model Overview](https://opendrivelab.com/PlannerRFT/)。

## 来源与证据等级

本记录只使用以下一手资料：

1. [PlannerRFT arXiv v1 正文与附录](https://arxiv.org/html/2601.12901)，提交于
   2026-01-19；附录包含算法、PPO 损失、超参数、奖励和 nuMax 实现说明。
2. [PlannerRFT 官方项目页](https://opendrivelab.com/PlannerRFT/)，用于交叉核对方法概览、
   5-step DDIM、探索策略和数据集统计。
3. [Diffusion-Planner 官方仓库](https://github.com/ZhengYinan-AIR/Diffusion-Planner)，以及本仓库
   `ref/Diffusion-Planner` 固定的一手源码快照 `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`。

截至核查日期，PlannerRFT 官方项目页只链接论文和结果展示，arXiv 条目也没有作者提供的
代码仓库链接；在 OpenDriveLab 的公开仓库列表中未找到 PlannerRFT 或 nuMax 实现。因此以下
结论属于**论文级复现规范**，不是对作者代码的逐行复刻。若后续出现官方代码，应重新核查并
用代码事实覆盖本文的“未公开”项。

来源：[PlannerRFT 官方项目页](https://opendrivelab.com/PlannerRFT/)，
[arXiv 条目](https://arxiv.org/abs/2601.12901)，
[OpenDriveLab 官方仓库列表](https://github.com/orgs/OpenDriveLab/repositories)。

## 一、论文已经规定的行为

### 1. 模块所有权与可训练参数

完整 PlannerRFT 从 IL 预训练 Diffusion Planner 复制出一个冻结的全局 Reference DiT，并在
原模型旁新增 Exploration Policy。论文附录明确完整训练有两个 learnable module：
Exploration Policy 与 Fine-tuned DiT；前者用 PPO，后者用 GRPO。编码器冻结，只微调轨迹
DiT。固定参考轨迹而不是使用持续变化的 Fine-tuned DiT 输出作为引导中心，是作者给出的稳定
训练措施。

来源：[PlannerRFT §4.1、Appendix A Q2/Q3、Appendix B](https://arxiv.org/html/2601.12901)。

对 PPO-only 阶段可直接采用的参数所有权是：

- 冻结场景编码器和导航编码器；
- 冻结 Reference DiT；
- 若明确排除 GRPO，也冻结当前 DiT；
- 只更新 Exploration Policy 的 Guidance Head 与 Value Head。

其中“当前 DiT 也冻结”是本项目 PPO-only 范围推导，不是完整 PlannerRFT 的训练行为；
必须在实验名称和结果中标为 `PPO-only exploration`，不能简称 `PlannerRFT`。

### 2. Exploration Policy 的输入、动作与输出

论文规定 Exploration Policy 条件为驾驶上下文与参考轨迹：

\[
\boldsymbol{\eta}\sim
\pi_\phi(\cdot\mid \mathbf{s},\mathbf{x}^{\mathrm{ref}}),\qquad
\boldsymbol{\eta}=(\eta_{\mathrm{lat}},\eta_{\mathrm{lon}}).
\]

参考轨迹通过 MLP-Mixer 编为紧凑 token，再与 scene embedding 通过 cross-attention 融合。
Guidance Head 从融合表示预测横向和纵向两个 Beta 分布的参数；并行 Value Head 估计
\(V(s_t)\)。附录算法进一步列出 Exploration Policy 的输入为参考轨迹、scene feature 和
navigation feature，输出四个 Beta 参数
\((a_{\mathrm{lat}},b_{\mathrm{lat}},a_{\mathrm{lon}},b_{\mathrm{lon}})\)。

来源：[PlannerRFT §4.2 Eq. 4 与 Exploration Policy 设计](https://arxiv.org/html/2601.12901)，
[Appendix A Algorithm 1](https://arxiv.org/html/2601.12901)。

### 3. Reference trajectory 与 5-step DDIM

每个仿真步先用冻结的 Reference DiT 从标准高斯噪声开始执行 5 个 DDIM 步，得到
\(x^{\mathrm{ref}}\)。训练和论文最终推理都使用 5-step DDIM。DDIM 随机性参数在 RL 训练时
设为 \(\eta_{\mathrm{DDIM}}=1\)，评测时设为 \(0\)，即训练随机、评测确定性。这里的
\(\eta_{\mathrm{DDIM}}\) 与策略动作
\((\eta_{\mathrm{lat}},\eta_{\mathrm{lon}})\) 不是同一个量，配置和代码命名必须分开。

来源：[PlannerRFT §4.5](https://arxiv.org/html/2601.12901)，
[Appendix A Algorithm 1 与 Appendix B Eq. A1--A3](https://arxiv.org/html/2601.12901)。

### 4. 正交 guidance 的定义

论文以参考轨迹切向/法向坐标定义两项能量。横向能量要求计划轨迹相对参考轨迹的法向位移
靠近 \(\lambda_{\mathrm{lat}}\eta_{\mathrm{lat}}\)：

\[
\Psi_{\mathrm{lat}}=\frac{1}{T}\sum_{\tau=1}^{T}
\left(\mathbf n_\tau^\perp(\mathbf x_\tau-\mathbf x_\tau^{\mathrm{ref}})
-\lambda_{\mathrm{lat}}\eta_{\mathrm{lat}}\right)^2.
\]

纵向能量调制计划速度相对参考速度的切向偏差：

\[
\Psi_{\mathrm{lon}}=\frac{1}{T}\sum_{\tau=1}^{T}
\left(\mathbf n_\tau^\parallel
(\mathbf v_\tau-\lambda_{\mathrm{lon}}\eta_{\mathrm{lon}}
\mathbf v_\tau^{\mathrm{ref}})\right)^2.
\]

guided denoising 使用两项能量之和对轨迹的负梯度。论文明确不在 guidance 中加入地图约束或
车辆碰撞约束，而让不可行样本成为 RL 的负反馈。最大偏移采用
\(\lambda_{\mathrm{lat}}=2.5\,\mathrm m\)、
\(\lambda_{\mathrm{lon}}=25\%\)。

来源：[PlannerRFT §4.2 Eq. 2、3、5](https://arxiv.org/html/2601.12901)，
[Appendix B Table A1](https://arxiv.org/html/2601.12901)。

注意：本文这里只忠实记录论文公式。纵向公式的百分比如何转换为代码中的无量纲比例，以及
速度从轨迹点如何离散求取，官方材料没有给出实现细节，不能自行改写后仍称“论文规定”。

### 5. 闭环 rollout 的一步语义

论文规定每个仿真步按不同 guidance scale 生成 \(K\) 条候选轨迹，从候选集中随机选择一条
及其对应的 scale，只执行所选轨迹的第一个 action，然后得到下一状态与即时奖励。回放数据
至少包含当前状态、选中的两个 scale、奖励和当前 value。未来奖励通过 GAE 回传，PPO 学习
长时域的探索方向。

来源：[PlannerRFT §4.3--§4.4](https://arxiv.org/html/2601.12901)。

论文 Algorithm 1 在完整 PPO+GRPO 训练中以 GRPO group size 作为候选数并行生成样本；
公开材料没有为脱离 GRPO 的 PPO-only 阶段单独规定 \(K\)。

### 6. PPO 目标与已公开超参数

附录给出 clipped PPO objective，包含 value MSE 与 entropy bonus：

\[
r_t(\phi)=\frac{\pi_\phi(\eta_t\mid o_t)}
{\pi_{\phi_{\mathrm{old}}}(\eta_t\mid o_t)},
\]

\[
\mathcal L_{\mathrm{clip}}=
\min\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right).
\]

论文以“最大化形式”写总目标：clipped objective 减 value error、加 entropy。实现若采用常见的
loss 最小化接口，必须显式整体取负，不能只对部分项改符号。

PPO 超参数如下。

| 项目 | 论文值 |
|---|---:|
| 总 samples / environment steps | 40M |
| 初始学习率 | \(2.5\times10^{-4}\) |
| 学习率计划 | cosine decay |
| 并行环境数 | 128 |
| 每迭代每环境步数 | 32 |
| batch size | 4096 |
| mini-batch size | 4096 |
| steps per epoch | 1 |
| epochs | 4 |
| value coefficient \(c_v\) | 0.5 |
| entropy coefficient \(c_e\) | 0.01 |
| discount factor | 0.99 |
| GAE \(\lambda\) | 0.95 |
| clip range \(\epsilon\) | 0.2 |
| max gradient norm | 0.5 |

来源：[PlannerRFT Appendix B Eq. A4--A6 与 Table A1](https://arxiv.org/html/2601.12901)。

### 7. PPO 即时奖励

nuMax 的闭环奖励在附录中明确定义为 \([0,1]\) 的加权聚合：

\[
r_t=(\texttt{Col}\cdot\texttt{DAC}\cdot\texttt{WD})
\frac{5\texttt{TTC}+5\texttt{EP}+2\texttt{C}+4\texttt{Speed}}{16}.
\]

其中 collision、drivable-area compliance、wrong direction 是乘性 gate；TTC、ego progress、
comfort 和 speeding 是 soft score。发生碰撞或驶出可行驶区域时立即终止，奖励置零。comfort
涉及纵/横向加速度、jerk 与 steering rate；ego progress 是累计 route progress 相对 expert 的
归一化比值。

来源：[PlannerRFT Appendix C Eq. A10](https://arxiv.org/html/2601.12901)。

这是 nuPlan/nuMax 奖励的论文定义。迁移到 MetaDrive 时，各分量的时间窗、阈值、坐标、
route progress 基准和 expert 基准必须另行定义并验证；不能只保留同名字段就声称奖励等价。
GRPO 的 survival reward 属于另一分支，不应误用于 PPO 即时奖励。

### 8. 训练场景、控制器与评测边界

论文的完整实验使用 8 张 NVIDIA H100、40M environment steps。RFT 数据为 144,494 个
nuPlan 非重叠场景，10 Hz 采样；每个缓存样本含 20 帧历史、1 帧当前和 150 帧未来。作者以
预训练模型得分构造 `Fail`、`Lt90` 与 `All`，完整 RFT 在 `Lt90` 上最好。

nuMax 是 JAX/Waymax 派生的 GPU 并行仿真器，使用 nuPlan/PDM-Closed 的 LQR tracker 与运动学
自行车模型跟踪轨迹，并实现 nuPlan scorer。训练时周围车辆实际为 log replay；论文明确把
IDM 训练交通列为尚未解决的性能问题。最终公开评测则在 nuPlan simulator 的 non-reactive 和
reactive 两种交通设置上进行。

来源：[PlannerRFT §5.1](https://arxiv.org/html/2601.12901)，
[Appendix C](https://arxiv.org/html/2601.12901)，
[官方项目页 Fine-tuning Data Distribution](https://opendrivelab.com/PlannerRFT/)。

因此论文结果不能直接作为 MetaDrive 或本项目运动学执行条件下的目标数值；控制器、交通反应
模型、奖励实现和场景分布都构成实验条件。

## 二、官方 Diffusion-Planner 提供的可复用接口事实

PlannerRFT 论文以 Diffusion Planner 为 IL 预训练基座，但官方基座代码本身没有 PlannerRFT
模块。固定快照 `a3a621f0...` 中：

- decoder 的原始输出联合包含 ego 与预测邻车，每个时间点为
  \((x,y,\cos\theta,\sin\theta)\)；训练输入和输出都在 state normalizer 边界内；
- 推理从“当前状态 + 未来高斯噪声”构造 \(x_T\)，对当前状态施加硬约束，再在输出后 inverse
  normalize；
- 原始采样器是 10-step、二阶 multistep DPM-Solver++，不是 PlannerRFT 的 5-step DDIM；
- 原始 decoder 已有 classifier-guidance 回调，但其默认 guidance scale 固定为 0.5，官方示例
  guidance 是碰撞能量，不是 PlannerRFT 的参考轨迹正交 guidance；
- scene encoder 输出 agents、static objects 与 lanes 融合后的 token；route encoder 单独处理
  route lane 几何。这些张量是构建 Exploration Policy 输入的候选接口，但论文没有把其
  Exploration Policy 精确绑定到官方代码中的具体 tensor key。

来源：[官方 decoder.py（固定提交）](https://github.com/ZhengYinan-AIR/Diffusion-Planner/blob/a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df/diffusion_planner/model/module/decoder.py)，
[官方 sampling.py（固定提交）](https://github.com/ZhengYinan-AIR/Diffusion-Planner/blob/a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df/diffusion_planner/model/diffusion_utils/sampling.py)，
[官方 guidance wrapper（固定提交）](https://github.com/ZhengYinan-AIR/Diffusion-Planner/blob/a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df/diffusion_planner/model/guidance/guidance_wrapper.py)，
[官方 encoder.py（固定提交）](https://github.com/ZhengYinan-AIR/Diffusion-Planner/blob/a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df/diffusion_planner/model/module/encoder.py)。

这意味着复现不能只在现有 DPM-Solver guidance hook 上换一个 energy function；必须先实现并
数值验证 5-step DDIM、训练/评测随机性开关、reference trajectory、轨迹坐标/归一化边界和
scale log-prob 数据流。

## 三、官方材料未公开或存在歧义的内容

以下内容不能从论文、补充材料、项目页或官方 Diffusion-Planner 源码唯一恢复：

1. **Beta 到有符号 scale 的映射。**正文规定
   \(\eta_{\mathrm{lat}},\eta_{\mathrm{lon}}\in[-1,1]\)，但标准 Beta 分布支持集为
   \([0,1]\)，Algorithm 1 又直接写 \(\eta\sim\mathrm{Beta}(a,b)\)。官方材料没有说明是否用
   \(2u-1\)、其他仿射变换或不同分布参数化。这会同时影响 action、log-prob、entropy 和
   zero-mean 初始化，必须在实现前作出显式决定。
2. **纵向能量的零 scale 语义。**Eq. 3 按公开公式写为
   \(\mathbf v-\lambda_{\mathrm{lon}}\eta_{\mathrm{lon}}\mathbf v^{\mathrm{ref}}\)；当
   \(\eta_{\mathrm{lon}}=0\) 时，式中没有显式保留 \(\mathbf v^{\mathrm{ref}}\)。这与正文所称
   “相对参考速度的偏差”和 zero-mean reference-centered exploration 不能唯一对应。官方材料
   没有说明 \(\mathbf v\) 是否已是相对速度，也没有给出其他修正式。
3. **Beta 参数约束与初始化。**未公开如何从网络输出得到正的 \(a,b\)，是否加下界、初始
   concentration 多大，以及“zero parameters 产生 zero-mean guidance”对应的精确变换。
4. **Exploration Policy 网络尺寸。**MLP-Mixer 层数/宽度、reference token 数、cross-attention
   层数与头数、Guidance/Value Head 的层数、激活和参数共享方式均未给出。
5. **Value Head 参数所有权不一致。**正文记为 \(V_\psi\)，PPO 附录写成
   \(V_\phi\)。是否与 actor 共享 trunk、是否同一 optimizer、value loss 是否 clipping 均未说明。
6. **GAE 边界语义。**公开了 \(\gamma=0.99\)、GAE \(\lambda=0.95\)，但没有说明 time-limit
   truncation 的 bootstrap、终止状态 bootstrap、advantage normalization 或 value target 的
   具体计算。
7. **PPO optimizer 细节。**未公开 optimizer 类型、epsilon、weight decay、梯度累积、mixed
   precision，以及 cosine schedule 的 warmup 和终点学习率。
8. **候选数与 PPO-only 关系。**完整算法用 \(G_{\mathrm{grpo}}=8\) 生成组样本，但正文只称
   \(K\) 个候选；没有独立规定 PPO-only 应取多少候选，或能否每环境步只采一个 scale。
9. **噪声配对。**算法没有明确 reference、不同 guidance candidate 之间是否共享初始扩散噪声，
   也没有规定随机候选选择与噪声 seed 的配对策略。
10. **guidance 梯度注入细节。**未给出每个 DDIM step 的梯度系数、对 normalized 还是 physical
   trajectory 求导、梯度是否裁剪/归一化，以及 guidance 对 ego-only 还是 joint agents 生效。
11. **切向、法向与速度离散。**未公开从参考轨迹计算 tangent/normal 的端点规则、退化点错误
    处理、速度差分频率、坐标系和 heading 不连续处理。
12. **“第一个 action”的执行契约与奖励下标。**论文没有将 action 精确展开为轨迹点、控制
    目标或 tracker 时间段，也没有在算法中给出 planning cycle 与 simulator step 的对应频率。
    同一段文字先称转移得到即时奖励 \(r_t\)，随后 buffer tuple 又写 \(r_{t+1}\)，公开材料没有
    解决这个下标不一致。
13. **奖励分量实现常数。**聚合公式和权重已公开，但 TTC、comfort、speed、wrong-direction 等
    指标的完整阈值/窗口依赖 nuPlan scorer；迁移环境时没有官方等价定义。
14. **复现实验元数据。**未公开训练 seed、精确场景 ID 列表、预训练 checkpoint 哈希、软件
    版本锁定、DDIM timestep 子序列和 40M steps 对应的完整运行配置。

来源依据：[PlannerRFT 正文、Algorithm 1、Table A1、Eq. A10](https://arxiv.org/html/2601.12901)。
这些条目是对公开材料可识别信息的缺口审计，不是对作者私有实现的推测。

## 四、复现者必须显式作出的决定

在创建实现 issue 前，至少需要将以下决定写成可测试的配置与接口契约：

1. PPO-only 的成功标准：只验证 guidance action 分布、PPO 更新和闭环 reward 改善，还是保留
   Exploration Policy 作为非论文部署方式参与评测。两者必须分开命名。
2. 有符号 Beta action 的变换及其 Jacobian-corrected log-prob；训练 buffer 存储原始
   \(u\in(0,1)\) 还是物理 scale \(\eta\in(-1,1)\)。
3. Exploration Policy 的具体网络规模、actor/value 是否共享 trunk，以及所有 tensor 的
   shape、dtype、device 和 mask 语义。
4. reference 与 candidate 是否共用初始噪声，candidate 数 \(K\)，以及候选随机选择的 seed
   记录方式。
5. physical/normalized trajectory 的唯一 guidance 求导边界；米、秒、速度百分比和轨迹频率
   必须在入口验证。
6. tangent/normal、速度差分和端点的确定性算法；零长度段、非有限值或坐标不一致时立即失败。
7. PPO rollout 的 planning cycle、执行长度、终止/截断与 bootstrap 规则。
8. MetaDrive 奖励各分量的严格定义，以及它与论文 nuPlan reward **不等价**的实验标注。
9. 5-step DDIM 的 timestep、\(\alpha_s\)、训练 \(\eta_{\mathrm{DDIM}}=1\) 与评测
   \(\eta_{\mathrm{DDIM}}=0\) 的数值对齐测试。
10. 冻结参数清单与断言：每次更新后验证只有允许的 Exploration Policy 参数发生变化。

上述决策一旦进入实现，不应继续留在研究文档中充当系统事实；应按仓库规则进入 ADR、系统
契约或 GitHub Issues。实际数值证据只登记到 `docs/experiments/`。

## 五、建议的论文级最小验证顺序

以下顺序只是把已核查事实转为可验证门槛，不代表模块已经可用：

1. **采样器门槛**：同一 checkpoint 下验证 5-step DDIM 的 shape、有限性、确定性/随机性开关，
   并单独报告它与官方 10-step DPM-Solver 的基线差异。
2. **reference 门槛**：冻结 reference，固定 observation/noise seed 时逐次输出完全一致；所有
   参数保持无梯度、无更新。
3. **guidance 几何门槛**：在解析轨迹上分别验证 lateral-only、longitudinal-only、符号方向、
   单位和正交性，不使用地图投影或碰撞回退掩盖错误。
4. **policy 分布门槛**：验证 Beta 参数正性、action 支持区间、仿射变换、log-prob/entropy 和
   zero-mean 初始化；对手算样本做数值对齐。
5. **rollout 契约门槛**：逐字段验证
   \((s_t,\eta_{\mathrm{lat}},\eta_{\mathrm{lon}},r_t,V(s_t),done/truncated)\)，并证明执行轨迹与
   记账 action 是同一候选。
6. **PPO 数值门槛**：在小型确定性 batch 上核对 GAE、ratio、clipping、value、entropy 和
   gradient norm；冻结模块梯度必须为零。
7. **闭环学习门槛**：先在可诊断的短回合上证明 reward/行为变化来自 policy action 分布，
   再扩展并行环境和场景；结果只能标为 PPO-only exploration。
8. **论文消融门槛**：至少比较无 guidance、uniform、fixed Beta、learned Beta，并同时报告
   reward 均值、方差、轨迹多样性和终止类型；论文官方项目页使用的也是这四组对照。

来源：[PlannerRFT §5.3](https://arxiv.org/html/2601.12901)，
[官方项目页 Comparison of Exploration Policies](https://opendrivelab.com/PlannerRFT/)。

## 不应作出的表述

- “PPO 微调了 Diffusion Planner/DiT”——论文中更新 DiT 的是 GRPO。
- “完成 PPO 就得到可部署的 PlannerRFT”——论文部署会移除 PPO 学到的 Exploration Policy。
- “论文规定 Beta action 直接位于 \([-1,1]\)”——公开公式与标准 Beta 支持集之间缺少映射说明。
- “MetaDrive 同名 reward 等价于 nuMax/nuPlan reward”——控制器、指标实现和场景分布不同。
- “40M steps 或论文分数是本项目验收阈值”——论文使用 8 张 H100、nuMax、nuPlan 数据与特定
  scorer，这些条件尚未在本项目复现。
