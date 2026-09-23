# 实验工具与离线分析契约

涉及研究协议、固定批次、人工干预或离线报告时读取。这里只描述已实现工具；实际运行证据归 docs/experiments，研究假设归 docs/research。这是 [system contract](../system-contract.md) 的分篇。

## 工作流与机制归属

研究层按 `experiments.comparison/reward/credit/guidance/training` 组织。`experiments.protocol` 只定义共享 train/held-out 协议及配置组合，不负责执行或统计。底层模块和 analysis 不导入 experiments。

| 模块 | 输入与职责 |
| --- | --- |
| `rl.rollout.collection`、`fixed_batch` | 已 resolved 配置及 typed TrainingJobConfig；一次固定批次采集、读写、索引和拼接 |
| `reward.calibration`、`rl.reward` | reward 配置及校准参数；Progress/Comfort 校准与 energy-band 阈值数学、组件归一化与 safety-gate 缩放归 `reward`；`rl.reward` 只做 episode/audit 逐 substep 测量提取、调用 reward 纯函数并把 scalar reward / audit 写回 TensorDict（重加权、energy-only、重评分），离线归约与在线 transition 聚合复用同一 `reward.aggregate_substep_rewards` |
| `planning.policy` | policy 架构、affine-Beta 动作与采样、policy-only checkpoint 存取与 `policy_state_hash` |
| `rl.optimization` | PPO batch/GAE/normalization、advantage/critic 消融、actor backward、参数变化与更新后 KL 测量 |
| `evaluation.intervention` | 已准备的 runtime、环境、场景、动作与窗口；reset/step、固定噪声、终止处理和部分原始证据 |
| `evaluation.policy_intervention` | 已加载的 frozen `PolicyGuidanceRuntime`、环境、场景与显式 execution prefix；用 policy mean action 复现同一 matched-group 语义，并采集 same-state 双 policy 反事实 planner response |
| `analysis` | 已保存结果、统计、matched 差值、逐 seed 汇总和报告再生成；不执行训练、backward 或 gate 裁定 |

`just exp ...` 是无语义 alias。`scripts/experiments.py` 只负责参数解析、bootstrap、延迟分派与退出码。当前命令为 compare train/eval/analyze、reward collect/run/analyze、credit run/analyze、guidance authority run/analyze、guidance deferral run/analyze、guidance decomposition run/analyze、guidance execution-bridge run/analyze、guidance horizon run/analyze、guidance sweep run/analyze、training grid/diagnose/eval/critic-attribution run/analyze。旧 study 入口、stage A/B/C、晋升/pruning、独立 reproducibility 和 stability 数据库分析已删除，没有旧 CLI/import 别名或历史产物迁移层。历史实验记录保持原样。

## Comparison matched protocol

`configs/experiments/comparison/default.yaml` 默认三臂 a0（frozen）、a1（R0）、a2（energy λ=1）；`calibrated.yaml` 声明 r0/rstress 两个现有 calibrated reward profiles。arm 标签、reward profile 和有方向的 contrasts 均来自配置。相同执行和比较路径支持这两种设计。

默认训练池为 S/SC map seeds 0–7，held-out 为 S/SC seeds 16–23；协议校验两池不相交。默认 held-out 为 no-traffic、300 步 horizon、DDIM5、runtime seed 760025，训练 seed namespace 和 PPO overrides 显式配置。每次 train 显式指定 seed，replay_id=0；组合后核验 reward、seed、sampler、scenario 池和 num_scenarios 覆盖。initial checkpoint 只作为诊断标签，没有独立 update0 协议字段。

比较 YAML 中所有路径相对该文件解析，显式列出 protocol、可空的 baseline_evaluation_dir，以及 runs 的 arm/training_summary/checkpoint_label/evaluation_dir。有 frozen arm 时必须提供 baseline，无 frozen arm 时 baseline 为 null。训练 seed 和 reward 来自 typed summary；每个训练目录的 resolved config 必须精确覆盖协议训练场景池，同 seed 各臂除 reward 与 tracking 元数据外的 resolved 训练条件必须相同，并核验 initial policy、planner、probe 和随机流。评测 checkpoint hash/label 必须对应训练 summary 中的 initial/final 状态。held-out pool、seed、horizon 与 sampler 对照协议核验。没有按目录名推断或选取最佳 seed。

## 固定批次 reward 与 credit

reward collect 配置显式提供 job 和 overrides，底层采集不认识 scalar protocol、arm 标签或研究目录。采集只执行 initial-policy rollout，保存 resolved config、initial policy、runtime metadata、training TensorDict、episode audit NPZ、sample index 和 summary。保持无 optimizer/scheduler step，核对 batch 与动作、log-prob、value/next-value、reward、boundary 的配对。所有 reward/credit 条件共享同一 transition 顺序与初始策略。

reward run 读取该源 batch，重算原始/校准/energy-band reward 组件及权重对照，不恢复 actor 或执行 backward。credit run 从同一源 batch 独立重算校准；无需任何前序诊断目录。两者均保存实际 reward 配置和校准尺度。命名 credit 配置 sensitivity、objectives、ablation、energy-band 保留既有对照数值轴；不默认运行全部组合。

Progress 使用完整批次正向 delta 中位数除以目标分数。Comfort 仅对存在零分的子项使用 `max(原 limit, P50(abs(metric))/(2-target_score))`，保留原分段线性评分与四项取最小值。启动 transition 不删除，未失活子项与 Energy/TTC/Speed/Safety 不变。energy-band 由本 batch 的配置分位数推导，不核验 E-034/E-038 冻结数值。全局 reward profile 默认值不变。

校准后的 Comfort 仅表示该运动学执行分布下的相对平顺性，不重新定义物理限值。audit 保存有符号原始量、评分绝对值、原 limit 超限率、零分/满分率、含并列的最小值归因和逐 scenario/planning-cycle 数组。Energy-only 是逐 substep `safety_gate × energy component` 的聚合（与在线该目标 endpoint 同义），仅作诊断，不是新训练 profile。

credit 配置显式声明 reward arms、raw/center/z advantage forms 和 standard_gae/reward_only_gae/discounted_return credit forms。standard GAE 保留原 critic；reward-only GAE 同时置零 current/next value（含 tail bootstrap），保留 gamma/lambda/boundary；discounted return 逐 episode 反向递推，无 critic/bootstrap。raw 原样，center 仅减 full-batch 均值，z 复用训练 sample-std normalization。全部复用训练 `build_ppo_batch` 与 `ClipPPOLoss`，仅对 loss_objective backward，不混入 critic/entropy 梯度，不做 clipping、optimizer 或 scheduler step。

梯度按 actor head、shared trunk、lateral/longitudinal head 行分组。统计使用 float64，GAE/backward 保持训练 dtype/device。零初始化 head 阻断 initial trunk actor 梯度；零 norm cosine 和零分母 ratio 为 null，不解释为相同方向。配置中的 objective/attribution gate 只适用于当前实验；数值恒等式（梯度混合线性、center/z 比例、critic 抵消、discounted return）由独立测试覆盖。固定批次证据不证明训练后行为改变。

## Guidance control-authority intervention

authority 的实验层选择 matched groups、干预值和裁定规则。`evaluation.intervention` 接收准备好的 runtime/环境及 InterventionExecution；InterventionExecution 携带每 arm 的显式 2D `(lateral, longitudinal)` 常量动作，按设备上有限 float32 [B,2] 构造，允许闭区间 ±1，直接用于 orthogonal_policy，不经过 Beta 分布或 log-prob。

每组固定场景、reset seed、slot、batch shape 与 noise seed；每 arm 独立 reset，核对初始 observation、simulator state 与逐周期 initial noise。DDIM stochasticity=0。该诊断入口显式传 `execution_steps=1`，即每周期执行 0.1 s；默认首步 immediate、20 步为 2 s 窗口，env horizon 大于窗口。提前终止的 slot 不再 step、不补零，但保持 inference batch 与 noise draw 次数，终止后的 planner audit 不进入执行指标。异常时保存当前 arm 的部分证据并抛出。普通 evaluation（含 guidance sweep 与 policy evaluation）使用 canonical 0.5 s execution，不传该覆盖。

执行指标来自 TransitionMetrics；窗口 fuel proxy intensity 为实际累计 fuel/距离，零距离为 undefined。保存实际运动、终止、reference/guided prediction、guidance diagnostics 与 noise。实验层计算场景内 arm repeat 均值的 Spearman、±1 endpoint 差与重复噪声，阈值和所需一致方向场景数由配置指定。常量相关性为 undefined，零噪声不使零效应通过。安全、完整性、时间窗口方向冲突与 planner-output 到执行链路单列；gate 失败不等于执行失败。

## Guidance execution-horizon intervention

`guidance horizon` 复用 authority 的 matched-group 机制，但把“每次规划后连续执行多少 waypoint 再重规划”作为唯一诊断轴。实验层声明 `total_window_steps`（固定 2 s 窗口）与 `execution_horizons`；每个 horizon k 必须整除总窗口且 `k < PLANNER_HORIZON`，`replan 次数 = total_window_steps / k`。`evaluation.intervention` 每周期一次规划、执行 `execution_steps=k` 个 0.1 s 子步；提前终止的 slot 不再 step，剩余子步不补零，终止后的 planner audit 不进入执行指标。首次规划的完整 8 s 预测前向位移在固定 checkpoint（0.1/0.2/0.5/1/2/4/8 s）记录，并在跨 horizon 之间核对匹配。

统计按 horizon 分组：执行窗口 speed/endpoint speed/distance/progress/energy intensity/tracking error 的逐场景 Spearman、±1 endpoint 差与重复噪声；`planner_response` 为首次规划位移的 matched 差。预声明方向判据与 authority 相同，并据此给出 `gate_a.status`（`strong_evidence_for_receding_horizon_mismatch` / `sign_unchanged_by_horizon` / `mixed_or_inconclusive` / `safety_or_proxy_failure`）。该工作流是 diagnostic causal intervention，不修改 reward、PPO 或 baseline 执行方案；显式 `execution_steps` 覆盖仅是该诊断入口。

## Guidance replanning-deferral trace

`guidance deferral` 复用 authority/horizon 的 matched-group 机制，固定诊断 0.1 s receding-horizon execution（显式 `execution_steps=1`，每周期执行 1 个 0.1 s 子步再重规划），把“连续 replanning cycle 的逐 waypoint guidance effect 是否被反复推迟到执行 prefix 之外”作为唯一诊断轴。实验层声明 `total_window_steps`（固定 2 s 窗口）、`execution_steps`（必须为 1）与 5 个固定 longitudinal 臂；`replan 次数 = total_window_steps`。runner 只加载冻结 planner，不加载 policy；`collect_group` 以 opt-in `include_waypoints=True` 保存每周期的完整 80 waypoint 前向位移（默认关闭，其它 workflow 产物不变）。

统计以 matched 端点差 `Δ = D(g=+1) − D(g=-1)` 表示，先把逐 noise repeat 取中位，再逐 cycle 计算：首点（0.1 s）效应、8 s full-horizon 效应、首个正 response 的 zero-crossing waypoint step，以及跨 cycle 的稳定性。预声明场景判定要求多数 cycle 首点 ≤ 0、full-horizon > 0、正 response 越过执行 prefix 且 zero-crossing step 稳定；场景多数满足即 `gate_c.status = repeated_deferral`，首点多数为正为 `deferral_absent`，否则 `mixed_or_inconclusive`，安全/完整性或 proxy 失败覆盖为 `safety_or_proxy_failure`。该工作流是 diagnostic causal intervention，不修改 reward、PPO 或 baseline 执行方案。

## Guidance lon/lat component decomposition

`guidance decomposition` 复用 authority/horizon 的 matched-group 机制，把每个 training seed 的四个常量 guidance 臂（`r0`、`lon`、`lat`、`joint`，原生 `(lateral, longitudinal)` 顺序）作为唯一诊断轴；臂常量来自 E-041 frozen final-policy Beta mean。runner 组合 held-out evaluation job（`jobs/evaluation/no_traffic_heldout_manual`，E-040 matched 协议 + orthogonal_policy），只加载冻结 planner，不加载 policy。每个 (worker batch, training seed, noise seed) 为一个 group，四臂顺序固定 r0/lon/lat/joint；每周期显式执行 1 个 0.1 s 子步（`execution_steps=1`），episode 可变长。配置 validator 要求 `lon` 只在纵向维、`lat` 只在横向维偏离 `r0`，且 `joint == r0 + (lon-r0) + (lat-r0)`，值域 ±1。

统计按 (seed, scenario, arm) 配对：逐场景计算相对 `r0` 的 effect 与 `interaction = joint - lon - lat`，各 arm 方向由多数场景门槛决定。attribution verdict（`longitudinal-dominated` / `lateral-dominated` / `nonlinear-interaction` / `not-reproduced-state-dependent`）由预声明 `dominance_share` 与 `expected_joint_direction` 判定；overall 在 seed 不一致时为 `mixed-across-seeds`。该工作流是 diagnostic causal intervention，不修改 reward、PPO 或 baseline。

## Frozen-policy execution-contract bridge

`guidance execution-bridge` 复用 authority/horizon 的 matched-group 机制，但把每个 arm 的 guidance action 换成 E-040 冻结 exploration-policy checkpoint 的 deterministic Beta mean，并把“同一 frozen learned policy difference 在不同 execution prefix 下的闭环方向”与“同状态局部 policy→planner 时间响应”作为诊断轴。runner 通过 `--source-dir` 指向 E-040 study 目录，按相对路径加载 `r0/rstress × seeds {0,1}` 的 `policy-final.pt`，只加载冻结 planner 与 policy，不训练、不修改 reward/PPO/planner；`evaluation.policy_intervention` 以 policy mean action 复现固定 batch、固定逐周期 diffusion noise 与 matched reset，并在 runner 内核对 policy hash 不变。

Part A 在每个 execution prefix k（默认 1/2/5，必须整除 evaluated horizon）下闭环重放两臂，episode 可变长，记录 executed speed/energy/route/progress/distance、arrive/collision/out-of-road/stopped/episode length、planner-to-execution tracking error 与逐周期 policy guidance action；first-plan planner response 在跨 prefix 间核对匹配，不一致即抛错。统计按 (seed, scenario) 配对 `Rstress - R0`，逐 prefix 由预声明场景多数门槛给出方向；`gate` 的 Part A 判据要求两个 training seed 同时出现 k=1 negative、k=5 positive 的 crossover，否则记录 amplify 或 no-material。k=5 与 E-040 matched held-out 为方向复现对照，不是逐值复现（collector 与 evaluation engine 不同）。

Part B 在同一 held-out 状态分布上采集 matched context，然后用两个 policy 对**同一 observation、同一 planner、同一 diffusion noise**各评估一次，只替换 policy guidance；保存 0.1/0.2/0.5/1/2/8 s 的 forward/lateral displacement effect、Δg_lat/Δg_lon、zero-crossing。判据要求多数 context Δg_lon > 0 且 0.1 s 中位 ≤ 0、0.2/0.5 s 中位 > 0；若真实局部 policy response 在 0.1 s 已为正，则记为 local-response-differs，不能把 E-043 的全幅 ±1 sweep 直接等同于 learned operating point。最终 `gate.verdict` 在 1/2/3/4 四个预声明结论中选择。该工作流是 diagnostic causal intervention，不修改 baseline execution contract。

## Training

training grid 使用 learning rate × epochs × gradient norm 的显式笛卡尔积、update 预算和诊断阈值。复用现有 PPO 算法；先检查更新量、KL、ratio、probe/Beta 与行为条件，再对通过项进行 matched initial/final held-out 测量。所有通过项按最低 learning rate、再 epochs、再 gradient norm 选择；不按训练 reward 排名。没有通过项时 selected_config=null；训练异常保留 partial grid summary、失败异常和原始训练证据并传播，未完成网格不宣称成功。

training diagnose 的输入配置列出 training_summaries、training_seeds、mc_draws、mc_seed。checkpoint/Torch 测量在执行层完成并保存，明确区分训练 loss 记录的 pre-update approximate KL 与同一批数据上的 post-update KL。参数 delta、policy ratio、probe 变化与显式 MC 随机 seed 均保留。

training eval 显式配置 protocol、records（arm、training_summary、checkpoint_label、checkpoint_path、可空 deterministic_evaluation_dir）及 policy_action_seeds。可复用 deterministic 结果须匹配 checkpoint hash/label、mean action mode、held-out pool、horizon、sampler、runtime seed；与 sample action 评测按完整场景和 noise 条件配对。不存在 positive-control 固定目录布局依赖。逐训练 seed、逐 action seed 保留结果和缺项，不选择最佳 seed；exact replay 的正确性由训练/随机流测试覆盖。

## Training critic attribution

`training critic-attribution` 是无训练、无仿真、无 planner 的离线 backward-only 归因：输入显式列出训练 run（label/arm/training_seed/相对路径/reward profile）、baseline `standard_gae` 与对照 credit forms、`advantage_form`、`update_indices`、provenance 容差和预登记 materiality 阈值（actor_head/lateral/longitudinal cosine 下限与 advantage sign-flip 上限）。每个 update 从 `updates/update-NNN/*.npz` 经 `read_rollout_episode` 从 audit + tail 重建 episode，使用该 update 的 pre-update policy（update 0 为 `policy-initial.pt`，否则 `policy-update-(k-1)`），只用训练 `ClipPPOLoss` 的 `loss_objective` 做 full-batch backward，不 clipping、不做 optimizer step。

standard GAE 重建的 raw advantage mean/std 必须与源 summary 记录的逐 update 值在容差内一致；每个 checkpoint 前后 policy hash 不变且 optimizer steps 为 0。梯度按 actor head / shared trunk / lateral / longitudinal 分组，advantage 与梯度对照复用固定批次 credit 的统计量。all-checkpoint 判据不通过即 `critic_material_candidate`，全部通过为 `critic_not_material_to_actor_direction`；cross-arm r0/rstress 对照为 unmatched 描述量，不作因果。离线分析从 `diagnostics.npz` 重算 advantage 对照并与记录逐值核对，梯度测量不重算。

## 实验离线分析与报告

所有 analyze 从新生成的持久化证据独立写出 analysis.json、report.md 和 SVG/PNG。source/output 不得相同或互相嵌套，源文件不变。run 在保存原始证据后复用同一发布函数，默认在本次输出目录生成报告；--no-figures 禁用图片。help、offline analyze、typed summaries 不加载 Torch、MetaDrive/Panda3D 或执行器；仅启用图片时加载 Matplotlib，先设置 Agg。

| 工作流 | 持久化证据 |
| --- | --- |
| reward | summary.json、diagnostics.npz、sample_index.json、audit.json/audit.npz |
| credit | summary.json、diagnostics.npz、sample_index.json，含实际校准配置和原 gate |
| comparison | YAML 显式列出的 protocol、typed training/evaluation summaries |
| guidance authority | episodes.json、intervention_config.json、scenarios.json、decisions.json |
| guidance decomposition | episodes.json、intervention_config.json、scenarios.json、decisions.json |
| guidance deferral | episodes.json、intervention_config.json、scenarios.json、decisions.json |
| guidance execution-bridge | episodes.json、same_state.json、intervention_config.json、scenarios.json、decisions.json |
| guidance horizon | episodes.json、intervention_config.json、scenarios.json、decisions.json |
| guidance sweep | matrix_summary.json 和各 evaluation summaries |
| training | summary.json 中已持久化的 grid/diagnostics/evaluation 测量及各运行事实 |
| training critic-attribution | diagnostics.npz、sample_index.json、summary.json、diagnostic_config.yaml、runtime_metadata.json |

reward/credit 从数组重算分布与配对，核验 sample 长度、重复身份和 scenario 顺序；不以旧 summary 冒充缺失原始数组。advantage 全批标准差使用 ddof=1，value target 使用配置持久化的 value_target_ddof（ablation=1，其他命名配置=0），其余分布 ddof=0。Pearson/Spearman 使用 SciPy 的 ties 定义；常量或不足样本为 undefined。报告引用已保存的 gate，不重裁定。图提供 cosine、1−cosine、norm ratio 与 undefined 标注，不裁剪原值。

评测比较要求相同 workload、sampler、runtime seed，按 scenario/map/map seed/noise seed、evaluation mode、traffic density 精确配对；重复或缺失键报错。差值为 comparison−reference。失败原因与不可用配对数保留；不用零填补失败指标。碰撞/越界正常终止仍保留有效指标，completed 不代表到达。

Comparison 报告顺序为 completion/availability → safety → paired energy。completed rate 分母为所有回合；arrival、route completion 和 safety rate 分母为本 arm completed 回合，未知不计安全。每个配置 contrast 单独使用双方 completed 的 matched 交集；每 seed effect 为 mean energy delta（MetaDrive fuel proxy，mL），负值表示更低，需结合失败、安全和提前终止解释。

scenario bootstrap 配置显式指定 confidence_level、n_resamples、bootstrap_seed，使用 SciPy percentile 与独立 RNG。无 pair 时 estimate/CI 不可用；单 pair 保留 estimate、CI 不可用；常量双样本允许零宽区间。不重采样 training seed。每 seed 单列点估计、CI 与缺项；initial 不进入 final 方向计数。CI 只表示固定策略及可用场景的不确定性，不是跨 training seed 总体区间。

reward sanity 与 execution backend 仍分别属于 validation 与 benchmarking，使用既有独立入口。此次没有修改 planner、PPO 更新算法、reward 公式或仿真语义，没有重新运行历史研究，也不产生新的节能结论。
