# 实验工具与离线分析契约

涉及固定批次诊断、reward 校准、guidance intervention、matched protocol 或离线报告时读取；这里只描述已实现工具，实际运行证据归 docs/experiments，待验证假设归 docs/research。这是 [system contract](../system-contract.md) 的分篇，遵循入口中的使用与维护规则。

## Scalar reward matched protocol

scalar reward 因果研究的 matched protocol 由 `configs/experiments/reward/scalar.yaml` 的 typed manifest 与 `eco_planner.experiments.reward.scalar` runner 承载：三臂 A0（frozen Diffusion Planner，guidance=none）、A1（PPO + PlannerRFT R0 `plannerrft_no_energy_v1`）、A2（PPO + PlannerRFT Rλ=1 `plannerrft_energy_v1`）共用同一 held-out evaluation 作业语义——S/SC map seeds 16–23、no-traffic、300 步 horizon、DDIM5、runtime seed 760025、`env.num_scenarios=24`。训练场景池为 S/SC seeds 0–7（job `jobs/training/ppo_conservative`）、training seed namespace `{0,1,2}`（每次运行经 `--training-seed` 显式选择其一）、`replay_id=0`；manifest 校验训练池与 held-out 池的 (map, seed) 集合不相交，runner 在组合后校验 reward profile、seed（属于 namespace 且未被 override 改写）、sampler、scenario 集合（训练为协议池子集、评测为全集）与 `env.num_scenarios` 覆盖。update-0（initial checkpoint）evaluation 是诊断 artifact，不构成第四个主实验组。

## 固定批次 λ 可辨识性诊断

`experiments.reward.fixed_batch` 拥有共享批次读写、奖励变换/校准、策略恢复和 actor backward。`FixedBatch` 承载 episodes、sample index、解析后的训练配置、resolved config 与采集元数据，scenario 顺序从 sample index 派生。calibration、lambda-identifiability 合为各自单文件，objective-decomposition、critic-gae-ablation 保留 runner 与包含配置的 diagnostics；诊断计算不读写文件或创建模拟器。scalar reward 与 PPO stability 的配置组合独立于执行入口，stability 的评测比较模型独立于训练/评测调度。训练和诊断共用 `rl.optimization` 的 `build_ppo_batch`、`normalize_full_batch_advantage` 与 `PPO_BATCH_KEYS`；随机流派生由 `rl.rollout.seeds.derive_rollout_seeds` 拥有。

`just experiment reward fixed-batch collect --output-dir <batch-directory>` 按显式配置只采集一次 initial-policy rollout。采集配置拥有 protocol、training seed 和 overrides，采集目录保存 resolved/collection config、初始策略、runtime metadata、training TensorDict、episode audit NPZ、sample index 和采集 summary，不保存诊断 arms 或 diagnostics。`just experiment reward lambda-identifiability run --source-dir <batch-directory> --output-dir <diagnostic-directory>` 恢复该批次和初始策略；配置只拥有 lambdas 与 quantiles。所有 λ 共用 transitions、policy context、old log-prob、critic value/next value 与 episode boundary。离线从已保存的 component scores 和 safety gate 重组各 profile 的 reward，复用训练的 episode GAE、full-batch sample-std normalization 和 `ClipPPOLoss`，只反传 `loss_objective`；不混入 critic/entropy 梯度，不做 clipping、optimizer 或 scheduler step。统计用 float64，GAE/backward 保持训练 dtype/device 语义。

诊断目录保存样本索引、全部 λ 对的 advantage/gradient 诊断与逐 scenario/transition 差异，summary 显式记录 source batch 与 initial policy hash。梯度按 actor head、共享 trunk 及 lateral/longitudinal head 行分组；零 norm 的 cosine 和零分母 norm ratio 为 `null`，不得解释为方向相同。当前零初始化 actor head 会阻断 update-0 的 trunk actor 梯度。该入口仅提供连续诊断，不内置可辨识性阈值，不代表 learned behavioral effect。

`just experiment reward calibration run --source-dir <batch-directory> --reference-dir <lambda-diagnostic-directory> --output-dir <new-directory>`
是 Task B 的离线入口。它按 sample index 恢复原 episode audit、training TensorDict 和 initial policy，核对动作、log-prob、value、reward 和 episode boundary 的配对关系；参考诊断必须与采集目录、初始策略 hash、完整样本索引和 λ 轴匹配。不实例化 planner 或 simulator。原配置重放须与参考 λ 诊断数组在 `rtol=1e-5, atol=1e-6` 内一致。校准组只从原始量重算 Progress/Comfort，复用同一 Task A GAE/backward 路径。

研究专用 `configs/experiments/reward/calibration.yaml` 显式指定目标分数与诊断轴。Progress 使用完整批次正向 delta 的中位数除以目标分数；Comfort 仅对有零分样本的子项使用 `max(原 limit, P50(abs(metric))/(2-target_score))`，保留现有分段线性评分及四项取最小值。全部 transition（含启动阶段）保留，未失活子项与 Energy/TTC/Speed/Safety 保持原配置。实际尺度分别保存于两组 resolved config，六个 λ 共用相同校准。

校准后的 Comfort 只表示该运动学执行分布中的相对平顺性，不重新定义物理舒适标准。独立 audit 保存原始有符号量、评分用绝对值统计、原 limit 超限率、子项零分/满分率、包含并列项的最小值归因，以及逐 scenario、逐 planning-cycle 和逐 transition 数据。全局 reward 默认值不受该入口影响。结果只提供连续证据，不自动启动后续实验。

`just experiment reward objective-decomposition run --source-dir <batch-directory> --output-dir <new-directory>` 是 Issue #94 Task C 的离线入口。源 batch 必须是独立采集入口的 update-0 产物；入口按 sample index 恢复 episode 与 initial policy 并核对配对，不实例化 planner 或 simulator。校准由 Task B 冻结规则在本 batch 上重新执行，配置中的 E-034 冻结值以显式容差作为源batch 溯源守卫（防误用 batch/协议漂移，不要求逐位复现）。arms 为校准 R0、正 λ（分母16+λ）与 Energy-only endpoint（`safety_gate × reward_component_energy`，仅诊断用，不是训练 profile，不进入全局 reward 配置）。配置可携带可选 `energy_band` 节（Issue #94 Task E）：runner 从本 batch 执行强度分位数推导 efficiency-band 阈值、对照冻结 expected 值守卫后，把校准 profile 的 energy 切到`calibrated_band` 模式并重打分 episodes 的 `reward_component_energy`；无该节时协议与 E-035 逐字节兼容。每个 arm 在 raw、center-only、z 三种 advantage形式下各做一次 full-batch `loss_objective` backward；z 形式复用训练路径的 `normalize_full_batch_advantage`，center-only 仅减 full-batch 均值。梯度按 actor head/shared trunk/lateral/longitudinal 分组，Gate C 使用 actor-head cosine 与 normalized-advantage RMSE/sign-flip，阈值为 Issue #94 的工程 gate，不声明为一般理论阈值；归因标签（`objective_batch_collinearity` / `normalization_suppressed_identifiability` / `relative_scale_lambda_parameterization_too_weak`）由三种形式的 endpoint 可分性决定。mathematical 恒等式（Pearson 跨形式不变、center 与 z cosine 相等、λ 插值下 gradient 线性组合）由测试逐元素核验。结果只提供该 batch 的裁定，不代表 learned behavior。

`just experiment reward critic-gae-ablation run --source-dir <batch-directory> --reference-dir <decomposition-directory> --output-dir <new-directory>` 是 Issue #94 Task C4 的离线归因入口。源 batch 恢复、校准与溯源守卫与 Task C 相同；`--reference-dir` 指向同批次的新 decomposition 产物，并核对 source batch、初始策略和样本顺序。reference endpoint 由 arm 标签及记录的 index 定位，不假设 stress λ 数量。其 standard-GAE arm（r0 与 Energy-only 的 value/advantage 数组及全部梯度向量）必须与本次在容差内一致（默认 rtol=1e-5 / atol=1e-6），作为同 batch 同链路的 provenance 守卫。arms 为校准 R0 与 Energy-only endpoint；三种 temporal-credit 形式共享同一 reward/动作/log-prob/episode boundary，只改变 credit 计算：standard GAE（原 critic V）、reward-only GAE（V(s)=V(s')=0，置零 next value 同时消融 tail bootstrap，保持 gamma/gae-lambda/boundary）、discounted reward-to-go（逐 episode 反向递推，无 critic 无 bootstrap，仅诊断）。每 arm × credit form 在 raw/center/z 三种 advantage 形式下各做一次 backward-only `loss_objective`；C4 归因（`critic_gae_common_term_dominated` / `temporal_credit_structure_sensitivity` / `reward_batch_collinearity`）按 z 形式 endpoint 可分性判定，复用 Gate C endpoint 阈值。该入口不修改 PPO 训练定义、不重新 rollout、不执行 optimizer step。共享 critic 下 between-arm advantage 差分与 V 无关的抵消恒等式、V=0 GAE 等于 (γλ)-discounted return 与 reward-to-go 递推均由测试逐元素核验。

## Guidance control-authority intervention

`just experiment guidance control-authority run --output-dir <new-directory>` 是 Issue #94 Task D 的人工干预入口；配置位于 `configs/experiments/guidance/control-authority.yaml`。复用 scalar-reward protocol 的训练场景池和模型/仿真配置，但不构造 Exploration Policy、critic 或 optimizer。Fabric inference runtime 的可选 `guidance_action` 接收设备上的有限 `float32 [B,2]`，范围为闭区间 `[-1,1]`，只用于 `orthogonal_policy`；因此 ±1 是合法的直接 planner intervention，不经过 Beta 分布或 log-prob。

每个 matched group 固定场景、reset seed、物理 slot、batch shape 和 noise seed，依次运行五个 longitudinal arms，lateral 固定为 0。每 arm 独立 reset，并逐元素核对初始 observation、simulator state 与逐周期 initial noise。DDIM stochasticity 固定为 0；reference 与 guided pass 沿用 planner 的共享随机流契约。使用 `ExecutionMode.ROLLOUT`，每周期重规划后仅执行 0.1 s；首步为 immediate，固定 20 步为 2 s short horizon，env horizon 必须大于窗口。提前结束的 slot 不再 step、不补零；仍保留固定 inference batch 和 noise draw 次数，其终止后的 planner audit 不计入执行指标。运行错误保留当前 arm 的部分 episode 证据并抛出。

执行统计使用 `TransitionMetrics`；Energy 使用实际执行 fuel proxy 流和现有 `energy_score`，窗口 intensity 为累计 fuel 除以累计距离，零距离记未定义。保存每步实际运动、终止事实、reference/guided prediction、guidance diagnostics 与 initial noise。两个主指标为窗口平均执行速度与窗口 fuel proxy intensity，分别计算每场景的五个 arm repeat 均值的 Spearman、配对 ±1 endpoint 差，以及五个 arm 内 repeat 样本方差均值的平方根。常量相关性记未定义；零噪声不使零效应通过。阈值和一致方向所需场景数由实验配置显式给定；全部 episode 的安全/完整性、时间窗口方向冲突、planner-output 到执行的方向链路及 proxy 公式一致性单独记录。Gate 失败不等于实验执行失败，也不授权后续 PPO tuning。

## 实验离线分析与报告

研究实现按 `experiments.reward/guidance/training` 组织；CLI 使用 `just experiment <domain> <study> <action>`，配置位于相同三域的 `configs/experiments/<domain>/<study>.yaml`；含 evaluation 子配置的 energy-sweep 保留目录。`scripts/experiments.py` 是单文件 CLI，静态命令表只负责参数与延迟分派，分析及核心模块不导入实验模块。scalar run 显式选择 train/evaluate；stability run 显式选择 search/confirm/held-out/diagnostic，沿用内部 stage A/B/C 和已有预算、晋升与目录要求，不自动运行下一步。参数组合在 bootstrap 前核验。

reward sanity 的配置、计算与产物发布统一由单文件 `reward_validation` 拥有，配置为 `configs/validation/reward.yaml`；execution backend workload 核验由 `benchmarking.execution` 拥有。它们分别使用 `just validation reward` 和 `just benchmark execution`，不属于研究域。

guidance 的描述统计由 `analysis.guidance` 拥有，场景/指标阈值、方向计数和 Gate D 由实验层裁定，并保存至 `decisions.json`。离线报告重算描述统计，原样读取保存的逐场景 passed、指标 passed/方向计数和完整 gate_d；缺少判定文件或字段明确报错，不从重算统计推导替代判定。summary/report 的原有数值与字段含义保持不变。

scalar 比较 YAML 和协议由 reward scalar 实验层加载并核验，向 analysis 传入已解析、已核验的 baseline 与带 arm/checkpoint 标签的训练/评测记录；analysis 不解析研究协议。stability analyze 只读 study.db 和 manifest，将重新生成的 `stage-a-summary.json` 与其他分析产物写到独立输出目录，不更新源 study、原 gate 或候选晋升结果。

`experiments` 拥有采集、reward/校准、GAE/backward、gate、候选晋级和 correctness guards。读取已采集 batch 的 backward-only 实验仍属于实验执行，不属于描述统计。`analysis` 对已保存的 JSON/NPZ、typed evaluation/training summaries 和 Optuna study 重算统计与比较；`analysis.reporting` 仅负责 Markdown 和 Matplotlib 静态图。分析入口、training summary models 和 evaluation artifact readers 的导入不加载 Torch、MetaDrive、Panda3D、planner 或训练执行器。两层复用 float64 分布统计、相关性、cosine、RMSE、梯度向量比较和逐 scenario 配对差值；benchmark measurement 的原字段与算法保持不变。`eco_planner.rl` 根包的现有导出按需加载；访问纯 training summary 不触发 artifact I/O、policy 或 rollout 初始化，访问执行类时才加载所属模块。其符号与原所属模块保持同一对象。

`just experiment <domain> <study> analyze --source-dir <source> --output-dir <output>`是研究实验的显式离线入口。source/output 不能相同或互相嵌套；分析不写回源数据。已有实验运行/汇总入口在原始产物和守卫完成后复用报告流程，默认写入本次运行目录。两种入口均支持 `--no-figures`。派生输出为 `analysis.json`、`report.md`、`figures/*.svg` 与 `figures/*.png`；原有 summary、diagnostics、sample index、study database 等文件保留各自数值含义。只支持新的采集/诊断结构，不提供旧入口、历史产物兼容或版本迁移层。JSON 保存完整分析证据和图路径，Markdown 保留实验裁定、未定义说明和源产物链接。

| experiment | 离线输入 |
| --- | --- |
| `lambda-identifiability` / `objective-decomposition` / `critic-gae-ablation` | `summary.json`、`diagnostics.npz`、`sample_index.json` |
| `reward-calibration` | 根目录 summary/sample index/audit JSON 与 NPZ，original/calibrated 子目录的 summary/diagnostics |
| `energy-sweep` | `matrix_summary.json` 与按 job/guidance 保存的 evaluation summaries |
| `guidance-control-authority` | `episodes.json` 每步原始指标、`intervention_config.json`、`scenarios.json`、`decisions.json` |
| `scalar-reward` | `--config` 显式列出的 protocol、training summaries 和 evaluation 目录 |
| `ppo-stability` | `study_manifest.yaml`、`study.db`，以及已存在的 stage/diagnostic summaries |
| `reward-sanity` | `sanity_report.json`，含原 case 结果与 checks |
| `ppo-reproducibility` | `training_report.json` 及其 `source_dir` 指向的 seed/replay training summaries |
| `execution-backend` | 已通过 workload 验证的 `evaluation_modes.json` |

固定批次统计从各 arm 的已保存数组重算，严格核对 sample index 的长度、重复项与 scenario 顺序。quantiles 使用原 summary 声明的数值轴；保留原 ddof（包括 ablation 的 value target 使用 sample std）、并列 rank 与差值方向。缺少原始数组报错，不能使用旧 summary 冒充重算。报告只引用原 gate、校准、exact replay 等实验裁定，不重新执行或修改这些结论。calibration 的 audit 原文保留；新增 distributions 从 audit NPZ 重算有符号原始量和保存的 score 分布，按 scenario/planning cycle 分组，不重定义原物理限值。

评测比较要求相同 workload、sampler、runtime seed，并按 scenario/map/map seed/noise seed、evaluation mode 与 traffic density 精确配对；重复或缺失配对报错。差值为 comparison − reference。失败运行、失败原因与不可用配对数显式保留；可用配对的统计明确给出分母，不为失败样本补零。正常终止的碰撞/越界 episode 保留有效指标；completed 不代表成功到达。

Scalar final 报告顺序固定为 completion/availability → safety guardrails → paired energy。各 arm 的 completed rate 使用全部 episode 为分母，arrival rate、route completion 均值和 collision/out-of-road/wrong-direction rate 使用该 arm 的 completed episodes；同时保留失败数量和原因。安全统计不限定到配对交集，未知状态不计为安全。每组对照另报 available pairs / total。

主对照为同 training seed 的 A2−A1，辅以 A1−A0、A2−A0；各对照使用自身双方 completed 的 matched episode 交集。A0 只保存一次，不伪造 training seed。每 seed effect estimate 为该交集的 mean energy delta（MetaDrive fuel proxy，mL），负值表示更低；不对失败填补能耗，不设置 energy penalty 或 composite score，提前终止导致的低能耗须结合前两层解释。

协议显式配置 bootstrap 的 confidence_level=0.95、n_resamples=10000、bootstrap_seed=0。实验层将 bootstrap 配置及 training seed namespace 传入分析层；按稳定排序的完整配对键形成 scenario delta，使用 SciPy percentile bootstrap 和独立 RNG。无 pair 时 estimate/CI 不可用；单 pair 保留 estimate、CI 不可用；至少两个常量 delta 允许零宽区间。不重采样 training seed。每个 final 对照分别绘制 seed 0/1/2 点估计、95% CI、零线及缺项；按点估计符号报告降低、零、升高、不可用的 seed 数，CI 是否包含零单独记录。缺项标为部分结果，不用跨 seed 平均替代三点。initial/update-0 单列诊断，不计入 final 方向计数。CI 仅反映固定训练策略及可用场景下的 scenario 不确定性，不是跨 training seed 的总体区间。

相关性统一使用 SciPy Pearson/Spearman 系数；ties 使用库定义，常量或样本不足记 undefined，不增加显著性检验。现有 ddof、cosine、RMSE 和实验 gate 保持原定义。各报告按 reward → credit assignment → actor gradient → behavior 解释自身证据；fixed-batch backward 结果不证明训练后行为改变。Optuna 仍只服务现有 PPO stability 搜索、晋升与诊断。

Scalar reward 比较配置的路径均相对该 YAML 所在目录解析，最小结构如下：

```yaml
protocol: protocol.yaml
a0_evaluation_dir: a0/evaluation
runs:
  - arm: a2
    training_summary: a2/seed-0/summary.json
    checkpoint_label: final
    evaluation_dir: a2/seed-0/evaluation-final
```

`arm` 为 a1/a2；`checkpoint_label` 为现有 CLI 的 initial/final（initial 即 update-0 诊断）。seed/reward profile 来自 typed training summary，须属于 protocol；evaluation 的 checkpoint hash 必须匹配该 training summary 中相应的 initial/final hash。A0 的 held-out pool、seed、horizon 与 sampler 对照 protocol 核验。只运行单个 scalar reward job 时报告该 job；跨 arm 与 seed 的比较由用户显式列出源运行，不从目录名推断，也不选择最佳 seed。

静态图使用 Agg 后端，提供数值标注、单位和独立指标色标。零范数 cosine 和零分母 ratio 为`null`，图上显示 undefined；近共线诊断同时提供 cosine 和 `1 − cosine`，不裁剪原数值。PPO stability 直接调用 Optuna Matplotlib 的 history、parallel coordinate、importance、contour。参数重要性使用 manifest 的 sampler_seed 作为显式 evaluator seed；不改变搜索 sampler 或候选晋级规则。离线 study 通过 SQLite read-only mode 打开，不创建数据库或 study。不足 completed trials、无变化参数或 objective 等导致图不可用时，报告保存具体原因；不伪造图。生成报告不需要交互 HTML、图形 UI、模型 checkpoint 或新训练运行。
