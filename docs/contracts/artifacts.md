---
kind: contract
status: active
scope: [artifacts, provenance, readers, failures, tracking]
read_when:
  - changing artifact writers readers or missing-result semantics
  - validating experiment inputs or training tracking continuation
---

# Artifacts contract

MUST／MUST NOT 表示强制的软件保证。

本篇拥有 writer/reader、provenance 与失败含义；指标估计与研究判据分别归 [planning/evaluation](../research/protocols/planning-and-evaluation.md)、[training](../research/protocols/training.md)、[diagnostics](../research/protocols/diagnostic-studies.md)。
怎样登记真实运行仍见 [experiments README](../experiments/README.md)，本篇不复制登记模板或历史索引。

## Provenance 与身份

每次运行 MUST 保存 resolved config、显式 overrides、runtime Git head/branch/dirty、实际 seeds、请求与解析后的 accelerator/precision、实际 device、依赖环境和场景特征。
Sampler 的 backend/profile、步数、初始尺度、stochasticity、时间序列和 parity 标签必须可追溯；
地图 seed 与 diffusion/action seed 分别记录。Checkpoint label/path/state hash 必须对应所用策略。
Resolved config 只作 provenance，不是第二个 metric/result source。

正式结果的 clean-commit 要求由登记规则拥有。
Writer 保留 Git/runtime metadata，不复制 Python source 或 tracked diff，不因此添加新的 preflight／manifest／格式握手。
历史 records/artifacts 保持原样，不用当前 config 补造历史参数。

## Evaluation trace、summary 与失败

每 episode MUST 保存 summary 与 trace；启用 video 时保存闭环 GIF。
Raw observation 必须原值保存，不能以 normalized/device precision 副本替代。
Trace 至少能够对齐：

- Standard-normal 的未缩放 initial noise、完整 joint prediction、planning anchor；
- 目标与实际状态、逐点执行误差、逐子步 route heading error 与 domain speed/stopped/wrong-direction；
- Native MetaDrive step/episode energy 与 execution fuel proxy、配套实际 distance、终止 flags；
- 有交通时的独立 warmup、对象 IDs／数量、最近交通距离、history validity；
- 有 guidance 时的 reference joint prediction、action／targets、五步 objective delta、应用梯度 L2/max、未应用 neighbor gradient L2 和零速计数。

动态数组 MUST 在 planning、simulator、warmup 轴一致；
plan index 有序，实际 prefix 与 canonical 执行及末尾 terminal flag 对齐，计数非负。
保存 initial-state validity、普通／route 限速与 validity。
Guidance 数组要么完整出现，要么完整缺失，不能只写其中一部分。

| 状态 | 保存与解释 |
| --- | --- |
| Complete | 已完成 episode 的 typed metrics；可能因 collision/out-of-road 结束，不等于到达。 |
| Partial failed | 保留已产生的 trace 与适用能耗、failure 信息；不伪造成完整 episode。 |
| Empty failed | 无执行 trace 能耗值为 null，不填零。 |
| Undefined | 保留 null／validity，例如零距离 per-km；不能作为有效零进入统计。 |

Job/episode summary 和 runtime metadata MUST 严格拒绝多余字段、NaN/Inf，结果模型不可变。
Completed episode 的通用指标只保存在嵌套 metrics。
Trace 字段集合、NumPy shape/dtype/有限性 必须按 schema 验证，缺失或未声明数组都失败。
Reader 在 I/O 边界验证一次，再核对 trace 与 typed episode 的 status/count、route/traffic、seed pairing 及接口误差语义。
当前 evaluation 格式不增加 version 字段或历史 artifact 转换／兼容层。

通用 episode metrics 由完整执行事实唯一聚合。
Summary、matrix、报告 MUST 消费同一 typed 结果，不从 trace/config 重新计算同名 metric，亦不得把 native energy 替代 proxy。
实验特有 retention/safety/gate 留在研究层，不能塞进通用 structural validator。

Evaluation 只对显式 EpisodeFailure 保存阶段、异常类型、消息、traceback 并继续后续场景；
job 最终失败、CLI 非零。配置、checkpoint、runtime 初始化、artifact I/O 和未分类程序错误直接传播。
失败回合不能删除；完成与失败必须标注。
矩阵只接受预定义网格或其非空子集，缺少必需配置、summary、trace、metadata 或要求的可视化必须失败，不静默跳过坏样本。

## Training、replay 与 checkpoint

Training 与 evaluation 保持独立 schema：evaluation 不保存或聚合训练 reward。
Training episode audit MUST 保存 reward_total、base_total、safety_gate、五个 component 与独立 diagnostic 字段，不能用旧 dense_reward/terminal_override 混指不同 objective。
同时保存 policy context、Beta 参数、base/guidance action、old log-prob/value、initial noise、diffusion/action RNG states、episode status、collision flags、native/proxy energy、distance、强度、denominator validity、seeds，以及 [training contract](training.md#reward-与-audit-对齐) 的子步数据。
不保存完整 DDIM denoise chain。

每 run 保存 resolved config、runtime metadata、policy exports、training-state checkpoint 和严格 summary。
Policy export 仅含严格 checkpoint format_version 与 policy trainable state dict；不含 planner、optimizer、rollout 或 RNG。
Training-state checkpoint 的必要内容和恢复范围归 training contract，writer/reader 必须完整保持这些状态，不能用 export 或 seed 代替。

Update artifact 分别保存 evaluated minibatches、实际 optimizer steps、KL trigger、raw/normalized advantage 统计、更新后 full-batch ratio 及可选 gradient diagnostics。统计定义归 training protocol。
Audit、NPZ、sample index 与 summary 的身份／长度必须一致；
固定批次的 per-arm reward 诊断数组长度等于 sample 数，原始 substep 事实仍由源 batch audit 拥有。

## Matched inputs 与离线读取

Comparison 输入 MUST 显式列 protocol、arm、training summary、checkpoint label、evaluation path 及可空 baseline；相对路径按输入 YAML 所在目录解析，不按目录名推断实验身份。
Reader 核验完整训练池、同 seed 各臂控制条件、initial policy/planner/probe/RNG 身份，以及 evaluation checkpoint hash/label 与 training initial/final 状态一致。
Held-out 场景、seed、horizon、sampler 必须符合显式 protocol。
Scenario/map/map seed/noise seed、evaluation mode、traffic density 的配对键不得重复或缺失。
失败和不可用 pairs 保留；unknown 不能计安全，undefined 不能补零。

Offline analysis MUST 只读源证据，输出目录与 source 不相同或互相嵌套。
Run 可在保存原始证据后生成报告，standalone analyze 从持久化数据独立再生成。
报告不训练、不 backward、不重跑 simulation，不重裁已保存 gate；测量执行层与报告层分开。
Reward/credit 必须读取原始 arrays 核验长度、重复身份与 scenario 顺序，不能拿旧 summary 冒充缺失数组。
Critic attribution 的 offline reporting 可核对 advantage，不重算梯度测量。

下列是既有证据输入导航，不是新增通用 manifest：

| 工作流 | 必需证据类别 |
| --- | --- |
| Reward | summary、diagnostics NPZ、sample index、audit JSON/NPZ |
| Credit | summary、diagnostics NPZ、sample index、实际 calibration/reward config 与原 gate |
| Comparison | 显式 YAML protocol 与 typed training/evaluation summaries |
| Guidance authority/horizon/deferral/decomposition | episodes、intervention config、scenarios、decisions |
| Frozen-policy bridge | 同上，并含 same_state |
| Guidance sweep | matrix summary 与对应 evaluation summaries |
| Training grid/diagnose/eval | 持久化测量 summary 与各 run 事实 |
| Critic attribution | diagnostics NPZ、sample index、summary、diagnostic config、runtime metadata |

Help、offline analyze、summary/trace/report reader MUST 不加载 Torch、MetaDrive/Panda3D 或渲染器。
仅生成图片时加载绘图库并使用 headless backend。Analysis 不反向依赖 experiments，不将库调用链写成规范；
具体字段集合仍由 machine-readable schema 拥有。

## Tracking 身份与续写

Tracking 是研究 artifact 的索引和可视化，不接管 schema 或科研结论。
关闭 tracking 不建立数据库／Run。Collector、reward 与 PPO 数学不直接调用 tracking；
训练编排每个成功记录的 update 发布一次已有 summary，异常传播。
曲线 step MUST 是从零开始的绝对 update_index，恢复后不重置。
有限标量之外的缺失可选指标不发送、不补零；native/proxy 名称和单位分开。

全新训练创建独立 Run。Checkpoint 可保存 `run_id/tracking_uri`，恢复沿用该身份；关闭 tracking 仍保留继承身份并记录 metadata。Run 不存在或 URI 不匹配直接失败，不另建替代 Run。
同一 Run 固定 model、sampler、guidance、policy、reward、PPO（含 scheduler horizon）、runtime、scenarios、env、map query radius 和其余 training 参数；
允许 invocation 改变 job name、resources、tracking、目标 update 数及 resume 路径。这些写入规则不承诺跨执行配置的精确续训。

参数以 config.* 保存，每 invocation 独立保存执行参数与 artifacts。
旧 checkpoint 缺原始 resolved config 时只能以 continuation_config.* 标明恢复起已知配置，并注明历史未记录；
原始配置存在则先校验一致再关联，不能用当前配置冒充历史 provenance。
续写先检查 metric history，只补 checkpoint summaries 中缺失的 metric/step；已有点超过 checkpoint 或同一步值冲突则拒绝。

Invocation artifacts 包括 config、metadata、initial/final policy、按显式间隔选取的 update policy 及最新 training-state checkpoint；正式 rollout NPZ 保留原输出目录。
正常训练和上传均成功才 FINISHED，异常 FAILED，用户中断 KILLED；终结日志失败不得遮蔽原训练异常。

## 来源与代码导航

来源：[旧 system contract](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/agents/system-contract.md)、[training](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/agents/contracts/training.md)、[experiments](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/agents/contracts/experiments.md)。
接受依据包括
[ADR 0033](../adr/0033-track-ppo-with-fabric-and-mlflow.md)、
[0037](../adr/0037-simplify-experiments-and-report-seed-effects.md)、
[0038](../adr/0038-consolidate-scientific-workflows.md)、[0039](../adr/0039-unify-closed-loop-cadence.md)。
本篇不迁移历史 artifacts、不改 records，也不根据历史共同资产表创建新的当前规范。

| 任务 | 实现定位 | 相关测试（未运行） |
| --- | --- | --- |
| Evaluation schema / failure | [artifacts](../../src/eco_planner/evaluation/artifacts/) | [artifacts](../../tests/evaluation/test_artifacts.py) |
| Training audit / persistence | [RL artifacts](../../src/eco_planner/rl/artifacts/)、[training state](../../src/eco_planner/rl/training_state.py) | [rollout](../../tests/training/test_rollout.py)、[PPO](../../tests/training/test_ppo.py) |
| Fixed batch | [fixed batch](../../src/eco_planner/rl/rollout/fixed_batch.py) | [fixed batch](../../tests/training/test_fixed_batch.py)、[credit](../../tests/training/test_credit_assignment.py) |
| Pairing / read-only analysis | [comparison inputs](../../src/eco_planner/experiments/comparison/inputs.py)、[analysis](../../src/eco_planner/analysis/) | [scalar effects](../../tests/analysis/test_scalar_effects.py)、[reports](../../tests/analysis/test_reports.py) |
| Tracking | [tracking](../../src/eco_planner/rl/tracking.py) | [tracking](../../tests/training/test_tracking.py)、[simulation tracking](../../tests/simulation/test_training_tracking.py) |
