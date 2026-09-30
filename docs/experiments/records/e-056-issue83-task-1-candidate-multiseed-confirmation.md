# E-056 Issue #83 Task 1：candidate multi-seed long-run confirmation / Gate M

[返回实验索引](../README.md) · Issue #83 Task 1 ·
[E-055 候选提名](e-055-issue106-task-3-dose-response.md) ·
[E-052 冻结 control](e-052-issue105-phase-c-canonical-ppo-refreeze.md) ·
[预注册计划](../../research/issue83-task-1-candidate-multiseed-confirmation-plan.md)

**日期 / 类型 / 目的**：2026-09-30 / 正式 candidate multi-seed long-run confirmation / Issue #83 Task 1：
在 canonical k=5（`0.1 s × 5 = 0.5 s`）+ E-052 冻结 PPO control 下，对 calibrated R0 与 #106 提名候选
λ=4 / λ=8 统一运行 seeds {0,1,2} × 100 updates（`calibrated-canonical-confirmation.yaml`，
`ppo.gradient_diagnostics=true`），裁定 Gate M（multi-seed stability）。不重搜 PPO 超参 / reward
representation / cadence / λ。Gate M 裁定在实验记录中给出（与 E-050/E-051/E-054/E-055 一致）。

**裁定：Gate M FAILED — 两个候选均未通过，#83 candidate path 以
"candidate long-run instability / non-reproducibility" 收口；无候选进入 final A2。**
λ=4：0/3 seed 稳定（seed 0/1 训练完成但 Beta boundary collapse，probe boundary mass 0.5295 / 0.4692 ≫
0.2 ceiling；seed 2 训练中崩溃）。λ=8：seed 0 训练中崩溃于 update 70，3/3 完成已不可能，剩余 seed 未运行
（裁定已定，不追加失败证据）。R0 对照 3/3 seed 完成、机械健康。失败模式与
E-039（lr≥1.5e-4@epochs2、4.5e-4）及 E-052（3 个 lr≥3e-4 候选）记录的 Beta 边界塌缩同类：λ 加权
band energy penalty 在 100-update 确认协议的 LR 调度下把纵向 Beta α 压到 < 1，产生 boundary-seeking
分布；精确边界采样使训练 log-prob 非有限（loud fail）或使 final probe boundary mass 远超 ceiling。

## 冻结契约与协议

Task 1 在 #83 冻结研究 contract 下执行（frozen Diffusion Planner / DDIM5；同一 per-seed initial
policy；training S/SC map seeds 0–7；held-out S/SC seeds 16–23；deterministic Beta mean 主评测；E-034
校准 Progress/Comfort；E-038 冻结 efficiency-band energy；canonical execution 0.5 s；execution-trace
fuel proxy；PPO control 取 E-052 冻结结果）。

新增 protocol `configs/experiments/comparison/calibrated-canonical-confirmation.yaml`：arms
`r0 / lam4 / lam8`（reward profiles `plannerrft_no_energy_calibrated_v1`、
`plannerrft_energy_band_lam4_v1`、`plannerrft_energy_band_lam8_v1`），training seeds `{0,1,2}`，
`training.update_count=100`、`ppo.scheduler_total_optimizer_steps=100`、
`ppo.gradient_diagnostics=true`，contrasts `[[r0,lam4],[r0,lam8]]`；训练 overrides 其余与 E-055
逐值相同（E-052 冻结 control + canonical 调度），held-out contract 与 E-055 逐值相同。protocol pin
测试 `tests/training/test_training_workflows.py`（confirmation 协议）与
`tests/training/test_effective_update.py`（gradient diagnostics）覆盖。

### 实现改动（Task 1 范围内，均有测试）

- `extract_arm_metrics` + `GRADIENT_GROUPS`（commit `71b32b9`）：`ppo.gradient_diagnostics=true` 时
  额外报告六组 per-group gradient norm 序列（actor_head_policy / actor_head_entropy /
  shared_trunk_policy / shared_trunk_entropy / shared_trunk_critic / value_head_critic；
  measurement-only，训练语义不变）。
- **probe boundary 测量修复**（commit `2a22b3c`）：`probing.py` 的 probe 采样改为
  `ExplicitGeneratorBetaSampler.draw(..., validate_args=False)`。修复前，final policy 已 boundary
  collapse 时 probe 在精确边界 draw 上 raise `guidance action must be strictly inside (-1, 1)`，
  使 summary 无法写出——测量层缺陷而非应记录的失败本身：probe 的职责就是测量 boundary mass，
  崩溃使其无法报告所测失败。修复后失败仍以高 boundary mass 可见；对健康 policy 采样值逐位不变
  （r0 三 seed 修复前后 probe_after 逐值一致验证）。回归测试
  `tests/training/test_trainer.py::test_probe_reports_boundary_collapse_without_raising_on_exact_boundary_draws`。
  **训练 rollout 侧严格校验保持不变**（invalid action 仍 loud fail，见下文两个训练崩溃 run）。
- 契约登记（commit `bf19314`）：`docs/agents/contracts/experiments.md`（confirmation 协议 +
  gradient diagnostics 报告契约）。

## 命令

```powershell
# 9 matched training runs（r0/lam4/lam8 × seed {0,1,2}，100 updates，canonical k=5 + E-052 冻结 control）
just exp compare train --config configs/experiments/comparison/calibrated-canonical-confirmation.yaml `
  --arm r0 --training-seed 0 `
  --output-dir outputs/studies/scalar-reward/e-056-issue83-task-1-candidate-multiseed-confirmation/r0-seed-0
# ... 其余 arm × seed 同理；本实验实际完成 7 个（lam8-seed-1/2 未运行，见「与预注册计划的偏离」）

# 每臂训练机械健康（diagnose.yaml 见 study dir；覆盖 r0 三 seed）
just exp training diagnose `
  --config .../e-056-issue83-task-1-candidate-multiseed-confirmation/diagnose.yaml `
  --output-dir outputs/studies/scalar-reward/e-056-issue83-task-1-candidate-multiseed-confirmation-diagnostics
```

## Matched guards（runner/analysis 硬校验）

全部 7 个实际运行的 run 均在 clean commit `2a22b3c`（`git_status_short=[]`，含两个训练崩溃 run 的
runtime_metadata）执行；canonical cadence resolved `0.1 × 5 = 0.5 s`；frozen planner hash 前后不变。
Per-seed initial policy hash 各臂逐值一致：seed 0 `049697739ab5…`（=E-052/E-053/E-054/E-055 同一冻结
initial）、seed 1 `feee6d85b681…`、seed 2 `0cd7d93619b2…`（新值，预期内）；崩溃 run 的 initial hash
由 `policy-initial.pt` 以 `policy_state_hash` 同算法复算验证（lam4-seed-2 = seed 2 值、lam8-seed-0 =
seed 0 值）。`probe_before` 全 run 一致（boundary mass 0.0210）。

## 训练结果（9-run 状态表）

| run | initial hash | updates 持久化 | 状态 | 关键证据 |
| --- | --- | ---: | --- | --- |
| r0-seed-0 | `049697739ab5` | 100 | completed | min α/β 1.280/1.906；probe boundary mass 0.0718 |
| r0-seed-1 | `feee6d85b681` | 100 | completed | min α/β 1.964/1.432；probe boundary mass 0.0488 |
| r0-seed-2 | `0cd7d93619b2` | 100 | completed | min α/β 1.996/1.043；probe boundary mass 0.1504 |
| lam4-seed-0 | `049697739ab5` | 100 | completed（collapse） | α<1 始于 update 51，min α 0.457；probe boundary mass **0.5295** |
| lam4-seed-1 | `feee6d85b681` | 100 | completed（collapse） | α<1 始于 update 57，min α 0.544；probe boundary mass **0.4692** |
| lam4-seed-2 | `0cd7d93619b2` | 76 | **训练崩溃** | update-076 rollout：`old_joint_guidance_log_prob` 非有限（精确边界 draw → log-prob −inf）；最后持久化 update-075 min α 0.442 |
| lam8-seed-0 | `049697739ab5` | 70 | **训练崩溃** | update-070 rollout 同一失败；最后持久化 update-069 min α 0.331 |
| lam8-seed-1 | — | — | 未运行 | Gate M 已由 seed 0 崩溃裁定（3/3 不可能） |
| lam8-seed-2 | — | — | 未运行 | 同上 |

失败 seed 全部保留（partial updates / resolved_config / runtime_metadata / 训练日志），不删除、不补跑
伪造完整表格。

## 机制分析

1. **失败模式单一且同源**。四个候选失败全部是 Beta 边界塌缩链路：λ 加权 band energy penalty 持续把
   纵向 Beta α 压低；α<1 后分布 boundary-seeking，精确边界 draw 的 log density 无限。训练 rollout 中
   边界 draw 使 `old_joint_guidance_log_prob = −inf` → 严格校验 loud fail（lam4-seed-2、lam8-seed-0）；
   未触发边界 draw 的 run 训练完成但 final probe boundary mass 0.47–0.53（lam4-seed-0/1）。两个崩溃
   run 最后持久化 update 的 min α（0.442 / 0.331）与完成 run 的 collapse 终态（0.457 / 0.544）同水平，
   即崩溃与完成只是同一终态的两种采样实现。
2. **collapse 时点与 LR 调度对齐**。确认协议 `scheduler_total_optimizer_steps=100`（E-055 为 50）：
   update 50 处 E-056 的 LR 仍在初值约 50%，而 E-055 已衰减到 ~0。lam4 两 seed 的 α<1 起点在 update
   51/57，恰在 E-055 从未经历过的 LR 区间。E-055 λ=4/λ=8 在 50 updates 内 min α ≥ 1.776 无 collapse
   ——E-055 的 dose-response 证据在其 50-update horizon 内仍然有效，本实验不推翻它；instability 是
   50→100 update 延长 + 调度重标定下才暴露的长期性质，而这正是 Gate M 要检验的对象。
3. **与历史失败模式一致**。E-039（lr 1.5e-4@epochs2、4.5e-4）与 E-052（3 个 lr≥3e-4 候选）均记录
   「高 update 强度 → Beta 边界塌缩」为已知失败模式；本实验表明在冻结 control 下，λ=4/λ=8 的
   energy pressure 足以在延长 horizon 下触发同一模式。
4. **R0 对照健康**。三 seed 全部完成：min α/β ≥ 1.280/1.043（> 0.1 floor，且 > 1）；probe boundary
   mass 0.049–0.150（< 0.2 ceiling；seed 2 的 0.150 偏高，如实记录）；seeded-MC post-update KL
   median 3.32e-5 / 4.29e-5 / 8.69e-5（max 同量级，≪ target_kl 0.006，无 runaway），analytic Beta KL
   与 k3 single-draw 同量级一致（3.35e-5/3.88e-5/7.72e-5）；六组 per-group gradient 序列全部有限并
   已持久化（`gradient_diagnostics`）。R0 在 100 updates 下稳定，排除「确认协议本身对任何 arm 都
   不可训练」的混淆解释。

## Gate M 判定

```text
Verdict: FAILED（λ=4 与 λ=8 均失败；无候选进入 final A2）
```

| 条件 | λ=4 | λ=8 |
| --- | --- | --- |
| 3/3 seeds 完成；失败 seed 不删除 | ✗（2/3 完成，seed 2 训练崩溃，证据保留） | ✗（0/1 完成，seed 0 训练崩溃；seed 1/2 未运行） |
| 3/3 无 boundary collapse / KL runaway / invalid action | ✗（seed 0/1 boundary mass 0.5295/0.4692 > 0.2；seed 2 invalid action） | ✗（seed 0 invalid action） |
| 跨 seed effect 方向一致 / 非 outlier 驱动 / 50→100 无反转 / trade-off 可解释 | 未评估（前置稳定性条件已失败） | 未评估（同左） |
| provenance 完整、commit clean | ✓（7/7 run pin `2a22b3c` clean） | ✓（同左） |

R0 对照通过全部稳定性判据。Gate M 的 effect 方向类条目依赖 held-out 评测，因候选在稳定性前置条件上
已失败而未评估（见下）。

## 与预注册计划的偏离

计划（`docs/research/issue83-task-1-candidate-multiseed-confirmation-plan.md`）预注册 9 训练 + 21
held-out 评测 + compare analyze。实际执行偏离及理由：

1. **lam8-seed-1/2 未训练**。Gate M 要求 3/3 完成；lam8-seed-0 于 update 70 崩溃后该条件已不可能
   满足，继续训练只累积同型失败证据、不改变裁定（用户裁定：直接收口）。被中止的 lam8-seed-1
   partial 目录（44 updates，人工中断产物、非失败证据）已删除。
2. **21 次 held-out 评测与 compare analyze 未执行**。Gate M 为顺序门：稳定性是 effect 分析的前置
   条件；候选失败后 held-out trade-off 分析无对象（且 lam4-seed-2 / lam8-seed-0 无 final policy 可
   评）。预注册计划的 Gate M 清单本身按此顺序定义。
3. **diagnose 仅覆盖 r0 三 seed**。五个完成 summary 全量 diagnose 在 lam4-seed-0 update 58 中止：
   seeded-MC post-update KL cross-check 对已 collapse 的 policy 不可定义（对 old Beta 参数采样命中
   精确边界 → log density 无限 → `FloatingPointError`）。这是测量的数学性质而非实现缺陷（analytic
   KL 可算且有限），未为绕过它修改 estimator 契约（`post_update_kl` 的 float 序列被多个
   analysis/training 消费方依赖）。lam4 两 seed 的 collapse 证据由其 summary.json probe block 与
   β 参数序列完整承载；diagnose.yaml 头部登记此边界。
4. **probe 测量修复发生在首组训练之后**。首次 r0×3 + lam4×2 在修复前 commit（`71b32b9`）运行，
   lam4-seed-0/1 在 trainer probe 处崩溃（测量层缺陷，见「实现改动」）。修复后全部 run（含 r0
   重跑）统一在 `2a22b3c` 执行以满足「同一 clean commit」契约；r0 重跑与首次逐值一致（final hash
   与 probe_after 均相同），证明修复对健康 policy 训练与测量值中性。

## 局限与结论边界

- **结论范围**：本记录裁定的对象是「λ=4 / λ=8 在确认协议（seeds {0,1,2} × 100 updates、冻结
  control、canonical k=5）下的 multi-seed 长期稳定性」。结论是两者不稳定、不可复现为可用候选；
  不构成对更小 λ（如 1/2）或其他 reward representation 的否定——它们未经本协议检验。
- **E-055 证据仍有效**：λ=4/λ=8 的 50-update dose-response 效应（E-055）在其 horizon 与调度内
  成立；本实验表明该效应不能外推到 100 updates（collapse 后的 policy 不再是可用终点）。
- **未评估 held-out 行为**：候选 final policy 的 held-out energy/behavior 效应未在本协议下测量
  （collapse 已使 policy 不可用）；不引用 E-055 数字作为 100-update 行为证据。
- **λ=8 失败证据为 1 seed**：seed 0 的训练崩溃对 Gate M 3/3 条件是决定性的，但 λ=8 的 collapse
  具体时点/形态跨 seed 变化未刻画（seeds 1/2 未运行）。
- **probe 修复的边界**：`validate_args=False` 只影响测量路径；训练 rollout 的严格校验未动，边界
  draw 在训练中仍 loud fail（两个崩溃 run 即证据）。

## 产物

| 产物 | 路径 |
| --- | --- |
| 协议 | `configs/experiments/comparison/calibrated-canonical-confirmation.yaml` |
| 训练 runs（7） | `outputs/studies/scalar-reward/e-056-issue83-task-1-candidate-multiseed-confirmation/{r0,lam4}-seed-{0,1,2}/`、`.../lam8-seed-0/` |
| 训练日志 | `.../train-{arm}-seed-{n}.log`（崩溃 run 的完整 traceback 在 `train-lam4-seed-2.log` / `train-lam8-seed-0.log`） |
| 训练诊断（r0 三 seed） | `outputs/studies/scalar-reward/e-056-issue83-task-1-candidate-multiseed-confirmation-diagnostics/`（config：study dir `diagnose.yaml`） |
| 契约登记 | `docs/agents/contracts/experiments.md`（confirmation 协议 + gradient diagnostics） |
| 预注册计划 | `docs/research/issue83-task-1-candidate-multiseed-confirmation-plan.md`（状态：已执行） |
