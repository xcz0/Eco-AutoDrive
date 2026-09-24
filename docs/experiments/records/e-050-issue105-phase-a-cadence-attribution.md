# E-050 Issue #105 Phase A：matched cadence-PPO attribution（analytic Beta KL + k=1 diagnostic）

[返回实验索引](../README.md) · [Issue #105 Phase A](../../research/) ·
[E-039 PPO control](e-039-issue94-task-f-effective-update-region.md) ·
[E-049 canonical transfer](e-049-issue83-task-1b-effective-update-transfer.md)

**日期 / 类型 / 目的**：2026-09-24 / 正式 matched 训练归因诊断 / 完成 Issue #105 Phase A：在**同一 clean
commit、同一 calibrated R0、同一 initial policy、同一 training seed / replay / scenario 池、batch=128、
50-update 预算**下，只改变执行前缀长度（k=1 diagnostic 与 canonical k=5），把 E-039→E-049 的
effective-update 下降归因到 actor/state/advantage geometry、reward/value scale、physical-time credit
semantics、critic/shared-trunk/global-clipping coupling 四类机制之一或组合（Gate A）。不进入 optimizer
搜索，不做 counterfactual arms。

**裁定（Gate A）**：canonical k=5 的 actor effective update 下降**主要来自 critic/shared-trunk/global-clipping
coupling**：k=5 的 per-transition reward/value scale（reward total 4.94×、value target mean 3.35×）把
critic 梯度与 total pre-clip norm 抬高 ~4.9×，使 global clipping 对 actor-critic 共享梯度施加更强
down-scaling（effective global clip coefficient 0.0192 → 0.00393，≈0.20×），而 **actor_head policy
gradient norm 几乎不变（ratio 0.997）**、z-normalized advantage 结构不变（std ratio 1.000）。因此
primary 不是 actor/state/advantage geometry；reward/value scale 与 physical-time credit semantics 是
upstream 原因，通过 critic/clipping 路径传导。

## 运行来源与冻结契约

- Git commit `1a0fccef2efb7f5064edf16118430845a05383fa`（branch `issue83-engery-select`，
  `git_status_short=[]` clean，由 `runtime_metadata.json` 记录）；上游源码 `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`；
  EMA 276 tensors / 6,042,628 parameters。
- Windows-10-10.0.26200-SP0、Python 3.10.20、PyTorch 2.12.1+cu126、CUDA:0（RTX A4000）、Lightning 2.6.6、
  MetaDrive 0.4.3、pydantic 2.13.5；rollout `bf16-mixed`。
- protocol `experiments/comparison/calibrated.yaml` arm `r0` =
  `plannerrft_no_energy_calibrated_v1`（E-034 冻结 Progress/Comfort 校准）；base job
  `jobs/training/ppo_conservative`；DDIM5。
- 两臂唯一差异是 `training.diagnostic_execution_steps`（k=1=1，k=5=5 canonical）；`ClosedLoopCadenceConfig`
  仍 pin 在 canonical 0.1 s × 5 = 0.5 s，实际执行前缀由 runtime metadata `execution_steps` 显式记录
  （k=1 记录 1，k=5 记录 5）。其余 resolved 值逐项相同：lr=1.5e-4、epochs=1、batch=minibatch=128、
  target_kl=0.006、max_gradient_norm=0.5、`ppo.gradient_diagnostics=true`、update_count=50、
  16 scenarios（S/SC × seeds 0–7）× 8 transitions/env = 128 transitions/update、training/runtime seed 0、
  replay 0、`env.horizon=80`。
- initial policy hash `049697739ab5d6e7bd212938bbac82c35215eb9ed14c02557952098a9718d05d`（两臂一致，与
  E-038/E-040/E-048/E-049 一致）；frozen planner hash 前后一致（未被破坏）。

**命令**：

```powershell
just exp training cadence-attribution run `
  --output-dir outputs/studies/scalar-reward/e-050-issue105-phase-a-cadence-attribution
```

配置清单：`configs/experiments/training/cadence-attribution.yaml`；实现位于
`src/eco_planner/experiments/training/cadence_attribution/`，post-update KL 重算（含 analytic）位于
`src/eco_planner/rl/optimization/update_diagnostics.py`，analytic Beta KL 纯函数位于
`src/eco_planner/analysis/training.py`。

## 逐臂测量（50 updates）

| 指标 | k=1 diagnostic | k=5 canonical | k5/k1 |
| --- | ---: | ---: | ---: |
| analytic Beta KL median | 1.432e-6 | 6.284e-7 | 0.439 |
| seeded-MC post-update KL median | **1.812e-6** | **3.258e-7** | 0.180 |
| single-draw k3 median | 1.348e-6 | 6.040e-7 | 0.448 |
| analytic KL max | 1.742e-5 | 9.046e-6 | — |
| policy ratio change | 1.643e-3 | 1.097e-3 | 0.668 |
| probe guidance RMS shift | 0.03384 | 0.02205 | 0.652 |
| min α / β | 1.9080 / 2.0000 | 1.9304 / 1.9864 | — |
| boundary mass (probe after) | 0.02222 | 0.02197 | — |
| collision / OOR | 0 / 0 | 0 / 0 | — |
| episode length first → last | 8.0 → 8.0 | 8.0 → 8.0 | — |
| actor_head param delta | 0.028490 | 0.017714 | 0.622 |
| shared_trunk param delta | 1.797386 | 1.793632 | 0.998 |
| value_head param delta | 0.050853 | 0.050872 | 1.000 |
| value target mean median | 5.158 | 17.271 | 3.349 |
| reward total median | 100.879 | 498.761 | 4.944 |
| raw / center advantage std median | ×1 | ×5.031 | 5.031 |
| z-normalized advantage std median | 1.000 | 1.000 | 1.000 |
| pre-clip gradient norm median | 26.077 | 127.529 | 4.890 |
| effective global clip coefficient median | 0.01920 | 0.00393 | 0.204 |
| grad actor_head_policy | 0.36486 | 0.36358 | 0.997 |
| grad shared_trunk_policy | 0.009641 | 0.005893 | 0.611 |
| grad value_head_critic | 16.085 | 78.766 | 4.897 |
| grad shared_trunk_critic | 20.509 | 100.288 | 4.890 |
| grad actor_head_entropy | 0.007095 | 0.006855 | 0.966 |
| grad shared_trunk_entropy | 3.14e-5 | 2.73e-5 | 0.869 |
| final policy hash | `0c0ff16a42b1…` | `7a7809264b2f…` | — |

k=5 的 MC KL median `3.258e-7` 与 final policy hash `7a7809264b2f` **逐值复现 E-049**（同为 E-039 candidate
+ calibrated R0），确认 cadence transfer 失败稳定可复现且本次运行 provenance clean。

## Gate 判定（复用 E-039 Gate F 阈值）

| 条件 | k=1 | k=5 |
| --- | :---: | :---: |
| c1 median post-update KL ≥ 1e-6（MC） | ✓（1.812e-6） | ✗（3.258e-7） |
| c2 KL within target / 无 runaway | ✓ | ✓ |
| c3 policy ratio 变化 ≥ 1e-4 | ✓ | ✓ |
| c4 probe RMS shift ≥ 0.01 | ✓ | ✓ |
| c5 无 Beta boundary collapse | ✓ | ✓ |
| c6 无 behavioral collapse | ✓ | ✓ |
| c7 held-out 超噪声（≥1 指标 ≥0.1%） | ✓ | ✗ |
| passed | 7/7 | c1、c7 fail |

canonical k=5 在 c1（analytic 6.284e-7 与 MC 3.258e-7 都 < 1e-6）与 c7 上失败；diagnostic k=1 全通过。

## Matched held-out（始终执行）

initial → final policy，matched 协议（S/SC seeds 16–23、horizon 300、runtime seed 760025、ddim5）：

| 指标 | initial | k=1 final（相对变化） | k=5 final（相对变化） |
| --- | ---: | ---: | ---: |
| mean speed (m/s) | 9.998465 | 9.964381（−0.341%） | 9.996741（−0.017%） |
| energy intensity (mL/km) | 47.047213 | 46.983967（−0.134%） | 47.043657（−0.008%） |
| route completion | 0.946576 | 0.945151（−0.151%） | 0.946444（−0.014%） |
| distance (m) | 169.4826 | 169.2411（−0.143%） | 169.4928（+0.006%） |
| arrive_dest fraction | 0.8125 | 0.8125 | 0.8125 |
| collision / OOR | 0 / 0 | 0 / 0 | 0 / 0 |

k=1 的 held-out 变化（speed −0.341%、energy −0.134%）超过 E-031 0.1% 噪声界；k=5 全部 < 0.02%，未超噪声。
held-out 只用于归因，**不改写** E-049 的历史 Gate T2 裁定。

## Gate A 归因结论

四类机制证据（matched，唯一变量为 cadence）：

- **actor/state/advantage geometry：不支持为 primary**。z-normalized advantage std ratio=1.000（归一化按
  full batch 结构性消除 scale），actor_head_policy gradient norm ratio=0.997、actor_head param delta ratio
  0.622，与「geometry 主导」不符——advantage 形状本身未系统改变 actor-head policy 梯度尺度。
- **reward/value scale：成立但为 upstream**。reward total 4.944×、value target mean 3.349×、raw/center
  advantage std 5.031×，符合 ADR 0039 的多 substep per-transition reward 求和（≈5×）语义。
- **physical-time credit semantics：成立但为 upstream**。0.5 s transition + `gamma=0.99`（有效视界 ~10 s→~50 s）
  改变 return 结构；value target 3.349×（小于 reward 的 4.944×）即折扣/截断/bootstrap 结构差异的体现，
  bootstrap abs mean ratio 0.922。
- **critic/shared-trunk/global-clipping coupling：主要 proximal 机制**。critic 梯度（value_head_critic 4.897×、
  shared_trunk_critic 4.890×）与 total pre-clip norm（4.890×）几乎完全随 value scale 放大；global clipping
  因此把共享 actor-critic 梯度整体缩放 mgn/total，effective clip coefficient 由 0.0192 降到 0.00393（0.204×），
  在 actor_head policy 梯度尺度不变（0.997×）下使 actor 的实际步长近似减半（analytic KL 0.439×、ratio
  change 0.668×、probe RMS 0.652×、actor_head param delta 0.622×）。

**Gate A 结论**：缩小到 **critic/shared-trunk/global-clipping coupling**（primary），其 upstream 驱动为
**reward/value scale + physical-time credit semantics**；未缩小到 actor/state/advantage geometry 为主因。
剩余不确定性：single seed / 单 no-traffic 训练池，未分离 value-loss scaling 与 global clipping 的各自贡献，
该分离属于 Phase B diagnostic arms（不得直接进入正式 control）。

## 实现变更（本次任务落地）

- **analytic Beta KL 落地并验证**（Issue 完成判据）：`analysis.training.beta_kl` 给出 per-dimension
  `KL(Beta(old)‖Beta(new))` 闭式（`lgamma`/`digamma`，float64，affine Jacobian 抵消），`post_update_kl_series`
  新增 `post_update_kl_analytic`/`_median`/`_max`，与 seeded-MC、k3 并列。k=5 analytic median 6.284e-7 与
  MC 3.258e-7 同量级、k1 1.432e-6 与 1.812e-6 同量级，交叉一致。
- **k=1 diagnostic 训练执行**：新增显式 `training.diagnostic_execution_steps`（默认 `null`=canonical），
  `TrainingJobConfig` 校验范围 `[1, PLANNER_HORIZON)`，经 trainer → `VectorRolloutCollector` →
  `VectorMetaDriveEnv` 传递，并写入 runtime metadata；不放松 canonical cadence pin（ADR 0039 允许 matched
  causal intervention 显式覆盖前缀）。
- **matched 归因工作流**：新增 `training cadence-attribution run/analyze`、study manifest 与离线测量
  （advantage raw/center/z、value target、reward scale、offline GAE/bootstrap provenance 核对、gradient
  diagnostics、clipping、parameter delta、ratio、Beta、held-out），Gate A 证据按四类机制输出。
- 顺带修复：`test_trainer` 既有 `write_training_runtime_metadata` monkeypatch 需接受新 kwarg。

## 验证与产物

- 测试：`just test-target tests/training/test_effective_update.py tests/training/test_cadence_attribution.py
  tests/training/test_trainer.py tests/training/test_training_workflows.py tests/configuration/test_jobs.py`
  （新增 analytic KL 闭式与 k1/k5 schema、collector 转发、cadence-attribution run/CLI 覆盖）；`just lint`、
  `just typecheck` 0 errors。
- 只读核验：两臂 `git_status_short=[]`、`git_head=1a0fcce`、execution_steps 1/5、initial policy hash 一致、
  frozen planner hash 前后一致、offline GAE advantage/value-target 与记录值 provenance 一致、无非有限值。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-050-issue105-phase-a-cadence-attribution/`
  - `study_manifest.yaml`：resolved 研究清单。
  - `k1/`、`k5/`：各臂 `summary.json`（逐 update 测量）、`resolved_config.yaml`、`runtime_metadata.json`、
    `policy-*.pt`、`updates/update-NNN/*.npz`。
  - `heldout/initial|k1|k5/`：matched held-out 评测产物。
  - `summary.json`：逐臂 metrics、Gate、held-out 与 `attribution`（含 Gate A 证据）。
  - `analysis.json`、`report.md`、`figures/post-update-kl.*`。

## Issue Task provenance 清单

- commit `1a0fcce`、branch `issue83-engery-select`、`git_status_short=[]`（clean 运行）；上游 `a3a621f0…`；
- 环境：Windows-10-10.0.26200 / Python 3.10.20 / torch 2.12.1+cu126 / CUDA:0 RTX A4000 / Lightning 2.6.6 /
  MetaDrive 0.4.3；
- resolved config：各臂目录 `resolved_config.yaml`（cadence/reward/PPO/scenarios/execution_steps 已核对）；
- checkpoint identity：initial policy `049697739a…`（与 E-038/E-040/E-048/E-049 一致）、frozen planner hash
  前后不变；k=1 final `0c0ff16a…`、k=5 final `7a780926…`（=E-049）；
- seeds：training/runtime seed 0、replay 0、16 scenarios S/SC seeds 0–7；DDIM5；
- cadence：0.1 s × 5 = 0.5 s（canonical pin 不变；k=1 为显式诊断覆盖，runtime metadata 记录）；
- commands / artifact path：见上；
- failure 状态：无训练/数值崩溃；canonical k=5 c1、c7 fail（冻结阈值比较明确）。

## 结论边界

**支持**：在 canonical k=5、该 initial policy、该 no-traffic 池、calibrated R0 下，E-039 candidate 的 per-update
actor effective update 相对 k=1 近似减半，proximal 机制是 critic/shared-trunk/global-clipping coupling
（critic 梯度与总 pre-clip norm 随 reward/value scale ~5× 增大，global clip 对共享梯度 ~5× 强 down-scaling），
upstream 为 reward/value scale 与 physical-time credit；actor/state/advantage geometry 不是 primary。
k=5 MC KL 与 final policy hash 逐值复现 E-049。

**不支持 / 限制**：

- 单 seed、单 no-traffic 训练池、单 initial policy；不构成跨 seed 总体结论。
- 未分离 `value_loss` scaling 与 global clipping 的独立贡献（同一 total backward 下两者耦合），该分离是
  Phase B diagnostic arm。
- analytic KL 与 MC KL 方向一致但存在小量差异（k=5 6.284e-7 vs 3.258e-7），差异来自 MC 有限抽样噪声；
  阈值比较以两者同侧为准，不把单点读作精确值。
- held-out 变化方向（speed 小幅下降）不解释为节能或行为改善；k=5 未超噪声。
- 不做 optimizer 搜索、不重排 reward/PPO；不进入 Phase B/C，不宣称正式 control 已 re-freeze。

**后续（Issue #105）**：Phase B 按 Gate A 证据最小化执行 critic-coupling counterfactual（value-loss scaling /
shared-trunk critic gradient / global clipping，必要时 actor-only/separated clipping 仅作 diagnostic），
保持 k=5 cadence 与 calibrated R0；Phase C 仅在 A/B 给出足够归因后考虑 canonical PPO re-freeze。
