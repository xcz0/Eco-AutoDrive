# E-061 Issue #113 Task 1 — fuel proxy 结构性证明与 E-060 执行轨迹速度归因

**日期 / 类型 / 目的 / 状态**：2026-10-15 / 诊断研究（Issue #113 Task 1，P0/Gate P1）/ 对 `MetaDriveFuelProxyProvider` 做结构推导、合成反事实与 trace→公式→summary parity 审计，并对 E-060 final A2−A1 matched evidence 做精确速度水平 vs 速度分布形状分解 / **已完成（2026-10-15；分析计划于运行前冻结，结果与 Gate P1 裁定见下文）**。

**代码与运行身份**：commit、branch、dirty status 见本文「配置与命令」节运行时登记；本诊断为 source-mode 离线重算，不产生新仿真运行，无 runtime metadata；证据源为 E-060 既有 artifact（只读）。

**环境与模型**：本机 `.venv`（Python 3.10）；不加载 planner/policy checkpoint，不运行 simulator；分析对象为已保存的逐子步执行 trace 与 episode summary。

**研究条件**：E-060 final matched evidence —— A0 frozen planner（`heldout/a0-frozen-planner`，git pin `3bb59d9`）、A1 calibrated R0（E-057 复用，final policy hash `94c6dfd93a79`/`c239cc354925`/`4b5486d86ee1`，seeds {0,1,2}）、A2 λ=1（E-058 复用，final policy hash `55eb67f1d6a1`/`24d37e198cc0`/`b5750d4d95f3`，seeds {0,1,2}）；held-out S/SC map seeds 16–23（16 场景）、noise/eval seed 760025、DDIM5 stochasticity=0、deterministic Beta mean、canonical k=5（0.5 s）、horizon 300、no_traffic、warmup 0。相关规范：[Planning/evaluation protocol](../../research/protocols/planning-and-evaluation.md)、[Execution contract](../../contracts/execution.md)、[ADR 0032](../../adr/0032-separate-online-proxy-and-offline-fastsim-energy.md)、Issue #113、#83（E-060）。

## 冻结的分析计划（预注册；先于任何正式结果产生）

冻结机制：机器可读配置 `configs/experiments/energy-proxy/analysis.yaml`（本文引用其全部字段）+ 本节自然语言计划。运行前两者已定稿；运行后不得回改。

### 证据可用性先核验（P0 前置）

- 输入只读 E-060 manifest `outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison/comparison.yaml` 及其引用的评测目录；经 `load_comparison` 全量校验 matched 条件与 checkpoint pin（含 24 个 run 的 initial-hash/frozen-planner-hash/probe/matched resolved conditions/checkpoint hash 交叉核对）。
- 逐 episode 可用性：final 范围 A0 16 + A1 3×16 + A2 3×16 = 112 episodes；trace 状态、summary、resolved config、runtime metadata 必须可读。任一不可读 → 该 episode 标记 unavailable 并显式报告，不以 aggregate summary 伪造逐子步数据，不静默跳过。
- 覆盖率结论（2026-10-15 只读预检）：112/112 final episodes 的 `trace.npz`（逐子步 `executed_speed_mps` / `executed_step_distance_m` / `executed_fuel_proxy_step_energy_ml`）与 `summary.json` 全部在本地可读，`trace_status=complete`；manifest 引用的 22 个评测目录无缺失。**无 unavailable 缺口，无需最小复跑。**（initial/update-025/update-050 中间 checkpoint 共 240 episodes 亦在本地，但按冻结范围只做 pin 校验、不进 trace parity。）

### 1A-1 Provider 解析与 parity 审计

- 公式（v 为 m/s 子步结束速度，Δs 为 m，I 为 mL/km）：

  ```text
  E_i [mL] = 32.5 × exp(0.036 × v_i) × Δs_i / 1000
  I = 1000 × ΣE_i / ΣΔs_i = 32.5 × Σ w_i exp(0.036 × v_i)，w_i = Δs_i/ΣΔs_i
  ```

  与 `src/eco_planner/envs/domain/energy.py` 的在线实现（`3.25·exp(0.01·v_kmh)·Δs_km/100×1000`）代数等价；`32.5`、`0.036` 为折叠常数。v 取 `speed_mps[1:]` 即**子步结束速度**；Δs 为相邻实际 vehicle-center 欧氏位移；正式窗口只含 executed 数组，warmup 数组独立存储（E-060 中 warmup=0 步）。
- Parity 链：raw trace（速度/距离/逐步 mL）→ 按上式独立重算逐步 mL（容差 rtol=1e-10、atol=1e-12，与 guidance 诊断一致）→ 求和比对 summary `energy.total_ml`/`energy.ml_per_km`/`distance_m`；另核验 (a) trace 速度与 `executed_states` 速度范数逐位一致、(b) trace 距离与相邻 center 位移逐位一致（端点与几何定义核验）、(c) summary `speed_mps.{min,mean,max}`、`stopped_fraction` 与 trace 一致、(d) warmup 步数与 summary 一致。
- 零距离语义：`distance_m == 0` 时 total `fuel_ml = 0.0`（合法）而 `ml_per_km` 为 **undefined（None）**，不得作为有效零进入统计；配对分解只对双方 completed 且 I 有定义的 pair 进行，unavailable 原因显式记录。
- **任何 mismatch 即 Gate P1 裁定 1 失败：先定位数据/口径错误并暂停物理解读**，不以静默降级继续。

### 1A-2 合成反事实（直接调用 `MetaDriveFuelProxyProvider.measure`，非重推导公式）

冻结 profile 集合（dt=0.1 s，N=100 步，v0=10 m/s，幅值 a=3 m/s，静止尾巴 50 步）：

| Profile | 定义 | 检验命题 |
| --- | --- | --- |
| `constant` | v_i ≡ v0，Δs_i = v_i·dt | 基线 |
| `fluctuating` | v_i = v0 + a·[−1,0,+1,0][i mod 4]，Δs_i = v_i·dt | 与 constant 同 time-weighted 平均速度 |
| `reordered` | `fluctuating` 逆序 | 与 `fluctuating` 距离加权速度分布相同、片段排序/加减速时序不同 → I 必须不变 |
| `constant_with_stationary_tail` | constant + 50 步 v=0/Δs=0 | total 与 I 不变、time-weighted mean speed 改变 → 停车时间对 proxy 不可见 |
| `stationary` | 全部 v=0/Δs=0 | total=0.0、I=None（undefined 语义） |

预期（结构性，非实测）：reorder 不变性、fluctuation 使 I 严格上升（凸性 + 距离加权）、静止尾巴不改变 total/I、全静止 undefined。这些是**数学性质**，用于界定 proxy 能/不能感知什么；与 1B 实测统计分开陈述。

### 1A-3 结构性局限（结构推论，与实测分开）

当前 proxy 的直接输入只有子步结束速度与位移：无显式加速度、制动/再生损耗、怠速油耗、坡度或动力系统状态项。加减速只能通过所形成的距离加权速度分布间接改变 I；排序反事实证明加减速**时序**对 I 完全不可见。

### 1B E-060 final A2−A1（主）与 A1−A0（对照）严格速度归因

- 每 episode 量：I、total mL、distance、route completion、arrive_dest/terminated/truncated/terminal_reason、stopped fraction、普通（time-weighted）mean speed、**距离加权速度水平 μ_d = Σw_i v_i**、速度分位数（0/0.1/0.25/0.5/0.75/0.9/1）、加速度与 jerk 描述统计（由逐子步结束速度范数差分 ÷ 0.1 s 得出；**局限**：速度范数的差分不是带符号纵向动力学，不含坡度/制动损耗信息，仅作描述性关联，不得解释为物理能耗效应）。
- **精确数学分解**（恒等式，非一阶近似；w_i=Δs_i/ΣΔs_i，μ_d=Σw_i v_i）：

  ```text
  ln I = ln 32.5 + 0.036·μ_d + ln Σ w_i·exp[0.036·(v_i−μ_d)]
  Δln I = 0.036·Δμ_d（距离加权速度水平贡献） + Δln shape（速度分布形状贡献）
  ```

  记账恒等式精确成立（浮点残差需 ≤ 1e-12 量级；若残差显著，优先查代码/采样/窗口，不得解释为物理效应）。不得声称因果中介比例。
- 配对：按 (scenario, map_sequence, seed, noise_seed) 匹配，difference = comparison − reference；每 training seed 单独报告点估计 + **per-seed scenario bootstrap CI**（0.95 / 10000 resamples / seed 0 / percentile / matched_scenario_delta 单位 / jointly-completed 且 I 有定义的交集）；**不重采样 training seeds，不产生跨 seed 总体区间**。A1−A0 对照中 A0 无 training seed，按 A1 seed 逐一配对共享同一 A0 episodes。
- 稳健性：S/SC family 分层均值；leave-one-scenario-out 均值范围与符号翻转计数（不重采样统计）；极端回合检查（非 completed、截断、未到达、最低 distance / route_completion pairs 的 delta 单列）。
- 加速度/jerk 描述性关联：per-pair Δaccel-RMS、Δjerk-RMS 与 Δln shape / Δln I 的 Spearman（per seed + 合并），仅描述性，配合排序反事实界定解释边界。

### Gate P1 三项裁定（运行后回填）

1. trace→公式→summary 是否一致；
2. E-060 改善由速度水平与分布形状各解释多少、是否存在超出公式输入的独立信息；
3. 当前 proxy 能否支持「独立于降速的真实节能」表述，及适用场景/速度范围/统计边界。

## 配置与命令

- 冻结配置：`configs/experiments/energy-proxy/analysis.yaml`（本记录预注册节的机器可读形态）。
- 复现命令：`just exp energy proxy run --source-dir outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison --output-dir outputs/studies/energy-proxy/e-061-issue113-task-1-speed-attribution`
- 运行身份（正式运行，2026-10-15）：commit `7a3ca73`（`7a3ca7366e9f699552ce73e2f4c305a862dbb754`）、branch `main`、dirty workspace——本分析的全部新代码即未提交 diff（`src/eco_planner/experiments/energy_proxy/`、`tests/analysis/test_energy_proxy_analysis.py`、`configs/experiments/energy-proxy/analysis.yaml`、本记录）+ 修改 `scripts/experiments.py`（CLI 注册）。诊断类研究允许 dirty workspace，但运行产出的代码身份即上述 diff 本身；summary.json 的 `provenance` 节记录了源 manifest 的 24 runs + baseline 全部 policy hash / evaluation dir / reward profile。

### 计划偏离（运行前发生，显式登记）

冻结计划 1A-1 写明「经 `load_comparison` 全量校验」。实现时发现**版本偏斜**：E-057/E-058/E-060 的历史评测 `summary.json` 早于 2026-10-10 commit `75d4326` 加入 `JobSummary` 的 `policy_action` provenance 字段，当前 `load_comparison` 无法验证历史 summary（E-060 时点的全量校验在其自身 pin 下运行，非本诊断可复现）。该偏离在正式运行前已确认并改为 `_load_source` 定向 raw-JSON 守卫：对同一守卫范围（24 runs + baseline）做 matched scenario 集 / noise seed / sampler ddim5 stochasticity=0 / runtime seed / per-seed initial policy hash / frozen planner hash / probe / noise seeds 交叉核对 + policy hash 与 checkpoint label 匹配 + baseline 无 policy checkpoint。守卫范围与冻结计划一致，仅校验实现路径不同；偏斜事实以 `model_skew_note` 记入 provenance。

## 结果（2026-10-15 正式运行回填）

全部输出见 `outputs/studies/energy-proxy/e-061-issue113-task-1-speed-attribution/`（`summary.json` / `report.md` / `analysis.json` / `figures/`）。

### Parity 审计（裁定 1 输入）

- 范围：final label 的 112 episodes（A0 16 + A1 3×16 + A2 3×16），全部 audited、无 unavailable；warmup 步数全部为 0。
- 公式重算 vs 存储 `executed_fuel_proxy_step_energy_ml`：最大逐步相对误差 **6.98e-16**（容差 rtol=1e-10/atol=1e-12 内）。
- 恒等式残差（ln I 分解记账）：最大 per-episode **4.44e-16**。
- 交叉核验全部通过：trace 速度 = `executed_states[:,3:5]` 范数（误差 0）、trace 距离 = 相邻 center 位移（误差 0）、summary speed/stopped/distance 一致、逐步 mL 求和与 `energy.total_ml`/`ml_per_km` 一致。
- **跨实验交叉验证**：本诊断独立重算的 per-seed ΔI（−0.573 / −0.619 / −0.691 mL/km）与 E-060 登记 F2 值（其记录第 17/216 行）显示精度逐位一致。

### 合成反事实（直接调用 provider `measure`）

| Profile | time-mean m/s | total mL | I mL/km | ln shape |
| --- | ---: | ---: | ---: | ---: |
| constant | 10 | 4.6583 | 46.583 | ≈0（−1.11e-16） |
| fluctuating（同 time-mean） | 10 | 4.7475 | 47.475 | +0.00277 |
| reordered | 10 | 4.7475 | 47.475 | +0.00277 |
| constant_with_stationary_tail | 6.667 | 4.6583 | 46.583 | 0 |
| stationary | 0 | 0.0 | undefined | undefined |

四项冻结不变量全部确认：reorder 不变性（与 fluctuating 完全相同）、fluctuation 敏感性（同 time-mean 下 I 严格上升 +1.91%，说明**普通平均速度不足以决定 proxy**）、静止尾巴不变性（total/I 不变、仅 time-based 统计改变）、全静止 undefined 语义（total=0.0 合法、I=None 非 effective zero）。

### 主对比 A2−A1（final；difference = comparison − reference；noise seed 760025；16/16 pairs per seed；最大配对恒等式残差 7.41e-16）

| Seed | ΔI mL/km [CI] | Δln I [CI] | level 0.036·Δμ_d [CI] | shape Δln [CI] | level share | Δμ_d m/s | Δ time-mean speed m/s |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 0 | −0.573 [−0.690, −0.457] | −0.012047 [−0.01431, −0.00978] | −0.012058 [−0.014385, −0.009729] | +1.07e-05 [−7.25e-05, +9.28e-05] | 100.09% | −0.335 | −0.338 |
| 1 | −0.619 [−0.736, −0.501] | −0.012938 [−0.01515, −0.010678] | −0.012819 [−0.015103, −0.010486] | −1.19e-04 [−2.08e-04, −3.09e-05] | 99.08% | −0.356 | −0.341 |
| 2 | −0.691 [−0.828, −0.555] | −0.014412 [−0.01700, −0.011833] | −0.014262 [−0.01692, −0.011597] | −1.51e-04 [−2.44e-04, −5.66e-05] | 98.96% | −0.396 | −0.378 |

- Δln I 与 level 项的全部 CI 不含 0；shape 项 CI 量级 ≤1e-4，seed 0 含 0，seed 1/2 微负。
- **距离加权速度水平项解释每 seed Δln I 的 98.96%–100.09%**；shape 项 ≤1.04%，seed 0 甚至与改善方向相反（share −0.09%）。
- LOO：ln_intensity 与 level 项 0 符号翻转；shape 项 seed 0 有 4 次翻转（与 ≈0 一致）。Family 分层：S family 效应约为 SC 的 2 倍（如 seed 2 ln_intensity：S −0.01959 vs SC −0.00924），符号一致。
- Δμ_d 与 Δ time-mean speed 几乎相同（每 episode 速度分布近均匀），两者不可区分不是本诊断的限制而是 E-060 轨迹的实际性质。
- 加速度/jerk 描述性关联（Spearman，per-pair Δ）：Δaccel-RMS vs Δshape = 0.77–0.89（波动更大的场景形状项略高）；Δaccel-RMS vs Δln I = −0.38 ~ −0.64。结合排序反事实，这只能读作"加速度波动通过距离加权速度分布间接关联 I"，**加减速时序对 proxy 完全不可见**。
- 极端回合：无 non-completed（trace/summary 全部完整可读）；18 个 truncated/未到达 episode（trace 完整、I 有定义，**正常参与**配对统计，仅 termination 非 arrive_dest；全列于 summary.json）；最低 distance（s21）与最低 route completion（sc23）pairs 的 delta 方向与总体一致。

### 对照对比 A1−A0（final；A0 按 A1 seed 共享配对；最大残差 5.83e-16）

ΔI = +0.105 / +0.434 / +0.519 mL/km（Δln I = +0.00222 / +0.00900 / +0.01074；level 项 share ≈ 99%–101%）：A1 相对 frozen planner 的 intensity 上移同样几乎完全由速度水平载带，与 E-060 记录的 A1 learned 分量（速度/强度上移）一致。12 个 truncated/未到达 episode 单列。

### Gate P1 三项裁定

1. **trace→公式→summary 一致：通过。** 112/112 episodes，逐步公式误差 ≤6.98e-16、恒等式残差 ≤7.41e-16（均在浮点级，1e-12 界内），全部交叉核验通过；且重算 ΔI 与 E-060 登记 F2 值逐位一致。
2. **速度水平 vs 分布形状：水平项解释 98.96%–100.09%，形状项 ≤1.04% 且符号不稳定（seed 0 CI 含 0、LOO 有翻转）。无超出公式输入的独立信息**——加速度/jerk 仅通过距离加权速度分布间接关联，时序不可见（排序反事实证明）。
3. **当前 proxy 不能支持「独立于降速的真实节能」表述。** E-060 A2−A1 的 intensity 改善 ≥99% 由距离加权速度水平下降（Δμ_d = −0.335 ~ −0.396 m/s）载带；形状贡献 ≤1% 且不稳健；proxy 结构性缺失加速度、制动/再生、怠速、坡度与动力系统状态项。任何「非降速机制节能」声明在本 proxy 下**不可验证**（而非证伪）；物理性判断留给 Task 2 的 FASTSim 复核。

## 产物

- `outputs/studies/energy-proxy/e-061-issue113-task-1-speed-attribution/`：`summary.json`（provenance、逐 episode parity、配对归因与 bootstrap CI）、`report.md`（人读报告）、`analysis.json`（逐 episode 完整证据）、`figures/primary-ln-intensity-decomposition.{png,svg}`、`figures/primary-scenario-ln-intensity-delta.{png,svg}`。

## 结论边界

本诊断为只读离线重算：不重训、不重跑仿真、不修改 E-060 Gate F 历史裁定。结论只覆盖两方面：(a) MetaDrive fuel proxy 的**数学结构**——输入仅为子步结束速度与位移，加减速时序、停车时长、怠速/制动/坡度不可见；(b) E-060 final matched evidence（S/SC held-out seeds 16–23、v 量级 ≈10 m/s、kinematic waypoint boundary、A2−A1 final）的**速度归因**——intensity 改善 ≥99% 由距离加权速度水平载带。

不证明：真实车辆能耗、FASTSim 一致性、其他速度域/场景域的外推、λ 调度的物理节能机制。A1−A0 对照显示的 intensity 上移同样只是 proxy 口径的事实。「独立于降速的真实节能」在当前 proxy 下不可验证（非证伪），物理复核属 Task 2。
