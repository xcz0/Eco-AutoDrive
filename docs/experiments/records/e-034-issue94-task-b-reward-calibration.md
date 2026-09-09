# E-034 Issue #94 Task B：Reward 动态范围校准与固定批次重诊断

[返回实验索引](../README.md) · [源实验 E-033](e-033-issue94-task-a-lambda-identifiability.md)

**日期 / 类型 / 目的**：2026-09-09 / 正式离线固定批次诊断 / 完成 Task B 的分布审计、
一次预定校准及 Task A 重跑，提供连续证据；是否进入 Task C 由用户决定。

**代码 / 环境**：基线 `a3972366d6c971879d439a2e717fc22539e055e2` 加本次未提交改动；
`runtime_metadata.json` 和 `tracked_diff.patch` 记录运行时仓库状态，`source/` 保存实际
执行的 Task B 模块、CLI、Task A diagnostics 和评分函数。Windows、Python 3.10.20、
PyTorch 2.12.1+cu126、CUDA:0。actor backward 与 E-033 一样使用 float32，无 rollout
autocast；源 rollout 的 CUDA BF16 mixed precision 与其运行环境另外保留于 source runtime。

**数据 / 模型 / 协议**：直接复用本机 E-033 的 seed-0 initial checkpoint、128 个
transitions（S/SC × map seeds 0–7，16 episodes × 8 steps）、原 policy contexts、动作、
old log-prob、critic current/next value 和 episode boundary。源 batch 的 map、planner
noise、policy-action seed streams 不变；本实验没有重新采样，也没有实例化 frozen planner
或 simulator。每组 `λ={0,1,2,4,8,16}`，共享 TTC/Progress/Comfort/Speed 权重 5/5/2/4；
R0 分母 16，正 λ 分母 16+λ。只反传 PPO actor objective，optimizer/scheduler steps 为 0。

**配置 / 命令**：

```powershell
just reward-calibration `
  --source-dir outputs/studies/scalar-reward/e-033-issue94-task-a-identifiability `
  --output-dir outputs/studies/scalar-reward/e-034-issue94-task-b-calibration
```

使用 `configs/experiments/scalar_reward/calibration.yaml` 的显式目标分数 0.6。原组与校准组
各自保存完整 resolved config；没有修改全局 reward 默认值，没有按诊断结果再调整尺度。

## 原始量与 Comfort 失活归因

| 原始量（运动学量取绝对值） | Mean ± std | P25 / P50 / P75 | P90 / P95 | Min / max |
| --- | --- | --- | --- | --- |
| route progress delta (m) | 1.065527 ± 0.059000 | 1.038318 / 1.068809 / 1.097400 | 1.126407 / 1.160048 | 0.872080 / 1.242721 |
| longitudinal acceleration (m/s²) | 18.706099 ± 33.551455 | 2.990763 / 5.820568 / 11.832105 | 104.358638 / 107.471517 | 0.132649 / 110.950523 |
| lateral acceleration (m/s²) | 3.712622 ± 2.815505 | 1.598404 / 3.174968 / 5.082304 | 8.107873 / 9.015142 | 0.138831 / 13.566324 |
| jerk (m/s³) | 371.342230 ± 410.045662 | 97.379440 / 156.386818 / 564.533447 | 1079.411328 / 1100.185645 | 16.080027 / 1205.732178 |
| yaw rate (rad/s) | 0.024423 ± 0.021263 | 0.008632 / 0.020680 / 0.034109 | 0.049831 / 0.061780 | 0.000003 / 0.126269 |

Comfort 子项公式为 `clip(1-max(0,x-limit)/limit,0,1)`，因此超出 limit 开始扣分，
达到两倍 limit 才归零；不能把超限率与零分率混用。

| 子项 | 原 limit | 原 limit 超限率 | 原零分率 | 校准后零分率 |
| --- | ---: | ---: | ---: | ---: |
| longitudinal acceleration | 3 m/s² | 75.00% | 48.44% | 35.16% |
| lateral acceleration | 3 m/s² | 53.91% | 19.53% | 19.53% |
| jerk | 5 m/s³ | 100% | 100% | 36.72% |
| yaw rate | 0.5 rad/s | 0% | 0% | 0% |

Jerk 是每个 transition 都将 Comfort 压到零的直接项；纵向和横向加速度还有并列零分。
校准后纵向/横向/jerk/yaw 子项成为最小值的比例分别为 57.81%/42.97%/55.47%/10.16%，
包括并列项，因此这些比例不要求和为 100%。Yaw 的并列来自整体满分样本，不是新的失活项。

按 planning-cycle index 0–7，jerk 中位数约为
`[1072.54,1084.63,96.53,105.07,143.94,126.38,120.23,137.92] m/s³`。
前两步明显受启动瞬态影响，但后续步骤也远超原阈值，不能只用启动效应解释全部失活。

代码核对显示 acceleration 来自连续实际 velocity 的差分除以 0.1 s，jerk 来自连续二维
acceleration 差的模长再除以 0.1 s；extractor 保存前一速度和加速度，后者在 reset 时为零。
本批次为 no-traffic，`MetaDriveEnvSlot._warmup()` 直接返回，没有执行 stationary traffic
warmup。运动学 executor 原样写入 waypoint 的差分速度，启动时的大速度/加速度跳变与
前两步尖峰一致；现有证据没有显示单位换算错误，也不支持把它解释为高保真车辆动力学。
本次不修改执行轨迹、reset、差分公式或采样频率，也不平滑或删除启动样本。

## 一次预定校准与动态范围

Progress：`P50(positive delta)/0.6 = 1.7813475926717124 m`。
Comfort：有零分的子项使用 `max(old_limit,P50(abs(metric))/1.4)`；未失活项保持原值。

| 配置 | 原值 | 实际校准值 |
| --- | ---: | ---: |
| progress.full_score_delta_m | 1.0 | 1.7813475926717124 |
| comfort.longitudinal_acceleration_limit_mps2 | 3.0 | 4.157548461641585 |
| comfort.lateral_acceleration_limit_mps2 | 3.0 | 3.0 |
| comfort.jerk_limit_mps3 | 5.0 | 111.70486995152065 |
| comfort.yaw_rate_limit_radps | 0.5 | 0.5 |
| energy.reference_ml_per_km | 50.0 | 50.0 |

横向加速度虽有零分，但按预定规则计算的尺度低于原值，因此不调整。该规则只保证被实际
抬高尺度的子项典型分数为 0.6，不保证四项最小值的中位数也为 0.6。

| 分数 | 原 mean ± std | 校准后 mean ± std | 原→新零分率 | 原→新满分率 |
| --- | --- | --- | --- | --- |
| Progress | 0.995046 ± 0.017057 | 0.598158 ± 0.033121 | 0% → 0% | 86.72% → 0% |
| Comfort | 0 ± 0 | 0.291612 ± 0.392095 | 100% → 55.47% | 0% → 10.16% |

校准后 Progress 的 P25/P50/P75 为 0.582884/0.600000/0.616050，范围
0.489562–0.697629；恢复了中间动态区。Comfort 恢复了方差，但仍有 71/128 个零分样本，
中位数仍为零。Energy 保持原来的 0.385150 ± 0.007761，既有有限方差不支持修改其
normalization。TTC/Speed/Safety 的饱和及全部共享权重保持原样。

**解释边界**：新 Comfort 是本批次运动学执行分布中的相对平顺性评分，尤其新 jerk
尺度远高于原物理参考值；分数增加不表示车辆变舒适。原量、原 limit 超限率和启动尖峰
全部保留在 audit 中，不能用重标定掩盖实际运动失效。

## Task A 重跑：动态范围恢复未带来明显方向分离

| 校准 | λ 对比 | Pearson | Spearman | Sign flip | Actor-head cosine | Norm ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 原配置 | 0→8 | 0.999983210 | 0.987117439 | 0 | 0.999991174 | 1.00417095 |
| 原配置 | 0→16 | 0.999951781 | 0.978267743 | 0 | 0.999974801 | 1.00705915 |
| 校准后 | 0→8 | 0.999906736 | 0.999587988 | 0 | 0.999990753 | 0.982623477 |
| 校准后 | 0→16 | 0.999738750 | 0.999238921 | 0 | 0.999973469 | 0.970824007 |

λ0→16 的 normalized-advantage delta RMSE 从约 0.009782 增至 0.022769，最大绝对值
从约 0.020767 增至 0.063739；但无 sign flip，actor-head cosine 基本保持原水平。
Head norm 的相对变化从 +0.706% 变为 −2.918%；这表示梯度幅度有变化，不是方向明显分离。
Spearman 反而更接近 1，不能仅凭 Pearson 下降就宣称全面改善。

校准后 λ0→16 的 lateral/longitudinal head cosine 分别为 0.999965708/0.999977009，
norm ratio 为 0.976613213/0.970086557。所有 λ 的 shared-trunk norm 仍严格为零，
来自 initial actor-head zero-init；其 cosine/norm ratio 为 `null`，不作为不可辨识证据。

**连续证据结论**：本次预定校准恢复了 Progress/Comfort 的评分动态范围，但该固定
initial-policy batch 上的 λ 差异仍主要表现为幅度变化，尚未观察到明显的 normalized
advantage 符号或 actor-gradient 方向分离。该结果说明恢复分量动态范围本身不足以解决
当前 batch 的方向近似一致问题；它不证明 optimizer 存在问题，也不证明任何训练后的
behavioral effect。没有建立或事后调整通过阈值，是否进入 Task C 由用户判断。

## 验证与产物

原配置从原始量重建的 component/reward 通过数值容差核验；原配置全部 Task A 数组与
E-033 比较通过 `rtol=1e-5, atol=1e-6`，最大绝对差为 `8.3446503e-7`。
initial policy hash 与 E-033 一致，两组 backward 后策略未改变，optimizer steps 为零。

`just test-target tests/training/test_reward_calibration.py tests/training/test_reward.py
tests/training/test_lambda_identifiability.py tests/training/test_ppo.py` 首轮 47/48 通过；
唯一失败暴露 loader 核对时误将 training log-prob `[T]` 与 audit `[T,1]` 直接比较。
按既有契约仅在该字段核对时去掉末维后，Task B 12/12 通过；直接相关既有测试 36/36
已通过。另增加报告来源回归检查，确认 Task B 子报告明确标注复用批次，不继承 Task A
模板的“新采集”措辞。两份 Markdown 子报告从已完成的 summary 重新生成，诊断数组未重算；
`source/` 保留运行时源码快照，报告来源参数属于运行后的展示修正。
最终 Task B 测试 13/13 通过，连同已通过的既有 36 项共覆盖 49 项；最终 Ruff 通过，
修改模块 Pyright 0 errors，`git diff --check` 通过。未改 simulator 行为，无额外仿真运行。

产物目录：`outputs/studies/scalar-reward/e-034-issue94-task-b-calibration/`（git ignored）。

- `report.md` / `summary.json`：配对诊断、实际校准配置及原配置复现误差。
- `audit.json` / `audit.npz` / `sample_index.json`：总体、scenario、cycle 和逐 transition
  的原量、原物理参考超限与两组子项评分。
- `original/` / `calibrated/`：各组完整 Task A summary、diagnostics、report、resolved config。
- `calibration_config.yaml` / `runtime_metadata.json` / `tracked_diff.patch` / `source/`：
  规则和运行来源；原 training batch/checkpoint 仍从 E-033 目录读取，不冒充新采集产物。

Task B 实现、audit、一次校准和诊断重跑已完成；未自动裁定 Task C，未执行 C–E，
未选择最终 λ，未更新或关闭 GitHub Issue #94。
