# 当前研究结论

本页按研究问题提炼已读取 experiment records 支持的结论，不重新分析 artifact，不增强原结论。
引用完整记录文件名以区分重复 E-ID；运行事实和 provenance 仍由原 record/artifact 拥有。
待验证方法见 [Hypotheses](hypotheses/ablation-plan.md)，登记方式见 [实验记录](../experiments/README.md)。

## 历史条件与当前规范

下文 PPO 与 fixed-batch 历史证据来自旧 training `k=1 / 0.1 s` 条件；普通 evaluation 为
`k=5 / 0.5 s`，人工干预的 prefix 另行注明。[ADR 0039](../adr/0039-unify-closed-loop-cadence.md)
后来接受统一 cadence，同时改变 transition reward 归约和 `gamma` 对应的物理时间视界。
旧结果不能改称“当前统一 cadence 下已重新验证”；即使历史评测也是 k=5，也不能据此推断
改为 k=5 训练后的学习方向。#83 的新增状态见[本页末节](#canonical-cadence-transfer-的证据边界)。

## 能耗读数能否代表实际执行

**结论。** 在 E-019 的 MetaDrive 0.4.3 运动学 waypoint execution 中，native step/episode energy
恒零，不能与从实际位移和速度重算的 execution-trace fuel proxy 互换。负 guidance 降低总能耗时，
部分场景同时少走、低 progress 或 out-of-road，不能解释为节能收益。

**条件与证据。** [E-019：native/proxy 对照](../experiments/records/e-019-metadrive-native-energy-proxy-comparison.md)
使用 6 个固定场景、planner noise seed 0、DDIM5、BF16、10 Hz 执行与 2 Hz 重规划；
baseline、zero 与正负 longitudinal guidance 对照中，zero 与 baseline 逐值相同。
记录保存运行时 dirty provenance，保留其原分类与限制，不追认为符合现行 clean-commit 要求的正式结果。

**限制与未决问题。** 这是特定 phase boundary 下的测量结论，不证明真实车辆能耗、动力学可跟踪性、
FASTSim 一致性或跨 seed 节能。Proxy 结论能否在更高保真能耗模型下保持，仍需独立验证。

## PPO 稳定候选能支持到哪里

**结论。** Reset 修复后，E-026 式退化不再复现，旧“PPO update 强度过大”归因被修正。
E-028 找到受限条件下的稳定候选；E-030 在 PPO broadcasting 修复后支持它对 R0/Rλ=1 的短程迁移。
稳定性和机械有效性均不等于 learned energy improvement。

**条件与证据。**

- [E-028：reset 修复与稳定性搜索](../experiments/records/e-028-issue76-ppo-stability-search.md)：
  `metadrive_builtin_v1`、no-traffic S/SC、RTX A4000/BF16、training seeds 0/1/2；
  config-0001（batch=minibatch=128、epochs=1、lr=1.6301e-5）为唯一通过 100 updates
  及 held-out 评测的候选。config-0003 在 50 updates 稳定，100 updates 的 seed 0 出现 Beta 边界塌缩。
- [E-030：标准 PPO 目标下 transfer gate](../experiments/records/e-030-issue81-ppo-transfer-gate.md)：
  标准 PPO 目标修复后，runtime seed 0/replay 0、S/SC map seeds 0–7、R0 与旧 Rλ=1 profile
  各 20 updates，机械有效且短程稳定；两臂几乎无行为差异。其 batch 为 16×8，
  E-028 候选为 8×16，不能宣称逐值可比。修复前的 cross-paired PPO 产物不作为 transfer 证据。

E-030 引用的旧共同资产只能沿
[固定 revision 的原 README](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/experiments/README.md#共同资产)
解释，不能用当前配置补造历史来源。

**限制与未决问题。** E-028 的搜索不能建立超参数因果机制；其条件含运行时工作流修复，
原始 provenance 保留。E-030 不支持超过 20 updates 的稳定性、reward 优劣或节能。
两者不能自动外推到 traffic、新 observation、encoder 微调或 canonical k=5 training。

## Objective 可辨识性是否足以学出节能

**结论。** 原 reward/batch 中，增加 λ 和恢复 Progress/Comfort 动态范围没有形成明显 actor 方向分离；
更换为 calibrated efficiency-band 后，Energy-only endpoint 在该 fixed batch 上可辨识。
这是表示与样本条件下的诊断结果，尚不等于训练后的有效节能方向。

**条件与证据。** 证据使用 seed-0 initial policy、no-traffic S/SC map seeds 0–7、
128-transition update-0 batch、旧 k=1 cadence；E-034 复用源 batch，E-035/E-038 重采，
按各记录的同协议/同 hash 边界解释，不笼统称为所有实验逐位复用同一 batch。

- [E-033：λ 可辨识性](../experiments/records/e-033-issue94-task-a-lambda-identifiability.md) 与
  [E-034：分量校准](../experiments/records/e-034-issue94-task-b-reward-calibration.md)：
  λ0→16 均无 advantage sign flip，actor-head cosine 约 0.99997；校准恢复分量方差，
  不足以形成明显方向分离。
- [E-035：objective decomposition](../experiments/records/e-035-issue94-task-c-objective-decomposition.md)
  的 Gate C FAILED 保持不变；其“共享 critic 主导”解释随后被
  [E-036：critic/GAE common-term ablation](../experiments/records/e-036-issue94-task-c4-critic-gae-common-term-ablation.md)
  修正。移除 critic、改用 discounted return 后 endpoint 仍近共线；支持
  `reward_batch_collinearity`，不能继续把 E-035 初始解释当成当前结论。
- [E-038：efficiency-band 表示](../experiments/records/e-038-issue94-task-e-energy-representation.md)：
  Gate E PASSED，z-form head cosine 0.9765、sign-flip 25.8%，raw 形式也可分；
  λ64/λ256 达 endpoint angular separation 的约 74%/92%。该非仿射表示改变排序与间距，
  不靠关闭 normalization 或乘大常数通过。

**限制与未决问题。** Energy-only 是诊断 endpoint，不是已选训练 objective；λ16 仅达 41.5%，
不能推断小 λ 足够。E-038 未重跑 C4，原记录说明其范围与机制限制；不把该缺口写成已验证。
初始 actor-head 零初始化导致 trunk gradient 为零，其 cosine 为 undefined，不能算方向一致。
新 cadence transfer 与闭环收益是不同待核验层级。

## 有效更新是否保证 objective 方向正确

**结论。** E-039 得到机械上可测的更新区域；E-040 的 reward 差异确实传导为可复现闭环行为差异，
但 Rstress 更快、单位里程 proxy 能耗更高，Gate G 的 c4 失败。该 negative result 不能改写为
“positive control 已通过”。

**条件与证据。**

- [E-039：effective-update region](../experiments/records/e-039-issue94-task-f-effective-update-region.md)：
  旧 k=1 training、未校准 R0、seed 0、BF16、batch=minibatch=128、50 updates；
  lr=1.5e-4、epochs=1、max-gradient-norm=0.5 通过 Gate F。
  Seeded-MC post-update KL 中位约 1.16e-6，held-out speed 变化约 0.14%，均属边际过线；
  不证明有意义的节能。单 minibatch 下 pre-update `kl_approx≈0` 存在测量伪影，
  不能只靠该数值归因 under-update；记录保留修正测量后的证据。
- [E-040：objective positive control](../experiments/records/e-040-issue94-task-g-objective-positive-control.md)：
  calibrated R0 vs calibrated-band λ64，training seeds 0/1 各 50 updates，
  旧 k=1 training / k=5 deterministic mean evaluation、no-traffic held-out 对照；
  Rstress−R0 speed 为 +1.06%/+1.41%，energy intensity 为 +0.41%/+0.53%，均超过原噪声界。
  机械健康、双 seed 分离成立，失败项是 objective 方向。

**限制与未决问题。** E-039 为未校准 R0，E-040 为校准 R0，单臂绝对移动不能直接比较。
E-040 自身没有证明 band 表示错误或 λ64 必然反向；后续执行干预提供的解释见下节。
这两项结果都不支持 canonical cadence 下已训练验证或真实车辆节能。

## Guidance 与 execution prefix 如何共同决定方向

**结论。** 已有因果干预支持：guidance 的时间响应与只执行首点的 receding horizon 相互作用，
能使完整计划的正响应与执行响应异号。Frozen learned policies 的 prefix crossover 把该解释
连接到 E-040，不能只用“policy 未学到 objective”解释方向问题。

**三个证据层级。**

1. **人工全幅 guidance。**
   [E-043：execution horizon](../experiments/records/e-043-issue98-task-a-execution-horizon.md)
   在 no-traffic S/SC seeds 0–7、3 noise repeats、2 s 窗口中，只改 prefix k∈{1,2,5,10,20}：
   executed speed 从 k=1 多数负方向变为 k≥2 正方向。
   [E-045：replanning deferral](../experiments/records/e-045-issue98-task-c-replanning-deferral.md)
   在 k=1 连续重规划下，13/16 场景出现 repeated deferral，正响应越过执行首点，
   zero-crossing 稳定在 0.2 s；另外 3 个场景保留，不挑 seed。
2. **由 learned mean 构造的常量干预。**
   [E-044：lon/lat decomposition](../experiments/records/e-044-issue98-task-b-guidance-decomposition.md)
   在 no-traffic held-out S/SC seeds 16–23、300 步、k=1、BF16 条件下，energy 双 seed 为
   longitudinal-dominated；speed 的 seed 0 因 lon/joint=0.72 低于 0.75 阈值，
   overall 保留 `mixed-across-seeds`。不支持横纵耦合为主因；常量注入不是 state-dependent policy。
3. **Frozen learned policy 与 same-state response。**
   [E-047：execution bridge](../experiments/records/e-047-issue98-frozen-policy-execution-bridge.md)
   使用 E-040 四个 final checkpoints、training seeds 0/1、no-traffic S/SC seeds 16–23、
   300 步、BF16、DDIM5、runtime noise seed 760025、Beta mean，只改 prefix。
   两 seed 的 Rstress−R0 speed/energy 均由 k=1 负转为 k=2/5 正；k=5 与 E-040 方向和量级一致，
   不宣称逐值重现。Same-state 双 policy 审计也呈 0.1 s 中位非正、0.2/0.5 s 为正的
   `local_temporal_bridge`，将全幅干预的解释延伸到 learned policy 的局部工作点。

**失败与限制。** E-043 的 k=2 有 73/1200 off-route，机器 gate 为 `safety_or_proxy_failure`，
不能写成整项安全通过；E-044 的 SC 有 49/64 off-route，限制全里程解释；
E-047 有 25/192 unsafe（全部 OOR、零 collision），同一 seed/prefix 两臂安全计数相同，k=5 无 OOR。
这些样本未删除；E-043/E-047 明确记录 dirty 运行，仅作诊断证据。
Proxy energy 随速度变化，两者不是独立证据。E-047 的 same-state contexts 来自 k=5 held-out
分布，不能外推 k=1 training 状态分布、统一 cadence 后的重新训练或真实动力学。
Training-budget sweep 没有运行；其非必要性判断不等于已验证所有学习动态。

## Critic 和 GAE 是否可以完全排除

**结论。** E-046 支持 critic 在部分后期 checkpoint 上产生有界、未重排 advantage 的 actor 方向旋转；
不是 E-040 反向行为的首要解释，但不能认证“critic 完全无关”。这与 E-036 的初始 fixed-batch
归因适用条件不同，不应合并成一个普遍结论。

**条件与证据。** [E-046：training adequacy / critic attribution](../experiments/records/e-046-issue98-task-d-training-adequacy.md)
复用 E-040 的 4 runs×50 updates、旧 k=1 rollout、training seeds 0/1、no-traffic/BF16
产物，以 pre-update policy 做 CPU backward-only 对照，无 optimizer step。Within-arm standard
GAE vs V=0 中，20/200 checkpoints 触发 cosine<0.99；sign-flip≤1/128、Spearman≥0.9945，
因此预登记裁定保留 `critic_material_candidate`。

**限制与未决问题。** 该判定阈值敏感，中位 cosine≈0.9998，仅一个 group-checkpoint<0.95。
Cross-arm gradient 对照未 matched，只能描述；离线梯度不证明重新训练后的行为。
Critic 拟合、value target、GAE λ 与 normalization 的剩余影响需独立受控研究，
见 [reward/credit 假设](hypotheses/reward-objective.md)。

## Canonical cadence transfer 的证据边界

**已接受决定与待验收工作。** [ADR 0039](../adr/0039-unify-closed-loop-cadence.md) 的 cadence 决定
不能替代实验验收。[Issue #83](https://github.com/xcz0/Eco-AutoDrive/issues/83) 拥有 T1 可辨识性、
T2 有效更新、T3 positive-control bridge 以及后续 λ sweep 的任务与验收。

**2026-09-24 文档核对发现的状态差异。** #102 的固定 revision 盘点只读到 Task 0/Gate I；
此后 [#83 Task 1A 评论](https://github.com/xcz0/Eco-AutoDrive/issues/83#issuecomment-5806167952)
已报告 **Gate T1 PASSED**，运行 commit `eaafe5b`、记录提交 `70768a0`，并说明下一步为 Task 1B，
未启动正式训练或 sweep。不能继续把 T1 描述为“从未执行”。

但本次工作区 `21704d3` 未包含评论所指
`docs/experiments/records/e-048-issue83-task-1a-objective-identifiability-transfer.md`；
本地 Git 无 `70768a0` 对象，GitHub 按该 revision 读取也返回 `No commit found`。
因此本页仅登记 **Issue 已报告通过、原 record 尚未核到** 的来源差异，不复制评论数值成为
已核验 Finding，不补造 E-048 record，也不把缺记录推断成从未运行。
待原记录可读取后再提炼其条件与结论；T2/T3 的通过不由这条评论或旧 E-039/E-040/E-047 推出。
