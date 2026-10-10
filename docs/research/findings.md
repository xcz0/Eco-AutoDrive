# 当前研究结论

本页按研究问题提炼已读取 experiment records 支持的结论，不重新分析 artifact，不增强原结论。
引用完整记录文件名以区分重复 E-ID；运行事实和 provenance 仍由原 record/artifact 拥有。
待验证方法见 [Hypotheses](hypotheses/ablation-plan.md)，登记方式见 [实验记录](../experiments/README.md)。

## 历史条件与当前规范

E-048 之前的 PPO 与 fixed-batch 历史证据来自旧 training `k=1 / 0.1 s` 条件；普通 evaluation 为
`k=5 / 0.5 s`，人工干预的 prefix 另行注明。[ADR 0039](../adr/0039-unify-closed-loop-cadence.md)
后来接受统一 cadence，同时改变 transition reward 归约和 `gamma` 对应的物理时间视界。
旧结果不能改称“当前统一 cadence 下已重新验证”；即使历史评测也是 k=5，也不能据此推断
改为 k=5 训练后的学习方向。后续 canonical transfer、policy family 变更和 final comparison
分别见[本页后续结论](#canonical-cadence-transfer-的证据边界)。各自的 cadence、family、scheduler
history 与复用 provenance 不得混同；下文“未验证”均受所引用的历史条件限制。

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

**结论。** 冻结 reward representation 的可辨识性迁移到 k=5；原 PPO control 未通过有效更新
门限。Matched 归因把主要路径缩小到 critic/shared-trunk/global-clipping coupling，随后独立
control re-freeze 和复跑通过 T2。Canonical 训练与评测下的 positive control 方向与旧 k=1
训练结果相反。文档迁移只提炼本分支已有记录，不重跑实验或重裁历史 gate。

**条件与证据。** 以下使用 ADR 0040 之前的 policy family，no-traffic S/SC、冻结 planner/DDIM5：

- [E-048：objective identifiability transfer](../experiments/records/e-048-issue83-task-1a-objective-identifiability-transfer.md)
  在新的 k=5 initial-policy fixed batch 上复用 E-034/E-038 冻结表示，Gate T1 PASSED；
  仅 actor backward，无 optimizer/scheduler step，不建立 learned behavior。
- [E-049：effective-update transfer](../experiments/records/e-049-issue83-task-1b-effective-update-transfer.md)
  以 calibrated R0、seed 0、50 updates 测原 control，Gate T2 FAILED：median post-update
  seeded-MC KL 低于门限，c7 held-out 未执行；失败记录保持原裁定。
- [E-050：matched cadence attribution](../experiments/records/e-050-issue105-phase-a-cadence-attribution.md)
  只改 k=1/k=5，actor-head policy gradient 几乎不变，critic/total norm 增大、global clipping
  加强；支持 coupling 为 primary，reward/value scale 与 physical-time credit 为 upstream。
  [E-051：单机制 counterfactuals](../experiments/records/e-051-issue105-phase-b-counterfactuals.md)
  继续分离这条路径；没有单一 override-only 诊断臂恢复 k=1 effective-update region。
- [E-052：独立 control re-freeze](../experiments/records/e-052-issue105-phase-c-canonical-ppo-refreeze.md)
  按结果前冻结顺序选择 `time_consistent_value_scaled`，保留三个崩溃候选。
  [E-053：T2 re-entry](../experiments/records/e-053-issue106-task-1-gate-t2-reentry.md)
  在该 control 下独立复跑，Gate T2 PASSED（7/7），final hash 与 E-052 selected 一致。
  这是单 seed、50 updates 的机械有效性，KL floor 余量薄，不证明普遍最优或节能。
- [E-054：canonical positive-control bridge](../experiments/records/e-054-issue106-task-2-canonical-positive-control.md)
  比较 R0/λ64、seeds 0/1、50 updates，Gate T3 PASSED（6/6）。λ64 更慢且 proxy 强度更低，
  frozen-policy bridge 支持 canonical temporal mapping；不改写 E-040/E-047 的旧条件结果。
- [E-055：dose-response](../experiments/records/e-055-issue106-task-3-dose-response.md)
  以 seed 0、50-step schedule 比较 λ={0,1,2,4,8}，Gate D PASSED（6/6），提名 λ4/λ8。
  该单 seed、短程 nomination 不能外推到 100-update stability 或新 policy family。

**限制。** T1、T2、T3 分别回答可辨识性、有效更新、objective→behavior 方向，不能互相替代。
这些证据不证明真实车辆能耗、traffic 泛化或非降速节能；canonical control 是独立接受的选择，
不表示自动重定义所有 job。Main 整理时“E-048 原 record 未核到”的来源缺口已由本分支原记录补足。

## Interior policy 能否同时解决数值崩溃与行为饱和

**结论。** 旧 family 的 λ4/λ8 短程候选在多 seed 长程确认中失败。Interior 参数化消除了该
stress 矩阵中的数值崩溃，但 λ4/λ8 仍出现持续纵向饱和；数值稳定不等于可用候选。
新 family 下重新筛选后，仅 λ1 获得提名并通过后续确认。

**条件与证据。**

- [E-056：首次 multi-seed confirmation](../experiments/records/e-056-issue83-task-1-candidate-multiseed-confirmation.md)
  使用旧 family、100-update schedule、seeds 0/1/2，Gate M FAILED；λ4 无稳定 seed，
  λ8 seed 0 崩溃后其余 seeds 未运行。候选 final held-out 未测量，不能借 E-055 数字补齐。
- [ADR 0040](../adr/0040-beta-interior-concentration-parameterization.md) 改变 probability family
  和 initial hash；[E-057：interior stress](../experiments/records/e-057-issue114-task3-interior-stress.md)
  在 r0/λ4/λ8 × 三 seeds × 100 updates 下 9/9 机械稳定，Gate S2 PASSED；Gate S1 FAILED，
  λ8 boundary mass 超 ceiling，λ4/λ8 均持续负纵向饱和。它不证明所有配置均不会崩溃。
- [E-058：candidate revalidation](../experiments/records/e-058-issue114-task4-candidate-revalidation.md)
  在新 family 与同一 100-step schedule 下比较 λ={0,1,2,4,8}，复用 E-057 matched runs。
  λ1 唯一获提名，Gate S3 PASSED；λ2 因任务结构不一致落选，λ4/λ8 保留饱和否决。
- [E-059：λ1 confirmation re-entry](../experiments/records/e-059-issue83-task-1-reentry-lam1-multiseed-confirmation.md)
  复用 E-057 R0 与 E-058 λ1 的训练/评测证据，Gate M PASSED（7/7）；三个 seeds、S/SC
  family 和 leave-one-scenario-out 方向一致，50→100 updates 同号加深。没有新增训练，
  不应把 E-058/E-059 算作两次独立复现。

**限制。** 跨 family、50/100-step schedule 的 checkpoint 不可作 matched 数值比较。
Boundary mass 仅在 initial/final 测量；中间分布形态不能伪称直接 probe 证据。
λ1 的 proxy 改善仍伴随降速，distribution gates 不建立独立的节能机理。

## Final A0/A1/A2 比较支持什么

**结论。** [E-060：final matched comparison](../experiments/records/e-060-issue83-task-3-final-a0-a1-a2-matched-comparison.md)
的 A2−A1 Gate F PASSED（7/7）：λ1 相对 calibrated R0，在预注册任务/速度/安全 margins 内
降低 execution-trace fuel proxy 强度。该改善由降速载带，不是独立于降速机制的节能结论。
A1−A0 有稳定 learned behavioral effect，但 R0 相对 frozen planner 更快且 proxy 强度更高。

**条件与证据。** Canonical k=5、ADR 0040 interior family、E-052 control、同一 100-update
scheduler history、training seeds 0/1/2、no-traffic S/SC train seeds 0–7、held-out 16–23，
300 子步、DDIM5、runtime seed 760025、deterministic Beta mean。A1/A2 复用 E-057/E-058 的
matched artifacts；A0 在 `3bb59d9` 重跑，不使用 E-029 的历史 artifact 作当前基线。

A2−A1 的三个 seed final 强度下降 1.22%/1.30%/1.45%，scenario-bootstrap CI 不含零；
速度约下降 3.3%–3.7%，route completion 约下降 0.27%–0.43%。Arrival 保持 13/16，
collision/out-of-road/wrong-direction 均为零；48/48 评测产物可用，逐项 16/16 配对。
S/SC 与 leave-one-scenario-out 方向保持，但 S 效应幅度约为 SC 的 2.2–2.3 倍。

**限制。** CI 条件于固定 policy 和可用场景，不是跨 training seed 的总体区间。
Proxy 不是真实动力系统能耗，energy_distance 与 distance 是同一估计器，不能当独立证据。
A1 的 learned 分量须区分 initial guidance 偏移与 initial→final 变化。
结论仅适用本 λ1、reward family、cadence 与无交通矩阵，不外推其他 λ、traffic 或真实车辆。
是否存在不依赖降速的收益仍属 [reward/objective 研究问题](hypotheses/reward-objective.md)。
