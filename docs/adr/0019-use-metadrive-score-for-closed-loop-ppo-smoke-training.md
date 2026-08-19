# Use the MetaDrive score for closed-loop PPO smoke training

在引入项目最终研究 reward 之前，需要先验证 Exploration Policy、MetaDrive closed loop、rollout collection、GAE、PPO update 和 checkpoint 路径能够组成一个真实环境中的端到端训练循环。

因此，closed-loop PPO smoke training 使用 MetaDrive 的 `metadrive_builtin_v1` score 作为唯一优化 reward。

选择该 reward 的目的不是把 MetaDrive 内置分数定义为项目最终目标，而是避免在验证 PPO 基础设施的同时引入新的 reward-model 不确定性。Smoke run 首先回答的是：

“环境 reward 能否经过完整 rollout 和 PPO 数据路径稳定地更新 Exploration Policy？”

而不是：

“该 reward 是否已经代表安全、能耗或 PlannerRFT parity？”

MetaDrive reward 的组成仍应通过独立 audit 数据解释。Dense step reward、terminal reward change、route progress、distance、speed、停止状态、termination 和 execution failure 等信息需要保留为可审计指标，但除明确配置为 reward 的部分外，不应静默加入 PPO objective。

因此，MetaDrive score 上升本身不能被解释为：

- 安全性改善；
- 能耗降低；
- PlannerRFT reward parity；
- 最终研究目标改善。

最初的 smoke experiment 可以使用较小的固定配置，例如两个串行 logical environment slots、每个 slot 每次 update 收集 16 个 10 Hz transitions，并执行少量 PPO updates。这样的 `2 × 16 × 4` 等规模属于实验 profile，只用于使 smoke test 快速、可重复和易于检查。

Environment 数量、rollout 长度、update 数量、minibatch 划分以及后续并行规模均由训练配置决定，不是 PPO framework invariant。

Diffusion noise generator 和 policy action generator 保持相互独立，并从训练 seed 确定性派生，使 planner sampling randomness 与 Exploration Policy sampling randomness 可以分别审计。

预训练 Diffusion Planner 在该训练路径中保持冻结；PPO optimizer 只更新 Exploration Policy。训练前后应能够验证 planner 参数未被修改。

Policy export checkpoint 和 resumable training-state checkpoint 使用不同语义：

- policy export 用于保存和重新加载 Exploration Policy 本身；
- training-state checkpoint 用于继续一次中断的训练，并额外保存 optimizer、scheduler、RNG 和 loop progress 等运行状态。

Policy export 中的格式版本只描述该 checkpoint 文件自身的结构，不是 RL artifact 或 evaluation artifact 的全局 schema 版本。

训练输出与 evaluation 输出保持独立的数据边界。完整的当前 rollout 字段、checkpoint 内容、summary 和 failure contract 由 `docs/agents/system-contract.md` 定义，而不在本 ADR 中复制。
