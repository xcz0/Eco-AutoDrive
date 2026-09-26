# Repository Guidelines

## 目标与完成条件

Eco-AutoDrive 是个人科研代码库。优先保证**逻辑正确、实验语义明确、结论边界可追溯**。
baseline、sampling、reward、metric、timing、termination、randomness 等研究定义的改变必须显式，不得作为普通重构顺带发生。

完成当前任务所需的最小正确改动，并做足以证明该改动正确的验证。
针对性验证通过、验收标准满足且没有新的失败证据时停止，不扩展到理论上的改进机会。

## 按问题读取权威来源

只读取当前任务需要的分篇和相关章节，不预读整套文档：

| 需要回答的问题 | 权威来源 |
| --- | --- |
| Semantics：概念是什么意思、解释边界是什么 | [Semantics](docs/research/semantics.md) |
| Protocol：baseline、sampling、guidance、正式 cadence、评测指标与 matched comparison | [Planning/evaluation](docs/research/protocols/planning-and-evaluation.md) |
| Protocol：优化对象、action sampling、reward、GAE/PPO 与训练统计 | [Training](docs/research/protocols/training.md) |
| Protocol：固定批次、credit attribution、人工干预与测量设计 | [Diagnostic studies](docs/research/protocols/diagnostic-studies.md) |
| Contract：shape、dtype、unit、坐标、padding 与模型 ABI | [Data/model](docs/contracts/data-and-model.md) |
| Contract：执行时序、配置/资源、simulator boundary、diffusion RNG 与所有权 | [Execution](docs/contracts/execution.md) |
| Contract：policy 概率空间、rollout、RNG、bootstrap 与 resume | [Training](docs/contracts/training.md) |
| Contract：artifact、provenance、reader、失败/缺失语义与 tracking | [Artifacts](docs/contracts/artifacts.md) |
| Implementation：当前实际怎样运行 | runtime observation、相关 code/tests、machine-readable config/依赖 |
| Decision：为什么选择某项长期设计 | 相关 [ADR](docs/adr/) |
| Evidence：某次真实运行观察到了什么 | 对应 [record](docs/experiments/records/) 与实际 resolved config/artifact；登记与检索见[实验 README](docs/experiments/README.md) |
| Finding：当前证据支持什么科研结论 | [Findings](docs/research/findings.md) 的相关主题及 supporting evidence |
| Hypothesis：尚未解决的研究问题与候选方法 | [Hypotheses](docs/research/hypotheses/)；主题导航见[研究 README](docs/research/README.md) |
| 当前任务、范围、验收与已接受待办 | 当前用户请求 + 对应 GitHub Issue |
| Procedure：怎样执行某类智能体工作 | 对应 [.agents/skills/](.agents/skills/) |
| 使用入口、依赖与可执行命令 | [README](README.md)、`justfile`、`pyproject.toml`、`uv.lock` |

**实际行为**由 observation、code/tests/config 回答；**规范行为**由 active Protocol + Contract 回答。
实现与 active 规范不一致是 **conformance mismatch**，应明确指出并调查，不能仅凭代码当前如此就判定规范过期，也不能自行融合成第三种解释。没有明确接受依据的要求不得因存在实现而成为 active。
Issue、Hypothesis 与外部参考资料不替代 active 规范或已运行证据；ADR 保存理由与历史决定。

## 执行边界与长期原则

* 回答、解释、评审、诊断或规划：检查相关材料并给出结论，不实现未请求的修改。
* 修改、修复或实现：直接完成范围内的本地修改及必要的非破坏性验证，无需逐步确认。
* GitHub 写入、外部发布、破坏性操作或明显扩大范围：仅在当前请求或明确工作流授权时执行。
* 优先从用户请求、相关规范、代码、配置和测试解决局部歧义。仅在缺失信息实质改变目标、科研语义或不可逆决策且无法可靠确定时，在最小阻塞点说明缺失项。
* 保持研究参数、随机性和执行条件显式；不以猜测、默认值或兼容回退替代缺失契约。不让机器资源变化静默改变实验定义。
* 保留原始失败和 undefined 状态；不以静默降级、宽泛异常捕获、坏样本跳过、伪造缺失值、隐式 clip/repair/fallback 或挑 seed 掩盖模型、数据和配置失效，除非研究方法明确规定。
* 优先使用成熟库；按语义所有权和依赖方向组织代码，复用机制但不合并不同研究语义。具体跨模块保证由 Contract 拥有，不为目录整齐或假想扩展预建框架。
* 没有明确要求或失败证据时，不新增抽象层、preflight、checksum、manifest、兼容层、冗余审计状态或常规重复实验；未完成工作不能用占位实现或文档声称已可用。

## Repository gotchas 与验证

* 使用 Windows PowerShell、Python 3.10 和已准备好的 `.venv`。优先使用 `justfile` 入口；不把 setup、环境/依赖探测当常规 preflight，不改用 `uv run`。
* `justfile` 是无语义 task alias；参数与默认值归对应 CLI/Hydra 配置。
* 测试按沙箱要求申请沙箱外执行，不重建环境或换入口规避权限。
* `third_party/metadrive/` 是运行时 editable 源码；`ref/` 是只读上游快照，业务代码不得从中导入。
* `checkpoints/`、`outputs/`、`.venv/`、`.env`、`ref/` 及被忽略的上游源码不得提交。

验证强度与语义风险和影响范围匹配，优先覆盖改变行为的最小验证。文档修改只检查链接、职责边界与权威来源一致性，不运行代码测试。bug 修复需回归证据；跨模块、仿真语义或实验方法变化须覆盖对应路径，只有结论依赖 baseline 才运行 baseline。依赖变化核对声明与锁文件一致。
仅有新失败、范围扩大或实质性未验证风险时扩大验证。命令与开发约定见 [README](README.md#开发与验证)。
软件测试不自动证明安全、节能、parity 或其他科研结论。

## 文档写回

只更新本次改变的知识类型，定义保留唯一自然语言 owner；代码实现、测试 assertion 和引用可并存：

* 概念或解释边界 → Semantics；研究方法、指标或比较设计 → 对应 Protocol。
* ABI、RNG、boundary、persistence 或 ownership guarantee → 对应 Contract。
* 仅局部实现变化 → code/tests/config，不维护自然语言实现镜像。
* 新长期设计理由 → ADR；现行要求仍引用 Protocol/Contract，不改写历史决定。
* 新真实实验 → Evidence record；新证据改变当前结论 → Findings，并引用具体 record 路径。
* 新未决研究问题 → Hypothesis；已接受但尚未完成的实施工作 → GitHub Issue。
* 重复工作方法 → Skill，只路由规范，不复制规范事实。

历史 records 原则上 append-only，不为当前导航清理改写旧参数、命令或锚点；必要的历史来源使用固定 Git revision。实验 README 只维护登记、provenance、检索和模板，不追加历史索引。

当 Issue 是任务来源时，按需读取目标、范围、non-goals、验收和影响执行的依赖；当前用户请求与 Issue 冲突时以当前请求为边界并指出差异，不擅自改 Issue。获授权的 Issue 更新只记录任务状态、阻塞与新验收信息，从 Issue 引用各权威文档；满足验收或明确取消后才关闭。验收依赖真实实验时，先登记运行证据再引用，不能用软件检查代替。
