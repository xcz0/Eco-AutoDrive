# Restructure evaluation artifacts and analysis

**Status:** Accepted and implemented
**Date:** 2026-08-14

Evaluation 的在线执行、artifact 持久化和离线分析曾位于同一平面包中。这样会模糊依赖方向：离线分析容易意外依赖 MetaDrive、Lightning 或模型 runtime，而在线代码也容易承担本应只属于实验分析的职责。

此外，已经在 artifact boundary 完成类型验证的数据，在分析端退化为普通 `dict` 后再次进行字段解释和校验，造成重复逻辑和不一致风险。

因此，将 evaluation 明确划分为三个方向不同的领域：

`runtime/` 负责在线运行和模型执行。

`artifacts/` 负责运行结果的持久化边界和 typed reader/writer，不依赖 MetaDrive、Lightning 或在线 episode runtime。

`analysis/` 负责离线 consumer，只依赖 typed artifacts、NumPy 以及分析所需的轻量组件，不参与在线 episode 执行。

离线 reader 返回经过验证的 typed models，而不是重新暴露未经约束的 JSON dictionary。这样，artifact boundary 是结构化数据验证的唯一入口，analysis 可以直接处理已经验证的数据。

Trace 的字段声明同时服务于 recording allocation 和持久化 validation，使数组名称、shape、dtype 及其基本约束拥有单一声明来源。项目特有的跨数组关系仍由项目代码验证，因此继续使用 NumPy NPZ，而不引入额外的通用 array-schema 框架。

一次性实验验收、阶段性 benchmark 或历史 migration 工具不属于稳定的 evaluation domain API。需要保留的实验记录应存在于实验文档或独立脚本中，而不是长期安装到运行时包。

该分层的主要目的不是规定某一版 artifact schema，而是建立稳定的依赖方向：

runtime → artifacts

analysis → artifacts

artifacts 不反向依赖 runtime 或 analysis。

完整的当前 artifact 字段、trace contract 和 reader 行为由 `docs/agents/system-contract.md` 描述。
