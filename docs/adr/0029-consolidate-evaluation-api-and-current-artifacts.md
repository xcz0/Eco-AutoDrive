# Consolidate the evaluation API and current artifact contract

**Status:** Accepted and implemented
**Date:** 2026-09-02

ADR 0021 将 evaluation 压平为职责模块，但仓库消费者仍直接依赖多个子模块，在线 trace
recorder 与离线 schema/reader 也仍共享含 Torch 的导入路径。Issue 78 完成统一 engine、metrics
与 experiment ownership 后，这些路径成为剩余的组织负担。

因此，evaluation 以 `engine.py` 作为在线编排入口、`report.py` 作为通用离线报告入口，typed
I/O 位于 `io.py`。`trace.py` 只保留 NumPy 字段声明和结构校验，在线 recorder 独立位于
`recorder.py`。包根通过显式延迟导出提供仓库级 API，使 jobs、RL、benchmarking、experiments
和 scripts 不再依赖内部文件布局，同时保持离线 reader/report 不加载 Torch、MetaDrive 或
Panda3D。

当前科研代码只读取当前产物：JSON/NPZ 不保存全局 artifact 版本，也不提供历史兼容层。
Completed episode 将通用指标直接持久化为单一 `metrics` 对象，不再同时保存一组平铺副本。
结构校验只在 I/O 边界执行一次；在线受控数据流保留容量和生命周期检查，evaluation semantic
validation 继续负责 summary/trace、traffic、seed 和接口误差语义。

旧 `runner.py`、`analysis.py`、`artifacts.py`、`execution.py` 与 `summaries.py` 路径直接删除，
不保留转发模块。当前字段与执行行为由 `docs/agents/system-contract.md` 规定。
