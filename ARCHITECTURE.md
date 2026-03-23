# Frequency AI Engine Skeleton

## Runtime split

- `app/services`: 继续承接当前 HTTP 接口直接调用的能力。
- `app/agent_runtime`: 预留给 LangGraph/长流程/多智能体工作流。
- `app/agent_runtime/workflows/match_workflow.py`: 对接当前的同频匹配能力。
- `app/agent_runtime/workflows/research_workflow.py`: 预留给深度研究/工具路由。
- `app/agent_runtime/workflows/long_task_workflow.py`: 预留给 checkpoint、人工审批和可恢复任务。

## Principle

- 现有 `FastAPI` 接口先不打断。
- 先把 runtime 和 service 拆开，再迁具体能力。
- Java 主系统只调用 Python 的重型 workflow，不直接依赖其内部实现。
