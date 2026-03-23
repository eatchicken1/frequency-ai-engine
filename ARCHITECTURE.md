# Frequency Agent Runtime Skeleton

## Runtime split

- `app/services`: 保留底层能力组件，例如 `VibeEngine`、向量检索、文件解析。
- `app/agent_runtime`: 对外唯一入口，承接 LangGraph/长流程/多智能体工作流。
- `app/agent_runtime/workflows/match_workflow.py`: 对接当前的同频匹配能力。
- `app/agent_runtime/workflows/research_workflow.py`: 预留给深度研究/工具路由。
- `app/agent_runtime/workflows/long_task_workflow.py`: 预留给 checkpoint、人工审批和可恢复任务。

## Principle

- Python 只暴露 `/workflow/**`，不再直接承接普通聊天、RAG、知识训练接口。
- Java 主系统只调用 Python 的重型 workflow，不直接依赖其内部实现。
- 聊天、RAG、tool calling、知识库训练/检索由 Java AI 应用层负责。
