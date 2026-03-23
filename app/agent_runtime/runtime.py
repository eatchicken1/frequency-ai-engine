from app.agent_runtime.workflows.long_task_workflow import LongTaskWorkflow
from app.agent_runtime.workflows.match_workflow import MatchWorkflow
from app.agent_runtime.workflows.research_workflow import ResearchWorkflow


class AgentRuntime:
    """
    负责管理 Python 侧的重型 agent workflow。
    这里先只做注册与分发，不影响现有 HTTP 接口。
    """

    def __init__(self) -> None:
        self._workflows = {}

    def register(self, workflow) -> None:
        self._workflows[workflow.name] = workflow

    def get(self, name: str):
        return self._workflows[name]

    def list(self):
        return list(self._workflows.keys())


def build_default_runtime() -> AgentRuntime:
    runtime = AgentRuntime()
    runtime.register(MatchWorkflow())
    runtime.register(ResearchWorkflow())
    runtime.register(LongTaskWorkflow())
    return runtime
