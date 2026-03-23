class AgentWorkflow:
    """
    所有 Python agent workflow 的基类。
    """

    name = "base"

    async def run(self, payload: dict):
        raise NotImplementedError("workflow must implement run(payload)")
