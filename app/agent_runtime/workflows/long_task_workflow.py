from app.agent_runtime.workflows.base import AgentWorkflow


class LongTaskWorkflow(AgentWorkflow):
    name = "long_task_orchestration"

    async def run(self, payload: dict):
        return {
            "status": "pending",
            "workflow": self.name,
            "checkpoint_key": payload.get("checkpoint_key"),
            "message": "Long task workflow scaffolded. Add checkpoint, retry, and HITL nodes here.",
        }
