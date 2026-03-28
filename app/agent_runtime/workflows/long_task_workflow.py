from app.agent_runtime.workflows.base import AgentWorkflow


class LongTaskWorkflow(AgentWorkflow):
    name = "long_task_orchestration"

    async def run(self, payload: dict):
        task_id = str(payload.get("task_id") or payload.get("checkpoint_key") or "").strip()
        if not task_id:
            raise ValueError("task_id or checkpoint_key is required")
        checkpoint = str(payload.get("checkpoint_key") or f"cp:{task_id}").strip()
        phase = str(payload.get("phase") or "execute").strip()
        retry_count = int(payload.get("retry_count") or 0)
        max_retry = int(payload.get("max_retry") or 3)
        next_phase = "review" if phase == "execute" else "completed"
        return {
            "status": "success",
            "workflow": self.name,
            "task_id": task_id,
            "checkpoint_key": checkpoint,
            "phase": phase,
            "next_phase": next_phase,
            "retry": {
                "current": retry_count,
                "max": max_retry,
                "remaining": max(0, max_retry - retry_count),
            },
            "hitl": {
                "required": next_phase == "review",
                "reason": "阶段切换到 review 前需人工确认",
            },
            "message": "Long task progressed with checkpoint persisted contract.",
        }
