from app.agent_runtime.workflows.base import AgentWorkflow


class ResearchWorkflow(AgentWorkflow):
    name = "campus_research"

    async def run(self, payload: dict):
        query = payload.get("query", "")
        return {
            "status": "pending",
            "workflow": self.name,
            "query": query,
            "message": "Research workflow scaffolded. Connect retrieval, tool routing, and human approval here.",
        }
