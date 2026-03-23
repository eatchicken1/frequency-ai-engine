from app.agent_runtime.workflows.base import AgentWorkflow
from app.services.vibe_engine import VibeEngine


class MatchWorkflow(AgentWorkflow):
    name = "resonance_match"

    async def run(self, payload: dict):
        engine = VibeEngine()
        user_a = payload["user_a"]
        user_b = payload["user_b"]
        rounds = payload.get("rounds", 3)

        dialogue = await engine.simulate_conversation(
            user_a_profile=user_a,
            user_b_profile=user_b,
            rounds=rounds,
        )
        analysis = await engine.analyze_result(dialogue)
        return {
            "status": "success",
            "score": analysis.get("score", 0),
            "summary": analysis.get("summary", ""),
            "dialogue": dialogue,
        }
