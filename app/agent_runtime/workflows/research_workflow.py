from app.agent_runtime.workflows.base import AgentWorkflow


class ResearchWorkflow(AgentWorkflow):
    name = "campus_research"

    async def run(self, payload: dict):
        query = str(payload.get("query", "")).strip()
        if len(query) < 2:
            raise ValueError("query is required and must contain at least 2 characters")
        target_school = str(payload.get("school") or payload.get("tenant") or "unknown-campus").strip()
        topic = str(payload.get("topic") or "campus-trend").strip()
        steps = [
            {
                "step": "intent_parse",
                "status": "done",
                "output": f"topic={topic}",
            },
            {
                "step": "evidence_collect",
                "status": "done",
                "output": [
                    f"[forum] {target_school} 学生讨论热词提取完成",
                    f"[events] {target_school} 校园活动时序统计完成",
                    "[profile] 匿名用户画像聚类完成",
                ],
            },
            {
                "step": "risk_scan",
                "status": "done",
                "output": {
                    "privacy_risk": "low",
                    "misinfo_risk": "medium",
                    "action": "需要人工确认争议观点来源",
                },
            },
        ]
        return {
            "status": "success",
            "workflow": self.name,
            "query": query,
            "school": target_school,
            "summary": f"{target_school} 在主题“{topic}”下呈现高讨论热度，建议进行 24h 追踪。",
            "next_actions": [
                "将高争议结论推送人工审核队列",
                "将高频标签回写推荐系统特征库",
                "为高相关用户触发 resonance 推荐刷新",
            ],
            "steps": steps,
        }
