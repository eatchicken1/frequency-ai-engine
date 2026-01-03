from app.core.logger import logger
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.schemas.knowledge import KnowledgeIngestResponse, KnowledgeDeleteRequest
from app.services.knowledge_engine import knowledge_engine
from app.services.vibe_engine import VibeEngine
from pydantic import BaseModel
from fastapi import HTTPException
from app.services.knowledge_trainer import train_from_oss
from app.schemas.KnowledgeTrainRequest import KnowledgeTrainRequest

# 1. 初始化 FastAPI 应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Frequency 社交平台 AI 核心引擎",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# 2. 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 请求参数 ---
class VibeCheckRequest(BaseModel):
    user_a: dict
    user_b: dict
    rounds: int = 3
    session_id: str = "default-session"  # 新增接收 Java 传来的 SessionID

@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"status": "online", "system": "Frequency AI Engine", "vibe": "Resonating"}

@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "UP", "service": settings.PROJECT_NAME}


# --- 核心接口 ---
@app.post(f"{settings.API_V1_STR}/ai/vibe-check")
async def start_vibe_check(request: VibeCheckRequest):
    """
    启动 AI 替身相亲局：对话 + 智能评价
    """
    engine = VibeEngine()

    try:
        logger.info(
            "🚀 开始同频测试: session_id={}, rounds={}, user_a={}, user_b={}",
            request.session_id,
            request.rounds,
            request.user_a.get("name"),
            request.user_b.get("name"),
        )

        # 1. 模拟对话
        dialogue = await engine.simulate_conversation(
            request.user_a,
            request.user_b,
            rounds=request.rounds
        )

        # 2. 智能分析 (这里返回的是 JSON 字典 {score, summary})
        analysis_result = await engine.analyze_result(dialogue)

        return {
            "status": "success",
            "score": analysis_result.get("score", 0),
            "summary": analysis_result.get("summary", "AI 正在思考人生..."),
            "dialogue": dialogue
        }

    except ValueError as e:
        logger.warning("Vibe check failed with value error: {}", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error executing Vibe Check: {}", e)
        raise HTTPException(status_code=500, detail="AI 服务内部错误")


@app.post("/ai/knowledge/train")
async def train_knowledge(request: KnowledgeTrainRequest):
    """
    文档级知识训练接口（业务系统调用）
    """
    try:
        return await train_from_oss(
            knowledge_id=request.knowledge_id,
            user_id=request.user_id,
            echo_id=request.echo_id,
            file_url=request.file_url,
            file_type=request.file_type,
            source_name=request.source_name,
        )
    except Exception as e:
        logger.exception("Knowledge train failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/knowledge/delete")
async def delete_knowledge_endpoint(request: KnowledgeDeleteRequest):
    """
    删除知识库文档对应的向量数据
    """
    try:
        logger.info(f"Delete request: knowledge_id={request.knowledge_id}, echo_id={request.echo_id}")
        return await knowledge_engine.delete(request)
    except Exception as e:
        logger.exception(f"Knowledge delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting Frequency AI Engine on port {}", settings.PORT)
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
