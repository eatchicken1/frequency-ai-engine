import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.knowledge import KnowledgeIngestResponse, KnowledgeIngestRequest
from app.services.knowledge_engine import knowledge_engine
from app.services.vibe_engine import VibeEngine

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
    return {"status": "online", "system": "Frequency AI Engine", "vibe": "Resonating"}

@app.get("/health")
def health_check():
    return {"status": "UP", "service": settings.PROJECT_NAME}


# --- 核心接口 ---
@app.post(f"{settings.API_V1_STR}/ai/vibe-check")
async def start_vibe_check(request: VibeCheckRequest):
    """
    启动 AI 替身相亲局：对话 + 智能评价
    """
    engine = VibeEngine()

    try:
        print(f"🚀 开始同频测试 Session: {request.session_id}")

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
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Error executing Vibe Check: {e}")
        raise HTTPException(status_code=500, detail="AI 服务内部错误")

@app.post("/knowledge/add", response_model=KnowledgeIngestResponse)
async def ingest_knowledge_endpoint(request: KnowledgeIngestRequest):
    """
    接收数字分身的记忆切片，并存入 Redis 向量数据库
    """
    try:
        return await knowledge_engine.ingest(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
