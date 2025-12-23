import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse

# 1. 初始化 FastAPI 应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Frequency 社交平台 AI 核心引擎",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# 2. 配置跨域资源共享 (CORS)
# 允许所有来源访问，方便开发调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 输出hello word
@app.get("/")
def read_root():
    return {"message": "Hello World from Frequency AI Engine!"}


# 3. 健康检查接口 (Health Check)
@app.get("/health")
def health_check():
    """
    健康检查，用于确认服务存活
    """
    return {
        "status": "UP",
        "service": settings.PROJECT_NAME,
        "mode": settings.ENV_MODE
    }


# 4. 核心测试接口
@app.post(f"{settings.API_V1_STR}/chat/test", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    """
    [测试接口] 验证 Java -> Python 的调用链路
    目前仅打印日志并返回模拟数据。
    """
    print(f"\n[AI Engine] 收到请求 | Session: {request.session_id}")
    print(f"[AI Engine] 用户发送: {request.user_message}")

    # 模拟 AI 处理逻辑
    reply_text = f"【AI Engine】收到你的消息：'{request.user_message}'。Python 环境运行正常！"

    return ChatResponse(
        response_text=reply_text,
        is_finished=True,
        data={
            "source": "python-engine",
            "echo_id": request.echo_id,
            "status": "simulated"
        }
    )


# 5. 启动入口
if __name__ == "__main__":
    # 使用 uvicorn 启动服务，开启热重载(reload)方便开发
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)