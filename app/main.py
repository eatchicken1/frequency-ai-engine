from app.agent_runtime.runtime import build_default_runtime
from app.core.config import settings
from app.core.logger import logger
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 1. 初始化 FastAPI 应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Frequency 重型 Agent Runtime",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

agent_runtime = build_default_runtime()

# 2. 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"status": "online", "system": "Frequency Agent Runtime", "mode": "workflow-only"}

@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "UP", "service": settings.PROJECT_NAME}

@app.get("/workflow/list")
def list_workflows():
    return {"status": "success", "items": agent_runtime.list()}


@app.post("/workflow/{name}/run")
async def run_workflow(name: str, payload: dict):
    try:
        workflow = agent_runtime.get(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"workflow not found: {name}")

    try:
        logger.info("Running workflow: name={}, payload_keys={}", name, list(payload.keys()))
        return await workflow.run(payload)
    except ValueError as e:
        logger.warning("Workflow execution failed: name={}, error={}", name, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Workflow execution error: name={}, error={}", name, e)
        raise HTTPException(status_code=500, detail="workflow runtime error")


if __name__ == "__main__":
    logger.info("Starting Frequency Agent Runtime on port {}", settings.PORT)
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
