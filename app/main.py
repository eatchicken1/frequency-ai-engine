from app.agent_runtime.runtime import build_default_runtime
from app.core.authorization import authorize_workflow_call
from app.core.config import settings
from app.core.exceptions import RuntimeAuthError
from app.core.logger import logger
from app.core.nonce_store import build_nonce_store
from app.core.runtime_metrics import (
    RUNTIME_WORKFLOW_DURATION_SECONDS,
    RUNTIME_WORKFLOW_RUNS_TOTAL,
    metrics_response,
)
from app.core.security import assert_runtime_secret_configured, verify_runtime_request
import time
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# 1. 初始化 FastAPI 应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Frequency 重型 Agent Runtime",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

agent_runtime = build_default_runtime()
assert_runtime_secret_configured()
nonce_store = build_nonce_store()
nonce_store.startup_check()

# 2. 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def runtime_auth_middleware(request, call_next):
    # 仅对 workflow 入口强制服务间鉴权，health/root 保持可探活
    if request.url.path.startswith("/workflow/"):
        try:
            body = await request.body()
            identity = verify_runtime_request(request, body, nonce_store)
            request.state.caller_identity = identity
        except RuntimeAuthError as exc:
            logger.warning("Runtime auth rejected: path={}, reason={}", request.url.path, exc)
            return JSONResponse(status_code=401, content={"detail": "unauthorized runtime request"})
        except Exception as exc:
            logger.exception("Runtime auth internal error: path={}, error={}", request.url.path, exc)
            return JSONResponse(status_code=500, content={"detail": "runtime auth verification error"})
    return await call_next(request)


@app.on_event("shutdown")
def on_shutdown():
    nonce_store.close()


@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"status": "online", "system": "Frequency Agent Runtime", "mode": "workflow-only"}

@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "UP", "service": settings.PROJECT_NAME}


@app.get("/metrics")
def metrics():
    return metrics_response()


@app.get("/workflow/list")
def list_workflows():
    return {"status": "success", "items": agent_runtime.list()}


@app.post("/workflow/{name}/run")
async def run_workflow(name: str, payload: dict, request: Request):
    caller = getattr(request.state, "caller_identity", None)
    role_bucket = "anonymous"
    if caller:
        try:
            role_bucket = authorize_workflow_call(name, caller, payload or {})
        except RuntimeAuthError as exc:
            logger.warning(
                "Runtime authorization rejected: workflow={}, client_id={}, reason={}",
                name,
                caller.client_id,
                exc,
            )
            RUNTIME_WORKFLOW_RUNS_TOTAL.labels(name, "forbidden", "unknown").inc()
            raise HTTPException(status_code=403, detail="forbidden workflow request")
        logger.info(
            "Runtime authorized call: client_id={}, user_id={}, username={}, roles={}, session_id={}",
            caller.client_id,
            caller.user_id,
            caller.username,
            caller.roles,
            caller.session_id,
        )
    try:
        workflow = agent_runtime.get(name)
    except KeyError:
        RUNTIME_WORKFLOW_RUNS_TOTAL.labels(name, "not_found", role_bucket).inc()
        raise HTTPException(status_code=404, detail=f"workflow not found: {name}")

    started = time.perf_counter()
    try:
        logger.info("Running workflow: name={}, payload_keys={}", name, list(payload.keys()))
        result = await workflow.run(payload)
        RUNTIME_WORKFLOW_RUNS_TOTAL.labels(name, "success", role_bucket).inc()
        return result
    except ValueError as e:
        logger.warning("Workflow execution failed: name={}, error={}", name, e)
        RUNTIME_WORKFLOW_RUNS_TOTAL.labels(name, "bad_request", role_bucket).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Workflow execution error: name={}, error={}", name, e)
        RUNTIME_WORKFLOW_RUNS_TOTAL.labels(name, "error", role_bucket).inc()
        raise HTTPException(status_code=500, detail="workflow runtime error")
    finally:
        RUNTIME_WORKFLOW_DURATION_SECONDS.labels(name).observe(time.perf_counter() - started)


if __name__ == "__main__":
    logger.info("Starting Frequency Agent Runtime on port {}", settings.PORT)
    try:
        uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
    finally:
        nonce_store.close()
