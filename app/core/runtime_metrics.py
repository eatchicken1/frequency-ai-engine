from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response


RUNTIME_AUTH_ATTEMPTS_TOTAL = Counter(
    "frequency_runtime_auth_attempts_total",
    "Total number of runtime auth verification attempts",
    ["client_id", "result", "reason"],
)

RUNTIME_AUTH_DURATION_SECONDS = Histogram(
    "frequency_runtime_auth_duration_seconds",
    "Runtime auth verification latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

RUNTIME_NONCE_STORE_TOTAL = Counter(
    "frequency_runtime_nonce_store_total",
    "Nonce store operations by store type and result",
    ["store_type", "result"],
)

RUNTIME_WORKFLOW_RUNS_TOTAL = Counter(
    "frequency_runtime_workflow_runs_total",
    "Workflow run requests by workflow, result and caller role bucket",
    ["workflow", "result", "role_bucket"],
)

RUNTIME_WORKFLOW_DURATION_SECONDS = Histogram(
    "frequency_runtime_workflow_duration_seconds",
    "Workflow execution latency in seconds",
    ["workflow"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
