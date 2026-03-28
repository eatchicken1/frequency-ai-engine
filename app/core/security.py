from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Request

from app.core.config import settings
from app.core.exceptions import RuntimeAuthError
from app.core.nonce_store import NonceStore
from app.core.runtime_metrics import RUNTIME_AUTH_ATTEMPTS_TOTAL, RUNTIME_AUTH_DURATION_SECONDS


HEADER_CLIENT_ID = "x-frequency-client-id"
HEADER_TIMESTAMP = "x-frequency-timestamp"
HEADER_NONCE = "x-frequency-nonce"
HEADER_SIGNATURE = "x-frequency-signature"
HEADER_USER_ID = "x-frequency-user-id"
HEADER_USER_NAME = "x-frequency-user-name"
HEADER_USER_ROLES = "x-frequency-user-roles"
HEADER_SESSION_ID = "x-frequency-session-id"
HEADER_FROM = "from"
FROM_IN = "Y"


@dataclass
class CallerIdentity:
    client_id: str
    user_id: Optional[str]
    username: Optional[str]
    roles: Optional[str]
    session_id: Optional[str]


def assert_runtime_secret_configured() -> None:
    """
    生产默认强制要求配置共享密钥；开发可显式降级。
    """
    if settings.RUNTIME_ALLOW_INSECURE:
        return
    if not settings.RUNTIME_SHARED_SECRET:
        raise RuntimeAuthError("RUNTIME_SHARED_SECRET is required when insecure mode is disabled")


def _body_hash_base64(raw_body: bytes) -> str:
    digest = hashlib.sha256(raw_body or b"").digest()
    return base64.b64encode(digest).decode("utf-8")


def _canonical_string(
    *,
    client_id: str,
    timestamp: str,
    nonce: str,
    method: str,
    path: str,
    body_hash: str,
    user_id: str,
    username: str,
    roles: str,
    session_id: str,
) -> str:
    return "\n".join(
        [
            client_id,
            timestamp,
            nonce,
            method.upper(),
            path,
            body_hash,
            user_id,
            username,
            roles,
            session_id,
        ]
    )


def _sign_base64(message: str, secret: str) -> str:
    signature = hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(signature).decode("utf-8")


def verify_runtime_request(request: Request, raw_body: bytes, nonce_store: NonceStore) -> CallerIdentity:
    """
    验证 Java -> Python runtime 的服务间调用签名、时效和来源身份。
    """
    started = time.perf_counter()
    client_id = request.headers.get(HEADER_CLIENT_ID, "").strip() or "unknown"
    try:
        timestamp = request.headers.get(HEADER_TIMESTAMP, "").strip()
        nonce = request.headers.get(HEADER_NONCE, "").strip()
        signature = request.headers.get(HEADER_SIGNATURE, "").strip()

        if not timestamp or not nonce or not signature or client_id == "unknown":
            raise RuntimeAuthError("missing required runtime auth headers")
        if request.headers.get(HEADER_FROM, "") != FROM_IN:
            raise RuntimeAuthError("missing internal from header")

        if settings.RUNTIME_ALLOWED_CLIENTS:
            allowed = {item.strip() for item in settings.RUNTIME_ALLOWED_CLIENTS.split(",") if item.strip()}
            if client_id not in allowed:
                raise RuntimeAuthError("caller client is not allowed")

        try:
            ts_value = int(timestamp)
        except ValueError as exc:
            raise RuntimeAuthError("invalid timestamp header") from exc

        now = int(time.time())
        skew = settings.RUNTIME_CLOCK_SKEW_SECONDS
        if abs(now - ts_value) > skew:
            raise RuntimeAuthError("request timestamp is outside allowed clock skew")
        _assert_nonce_fresh(client_id=client_id, nonce=nonce, ttl_seconds=skew, nonce_store=nonce_store)

        body_hash = _body_hash_base64(raw_body)
        user_id = request.headers.get(HEADER_USER_ID, "").strip()
        username = request.headers.get(HEADER_USER_NAME, "").strip()
        roles = request.headers.get(HEADER_USER_ROLES, "").strip()
        session_id = request.headers.get(HEADER_SESSION_ID, "").strip()
        canonical = _canonical_string(
            client_id=client_id,
            timestamp=timestamp,
            nonce=nonce,
            method=request.method,
            path=request.url.path,
            body_hash=body_hash,
            user_id=user_id,
            username=username,
            roles=roles,
            session_id=session_id,
        )
        expected = _sign_base64(canonical, settings.RUNTIME_SHARED_SECRET)
        if not hmac.compare_digest(expected, signature):
            raise RuntimeAuthError("invalid runtime signature")

        RUNTIME_AUTH_ATTEMPTS_TOTAL.labels(client_id, "ok", "none").inc()
        return CallerIdentity(
            client_id=client_id,
            user_id=user_id or None,
            username=username or None,
            roles=roles or None,
            session_id=session_id or None,
        )
    except RuntimeAuthError as exc:
        reason = _normalize_reason(str(exc))
        RUNTIME_AUTH_ATTEMPTS_TOTAL.labels(client_id, "rejected", reason).inc()
        raise
    except Exception:
        RUNTIME_AUTH_ATTEMPTS_TOTAL.labels(client_id, "error", "internal").inc()
        raise
    finally:
        RUNTIME_AUTH_DURATION_SECONDS.observe(time.perf_counter() - started)


def _assert_nonce_fresh(*, client_id: str, nonce: str, ttl_seconds: int, nonce_store: NonceStore) -> None:
    # 按 client_id 分桶：不同调用方 nonce 空间相互隔离
    key = f"{client_id}:{nonce}"
    inserted = nonce_store.set_if_absent(key=key, ttl_seconds=ttl_seconds)
    if not inserted:
        raise RuntimeAuthError("replayed nonce detected")


def _normalize_reason(message: str) -> str:
    text = (message or "").strip().lower()
    if "replayed nonce" in text:
        return "nonce_replay"
    if "signature" in text:
        return "bad_signature"
    if "timestamp" in text:
        return "timestamp_skew"
    if "missing required runtime auth headers" in text:
        return "missing_headers"
    if "missing internal from header" in text:
        return "missing_from_header"
    if "caller client is not allowed" in text:
        return "client_not_allowed"
    if "nonce store unavailable" in text:
        return "nonce_store_unavailable"
    return "other"
