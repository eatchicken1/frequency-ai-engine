from __future__ import annotations

import threading
from typing import Protocol

from redis import Redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.exceptions import RuntimeAuthError
from app.core.logger import logger
from app.core.runtime_metrics import RUNTIME_NONCE_STORE_TOTAL


class NonceStore(Protocol):
    def set_if_absent(self, key: str, ttl_seconds: int) -> bool:
        pass

    def startup_check(self) -> None:
        pass

    def close(self) -> None:
        pass


class InMemoryNonceStore:
    store_type = "memory"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, int] = {}

    def set_if_absent(self, key: str, ttl_seconds: int) -> bool:
        import time

        now = int(time.time())
        expire_before = now - ttl_seconds
        with self._lock:
            stale_keys = [item for item, ts in self._store.items() if ts < expire_before]
            for item in stale_keys:
                self._store.pop(item, None)
            if key in self._store:
                RUNTIME_NONCE_STORE_TOTAL.labels(self.store_type, "duplicate").inc()
                return False
            self._store[key] = now
            RUNTIME_NONCE_STORE_TOTAL.labels(self.store_type, "success").inc()
            return True

    def startup_check(self) -> None:
        logger.warning("Runtime nonce store is using in-memory mode. This is not suitable for multi-instance production.")

    def close(self) -> None:
        self._store.clear()


class RedisNonceStore:
    store_type = "redis"

    def __init__(self, redis_url: str, key_prefix: str) -> None:
        self._key_prefix = key_prefix
        self._client = Redis.from_url(redis_url, decode_responses=True)

    def set_if_absent(self, key: str, ttl_seconds: int) -> bool:
        redis_key = f"{self._key_prefix}{key}"
        try:
            # SET key value NX EX ttl
            success = bool(self._client.set(redis_key, "1", ex=ttl_seconds, nx=True))
            if success:
                RUNTIME_NONCE_STORE_TOTAL.labels(self.store_type, "success").inc()
            else:
                RUNTIME_NONCE_STORE_TOTAL.labels(self.store_type, "duplicate").inc()
            return success
        except RedisError as exc:
            RUNTIME_NONCE_STORE_TOTAL.labels(self.store_type, "error").inc()
            raise RuntimeAuthError(f"nonce store unavailable: {exc}") from exc

    def startup_check(self) -> None:
        try:
            pong = self._client.ping()
            if not pong:
                raise RuntimeAuthError("redis ping returned false")
            logger.info("Runtime nonce store connected: redis")
        except RedisError as exc:
            raise RuntimeAuthError(f"failed to connect redis nonce store: {exc}") from exc

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            # shutdown path, swallow to avoid masking main exception
            pass


def build_nonce_store() -> NonceStore:
    mode = (settings.RUNTIME_NONCE_STORE or "redis").strip().lower()
    if mode == "redis":
        return RedisNonceStore(redis_url=settings.REDIS_URL, key_prefix=settings.RUNTIME_NONCE_REDIS_PREFIX)
    if mode == "memory":
        return InMemoryNonceStore()
    raise RuntimeAuthError(f"unsupported runtime nonce store mode: {mode}")
