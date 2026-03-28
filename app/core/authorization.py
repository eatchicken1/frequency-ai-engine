from __future__ import annotations

from typing import Iterable

from app.core.exceptions import RuntimeAuthError
from app.core.security import CallerIdentity


# 企业级最小授权矩阵：默认拒绝，仅显式放行
WORKFLOW_ROLE_RULES = {
    "resonance_match": {"ROLE_ADMIN", "ROLE_USER", "AI_RUNTIME_MATCH", "AI_RUNTIME_ALL"},
    "campus_research": {"ROLE_ADMIN", "AI_RUNTIME_RESEARCH", "AI_RUNTIME_ALL"},
    "long_task_orchestration": {"ROLE_ADMIN", "AI_RUNTIME_LONG_TASK", "AI_RUNTIME_ALL"},
}


def authorize_workflow_call(workflow_name: str, identity: CallerIdentity, payload: dict) -> str:
    """
    基于角色 + 资源归属进行授权，返回 role_bucket 供监控打点。
    """
    normalized_roles = _normalize_roles(identity.roles)
    role_bucket = _to_role_bucket(normalized_roles)
    allowed_roles = WORKFLOW_ROLE_RULES.get(workflow_name, {"ROLE_ADMIN", "AI_RUNTIME_ALL"})
    if normalized_roles.isdisjoint(allowed_roles):
        raise RuntimeAuthError(f"forbidden: role is not allowed for workflow={workflow_name}")

    # 针对用户态 workflow 增加 owner 校验，防止跨用户代跑
    if workflow_name in {"resonance_match", "campus_research"}:
        _assert_owner(identity=identity, payload=payload)

    return role_bucket


def _assert_owner(identity: CallerIdentity, payload: dict) -> None:
    if not identity.user_id:
        raise RuntimeAuthError("forbidden: user identity missing")
    owner_fields = (
        payload.get("current_user_id"),
        payload.get("user_id"),
        payload.get("owner_user_id"),
        payload.get("initiator_id"),
    )
    owner_values = {str(value).strip() for value in owner_fields if value is not None and str(value).strip()}
    if owner_values and str(identity.user_id) not in owner_values:
        raise RuntimeAuthError("forbidden: payload owner mismatch")


def _normalize_roles(raw_roles: str | None) -> set[str]:
    if not raw_roles:
        return set()
    items = [item.strip() for item in raw_roles.replace("|", ",").split(",")]
    normalized = {item.upper() for item in items if item}
    # 兼容 Java 侧历史 roleCode（admin/common）
    alias = set()
    if "ADMIN" in normalized:
        alias.add("ROLE_ADMIN")
    if "COMMON" in normalized or "USER" in normalized:
        alias.add("ROLE_USER")
    return normalized | alias


def _to_role_bucket(roles: Iterable[str]) -> str:
    role_set = set(roles)
    if "ROLE_ADMIN" in role_set:
        return "admin"
    if "ROLE_USER" in role_set:
        return "user"
    if any(item.startswith("AI_RUNTIME_") for item in role_set):
        return "runtime"
    return "unknown"
